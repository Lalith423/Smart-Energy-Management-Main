"""
Optional MQTT ingestion layer.

The whole point of this module is that it is **optional**. If paho-mqtt is not
installed, or no broker is reachable, every entry point degrades gracefully and
the dashboard keeps running in simulation mode. Nothing here ever raises into
the UI thread.

Expected payload published by an ESP32/STM32 node
-------------------------------------------------
    {
      "device_id": "ENERGY_NODE_01",
      "timestamp": "2026-09-17T10:30:00",
      "voltage": 230.5,
      "current": 2.8,
      "power_factor": 0.94,
      "frequency": 50.0
    }

``active_power`` and ``energy`` may be omitted - the backend derives them with
P = V*I*PF and E = P*dt/1000, so a constrained MCU can send the bare minimum.

Compatibility: supports both paho-mqtt 1.x and the 2.x callback API.

Run directly (prints broker status, never hangs):
    python -m src.iot.mqtt_client
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ("voltage", "current", "power_factor", "frequency")


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------
def mqtt_library_available() -> bool:
    """True if paho-mqtt can be imported."""
    try:
        import paho.mqtt.client  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Payload validation
# --------------------------------------------------------------------------
class PayloadError(ValueError):
    """Raised when an incoming MQTT payload is malformed or implausible."""


def validate_payload(raw: str | bytes | dict) -> dict:
    """
    Parse and validate one sensor message.

    Checks JSON syntax, required fields, numeric types and physical plausibility
    against ``config.VALID_RANGES``. Returns a normalised reading dict with
    ``active_power`` and ``energy`` filled in.

    Raises
    ------
    PayloadError
        With a human-readable reason. Callers log it and drop the message
        rather than crashing the subscriber loop.
    """
    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PayloadError(f"invalid JSON: {exc}") from exc
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        raise PayloadError(f"unsupported payload type: {type(raw).__name__}")

    if not isinstance(payload, dict):
        raise PayloadError("payload is not a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise PayloadError(f"missing fields: {missing}")

    reading: dict = {}
    for field_name in REQUIRED_FIELDS:
        try:
            reading[field_name] = float(payload[field_name])
        except (TypeError, ValueError) as exc:
            raise PayloadError(f"field '{field_name}' is not numeric") from exc

    for field_name, value in reading.items():
        low, high = config.VALID_RANGES[field_name]
        if not low <= value <= high:
            raise PayloadError(
                f"field '{field_name}' = {value} outside plausible range [{low}, {high}]"
            )

    reading["device_id"] = str(payload.get("device_id", config.DEVICE_ID))
    reading["timestamp"] = str(
        payload.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    )

    active_power = payload.get("active_power")
    reading["active_power"] = (
        float(active_power)
        if active_power is not None
        else reading["voltage"] * reading["current"] * reading["power_factor"]
    )

    energy = payload.get("energy")
    reading["energy"] = (
        float(energy)
        if energy is not None
        else reading["active_power"] * config.sampling_interval_hours() / 1000.0
    )
    reading["mode"] = "MQTT"
    return reading


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
@dataclass
class MQTTStatus:
    """Snapshot of the connection, rendered by the dashboard health panel."""

    library_available: bool = False
    configured: bool = False
    connected: bool = False
    broker: str = ""
    port: int = 0
    topic: str = ""
    messages_received: int = 0
    messages_rejected: int = 0
    last_error: str = ""
    last_message_at: str = ""

    @property
    def label(self) -> str:
        """Short status string: Connected / Offline / Not configured / Unavailable."""
        if not self.library_available:
            return "Unavailable (paho-mqtt not installed)"
        if not self.configured:
            return "Not configured"
        return "Connected" if self.connected else "Offline"


class MQTTEnergyClient:
    """
    Thin, defensive wrapper around ``paho.mqtt.client.Client``.

    Usage::

        client = MQTTEnergyClient(on_reading=store_reading)
        if client.connect():           # returns False instead of raising
            ...                        # messages arrive on a background thread
        client.disconnect()
    """

    def __init__(
        self,
        broker: str | None = None,
        port: int | None = None,
        topic: str | None = None,
        on_reading: Callable[[dict], None] | None = None,
        client_id: str = "smart-energy-backend",
    ) -> None:
        self.broker = broker if broker is not None else config.MQTT_BROKER
        self.port = int(port if port is not None else config.MQTT_PORT)
        self.topic = topic if topic is not None else config.MQTT_TOPIC
        self.on_reading = on_reading
        self.client_id = client_id

        self._client = None
        self._lock = threading.Lock()
        self.status = MQTTStatus(
            library_available=mqtt_library_available(),
            configured=bool(self.broker),
            broker=self.broker,
            port=self.port,
            topic=self.topic,
        )

    # ----------------------------------------------------------- callbacks
    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        ok = (reason_code == 0) or (getattr(reason_code, "is_failure", None) is False)
        self.status.connected = bool(ok)
        if ok:
            client.subscribe(self.topic)
            logger.info("Connected to %s:%s, subscribed to '%s'",
                        self.broker, self.port, self.topic)
        else:
            self.status.last_error = f"connect failed (reason code {reason_code})"
            logger.warning(self.status.last_error)

    def _on_disconnect(self, client, userdata, *args) -> None:
        self.status.connected = False
        logger.info("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, message) -> None:
        """Validate and forward one message. Never lets an exception escape."""
        try:
            reading = validate_payload(message.payload)
        except PayloadError as exc:
            with self._lock:
                self.status.messages_rejected += 1
                self.status.last_error = str(exc)
            logger.warning("Rejected MQTT payload: %s", exc)
            return

        with self._lock:
            self.status.messages_received += 1
            self.status.last_message_at = datetime.now().isoformat(timespec="seconds")

        if self.on_reading is not None:
            try:
                self.on_reading(reading)
            except Exception as exc:  # pragma: no cover - user callback failure
                logger.exception("on_reading callback raised: %s", exc)

    # ------------------------------------------------------------ lifecycle
    def _build_client(self):
        """Create a paho client that works on both the 1.x and 2.x APIs."""
        import paho.mqtt.client as mqtt

        try:  # paho-mqtt >= 2.0
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        except (AttributeError, TypeError):  # paho-mqtt 1.x
            client = mqtt.Client(client_id=self.client_id)

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        # Exponential-ish reconnect backoff handled by paho itself.
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        return client

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Try to connect. Returns True on success, False on any failure.

        Never raises: a missing library, an unset broker or an unreachable host
        are all normal conditions for this project.
        """
        if not self.status.library_available:
            self.status.last_error = "paho-mqtt is not installed"
            logger.info("MQTT disabled - %s", self.status.last_error)
            return False
        if not self.broker:
            self.status.last_error = "MQTT_BROKER is not set in .env"
            logger.info("MQTT disabled - %s", self.status.last_error)
            return False

        try:
            self._client = self._build_client()
            self._client.connect(self.broker, self.port, keepalive=config.MQTT_KEEPALIVE)
            self._client.loop_start()
        except Exception as exc:
            self.status.connected = False
            self.status.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("MQTT connection failed - continuing offline (%s)",
                           self.status.last_error)
            self._client = None
            return False

        # Wait briefly for the CONNACK so callers get a truthful answer.
        waited = 0.0
        while waited < timeout and not self.status.connected:
            threading.Event().wait(0.1)
            waited += 0.1
        return self.status.connected

    def publish(self, reading: dict, topic: str | None = None) -> bool:
        """Publish a reading as JSON (used for local broker testing)."""
        if self._client is None or not self.status.connected:
            return False
        try:
            self._client.publish(topic or self.topic, json.dumps(reading))
            return True
        except Exception as exc:  # pragma: no cover
            self.status.last_error = str(exc)
            return False

    def disconnect(self) -> None:
        """Stop the network loop and close the socket. Safe to call twice."""
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.warning("Error during MQTT shutdown: %s", exc)
        finally:
            self._client = None
            self.status.connected = False
            logger.info("MQTT client shut down cleanly")

    def __enter__(self) -> "MQTTEnergyClient":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()


def get_mqtt_status() -> MQTTStatus:
    """Cheap status probe for the dashboard - does not open a connection."""
    return MQTTStatus(
        library_available=mqtt_library_available(),
        configured=bool(config.MQTT_BROKER),
        connected=False,
        broker=config.MQTT_BROKER,
        port=config.MQTT_PORT,
        topic=config.MQTT_TOPIC,
    )


def main() -> None:
    print("=== MQTT LAYER CHECK ===")
    print(f"paho-mqtt installed : {mqtt_library_available()}")
    print(f"Broker configured   : {config.MQTT_BROKER or '(none - set MQTT_BROKER in .env)'}")
    print(f"Topic               : {config.MQTT_TOPIC}")

    sample = {
        "device_id": "ENERGY_NODE_01",
        "timestamp": "2026-09-17T10:30:00",
        "voltage": 230.5,
        "current": 2.8,
        "power_factor": 0.94,
        "frequency": 50.0,
    }
    reading = validate_payload(json.dumps(sample))
    print(f"\nValid payload accepted -> P = {reading['active_power']:.2f} W, "
          f"E = {reading['energy']:.4f} kWh")

    for bad in ['{"voltage": 230}', "not json at all", '{"voltage": 999, "current": 2,'
                ' "power_factor": 0.9, "frequency": 50}']:
        try:
            validate_payload(bad)
        except PayloadError as exc:
            print(f"Rejected as expected   -> {exc}")

    client = MQTTEnergyClient()
    connected = client.connect(timeout=2.0)
    print(f"\nConnection attempt  : {'connected' if connected else 'offline'}")
    print(f"Status label        : {client.status.label}")
    print("Application continues normally without MQTT.")
    client.disconnect()


if __name__ == "__main__":
    main()
