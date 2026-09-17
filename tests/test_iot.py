"""Tests for the IoT layer: the simulator and the optional MQTT client."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.iot.mqtt_client import (
    MQTTEnergyClient,
    PayloadError,
    get_mqtt_status,
    mqtt_library_available,
    validate_payload,
)
from src.iot.simulator import EnergySimulator, SimulatorSettings
from src.utils import config


# ----------------------------------------------------------------- simulator
def test_reading_contains_every_expected_field():
    reading = EnergySimulator(seed=1).generate_reading()
    for key in ["device_id", "timestamp", "voltage", "current", "power_factor",
                "frequency", "active_power", "energy", "mode"]:
        assert key in reading


def test_simulated_values_are_physically_plausible():
    for reading in EnergySimulator(seed=2).generate_batch(200):
        for field in ["voltage", "current", "power_factor", "frequency", "active_power"]:
            low, high = config.VALID_RANGES[field]
            assert low <= reading[field] <= high, f"{field} = {reading[field]}"


def test_power_equation_holds_for_simulated_readings():
    for reading in EnergySimulator(seed=3).generate_batch(50):
        expected = reading["voltage"] * reading["current"] * reading["power_factor"]
        assert reading["active_power"] == pytest.approx(expected, rel=0.01)


def test_energy_uses_the_configured_interval():
    settings = SimulatorSettings(time_step_minutes=60)
    reading = EnergySimulator(settings=settings, seed=4).generate_reading()
    assert reading["energy"] == pytest.approx(reading["active_power"] / 1000.0, rel=1e-6)


def test_simulator_is_deterministic_with_a_seed():
    start = datetime(2025, 1, 1)
    a = EnergySimulator(seed=11, start_time=start).generate_batch(10)
    b = EnergySimulator(seed=11, start_time=start).generate_batch(10)
    assert a == b


def test_different_seeds_produce_different_streams():
    start = datetime(2025, 1, 1)
    a = EnergySimulator(seed=1, start_time=start).generate_batch(10)
    b = EnergySimulator(seed=2, start_time=start).generate_batch(10)
    assert a != b


def test_timestamps_advance_by_the_configured_step():
    simulator = EnergySimulator(seed=5, start_time=datetime(2025, 1, 1))
    readings = simulator.generate_batch(3)
    times = [datetime.fromisoformat(r["timestamp"]) for r in readings]
    step_minutes = simulator.settings.time_step_minutes
    assert (times[1] - times[0]).total_seconds() == step_minutes * 60
    assert (times[2] - times[1]).total_seconds() == step_minutes * 60


def test_peak_hours_draw_more_power_than_night_hours():
    simulator = EnergySimulator(seed=6, start_time=datetime(2025, 1, 1))
    day = simulator.generate_batch(24)
    by_hour = {datetime.fromisoformat(r["timestamp"]).hour: r["active_power"] for r in day}
    assert by_hour[20] > by_hour[3]


def test_reading_counter_and_mode_label():
    simulator = EnergySimulator(seed=7)
    simulator.generate_batch(5)
    assert simulator.readings_generated == 5
    assert simulator.generate_reading()["mode"] == "SIMULATION"


def test_stream_yields_the_requested_number_without_sleeping():
    readings = list(EnergySimulator(seed=8).stream(n=4, realtime=False))
    assert len(readings) == 4


# ---------------------------------------------------------- payload handling
def test_valid_payload_is_accepted_and_derived_fields_computed():
    payload = {
        "device_id": "ENERGY_NODE_01",
        "timestamp": "2026-09-17T10:30:00",
        "voltage": 230.5,
        "current": 2.8,
        "power_factor": 0.94,
        "frequency": 50.0,
    }
    reading = validate_payload(json.dumps(payload))
    assert reading["device_id"] == "ENERGY_NODE_01"
    assert reading["active_power"] == pytest.approx(230.5 * 2.8 * 0.94)
    assert reading["energy"] == pytest.approx(
        reading["active_power"] * config.sampling_interval_hours() / 1000.0
    )
    assert reading["mode"] == "MQTT"


def test_dict_payloads_are_accepted_too():
    reading = validate_payload(
        {"voltage": 230.0, "current": 2.0, "power_factor": 0.9, "frequency": 50.0}
    )
    assert reading["active_power"] == pytest.approx(414.0)


def test_malformed_json_is_rejected():
    with pytest.raises(PayloadError):
        validate_payload("{not json")


def test_missing_fields_are_rejected():
    with pytest.raises(PayloadError):
        validate_payload('{"voltage": 230.0}')


def test_non_numeric_fields_are_rejected():
    with pytest.raises(PayloadError):
        validate_payload(
            '{"voltage": "high", "current": 2.0, "power_factor": 0.9, "frequency": 50.0}'
        )


def test_implausible_values_are_rejected():
    with pytest.raises(PayloadError):
        validate_payload(
            '{"voltage": 900.0, "current": 2.0, "power_factor": 0.9, "frequency": 50.0}'
        )


def test_explicit_power_from_the_node_is_preserved():
    reading = validate_payload(
        {"voltage": 230.0, "current": 2.0, "power_factor": 0.9,
         "frequency": 50.0, "active_power": 400.0}
    )
    assert reading["active_power"] == pytest.approx(400.0)


def test_timestamp_defaults_to_now_when_absent():
    reading = validate_payload(
        {"voltage": 230.0, "current": 2.0, "power_factor": 0.9, "frequency": 50.0}
    )
    assert datetime.fromisoformat(reading["timestamp"])


# -------------------------------------------------- graceful MQTT degradation
def test_connect_returns_false_without_a_broker_instead_of_raising():
    client = MQTTEnergyClient(broker="", port=1883, topic="energy/data")
    assert client.connect(timeout=1.0) is False
    assert client.status.connected is False
    assert client.status.last_error


def test_disconnect_is_safe_when_never_connected():
    client = MQTTEnergyClient(broker="")
    client.disconnect()
    client.disconnect()


def test_context_manager_never_raises_when_offline():
    with MQTTEnergyClient(broker="") as client:
        assert client.status.connected is False


def test_publish_is_a_no_op_when_offline():
    assert MQTTEnergyClient(broker="").publish({"voltage": 230}) is False


def test_status_label_is_human_readable():
    status = get_mqtt_status()
    assert isinstance(status.library_available, bool)
    assert status.label in {
        "Connected", "Offline", "Not configured",
        "Unavailable (paho-mqtt not installed)",
    }


def test_library_availability_probe_returns_a_bool():
    assert isinstance(mqtt_library_available(), bool)


def test_unreachable_broker_does_not_crash_the_app():
    """The dashboard must survive a broker that is configured but down."""
    client = MQTTEnergyClient(broker="192.0.2.1", port=1883)  # TEST-NET-1, unroutable
    assert client.connect(timeout=1.0) is False
    client.disconnect()
