"""
Real-time energy-node simulator.

SIMULATION MODE - every value produced here is synthetic. It emulates what an
ESP32/STM32 node fitted with an isolated voltage/current front-end would
publish, so that the dashboard, the database and the anomaly detector can be
exercised without hardware.

The simulator reuses the same physics as the historical dataset:

    P = base_load * hour_factor * peak_factor * noise
    I = P / (V * PF)
    E = P * interval_hours / 1000

Determinism: pass a ``seed`` and the sequence is reproducible, which is what
the test-suite relies on.

Run directly:
    python -m src.iot.simulator
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from src.data.generate_data import HOURLY_PROFILE
from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatorSettings:
    """Tunable behaviour of the simulated node."""

    device_id: str = config.DEVICE_ID
    interval_seconds: float = config.SIMULATOR_INTERVAL_SECONDS
    base_load_w: float = 900.0
    noise_sigma: float = 0.10          # multiplicative noise on power
    peak_hours: tuple[int, ...] = (18, 19, 20, 21)
    peak_multiplier: float = 1.45
    anomaly_probability: float = 0.03  # chance of an abnormal sample
    # Simulated-time step per reading. Using one hour keeps the generated
    # stream compatible with the hourly feature pipeline; set to e.g. 1/60 for
    # minute-resolution demos.
    time_step_minutes: int = config.SAMPLING_INTERVAL_MINUTES


@dataclass
class EnergySimulator:
    """
    Generates realistic synthetic energy readings.

    Parameters
    ----------
    settings : SimulatorSettings, optional
    seed : int, optional
        Fixes the random stream for reproducible tests.
    start_time : datetime, optional
        Virtual clock start. Defaults to "24 hours ago" so a fresh run
        immediately has a day of context.
    """

    settings: SimulatorSettings = field(default_factory=SimulatorSettings)
    seed: int | None = None
    start_time: datetime | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._clock = self.start_time or (datetime.now() - timedelta(hours=24))
        self._n_generated = 0

    # ------------------------------------------------------------ internals
    def _load_multiplier(self, hour: int) -> float:
        multiplier = HOURLY_PROFILE[hour % 24]
        if hour in self.settings.peak_hours:
            multiplier *= self.settings.peak_multiplier
        return multiplier

    def _advance_clock(self) -> datetime:
        now = self._clock
        self._clock = self._clock + timedelta(minutes=self.settings.time_step_minutes)
        return now

    # -------------------------------------------------------------- reading
    def generate_reading(self, timestamp: datetime | None = None) -> dict:
        """
        Produce one reading as a JSON-serialisable dict.

        Keys match the MQTT contract plus the derived quantities:
        device_id, timestamp, voltage, current, power_factor, frequency,
        active_power, energy, mode.
        """
        ts = timestamp or self._advance_clock()
        s = self.settings

        power = s.base_load_w * self._load_multiplier(ts.hour)
        power *= float(self._rng.lognormal(0.0, s.noise_sigma))

        # Occasional abnormal episode so the anomaly page has something to show.
        if self._rng.random() < s.anomaly_probability:
            power *= float(self._rng.uniform(2.5, 3.5)) if self._rng.random() < 0.6 else 0.15

        power = float(np.clip(power, 40.0, 6000.0))
        relative_load = power / s.base_load_w

        voltage = float(
            np.clip(
                config.NOMINAL_VOLTAGE - 3.5 * (relative_load - 1.0)
                + self._rng.normal(0.0, 1.8),
                200.0, 250.0,
            )
        )
        power_factor = float(
            np.clip(0.80 + 0.14 * min(relative_load, 1.6) / 1.6
                    + self._rng.normal(0.0, 0.01), 0.70, 0.99)
        )
        frequency = float(
            np.clip(config.NOMINAL_FREQUENCY + self._rng.normal(0.0, 0.035), 49.5, 50.5)
        )
        current = power / (voltage * power_factor)
        energy = power * (s.time_step_minutes / 60.0) / 1000.0

        self._n_generated += 1
        return {
            "device_id": s.device_id,
            "timestamp": ts.isoformat(timespec="seconds"),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "power_factor": round(power_factor, 3),
            "frequency": round(frequency, 3),
            "active_power": round(power, 2),
            "energy": round(energy, 5),
            "mode": "SIMULATION",
        }

    def generate_batch(self, n: int) -> list[dict]:
        """Generate ``n`` consecutive readings (no real-time delay)."""
        return [self.generate_reading() for _ in range(n)]

    def stream(self, n: int | None = None, realtime: bool = True) -> Iterator[dict]:
        """
        Yield readings indefinitely (or ``n`` of them).

        With ``realtime=True`` the generator sleeps ``interval_seconds``
        between readings, emulating a live sensor feed.
        """
        emitted = 0
        while n is None or emitted < n:
            yield self.generate_reading()
            emitted += 1
            if realtime and (n is None or emitted < n):
                time.sleep(self.settings.interval_seconds)

    @property
    def readings_generated(self) -> int:
        """How many readings this instance has produced."""
        return self._n_generated


def main() -> None:
    print("=== SIMULATION MODE - synthetic data, no hardware attached ===\n")
    simulator = EnergySimulator(seed=config.RANDOM_SEED)
    header = f"{'timestamp':<20}{'V':>8}{'I':>8}{'PF':>7}{'f':>8}{'P (W)':>10}{'E (kWh)':>10}"
    print(header)
    print("-" * len(header))
    for reading in simulator.stream(n=8, realtime=False):
        print(
            f"{reading['timestamp']:<20}{reading['voltage']:>8.2f}{reading['current']:>8.3f}"
            f"{reading['power_factor']:>7.3f}{reading['frequency']:>8.3f}"
            f"{reading['active_power']:>10.2f}{reading['energy']:>10.4f}"
        )
    print(f"\nReadings generated: {simulator.readings_generated}")

    # Determinism check
    a = EnergySimulator(seed=7, start_time=datetime(2025, 1, 1)).generate_batch(3)
    b = EnergySimulator(seed=7, start_time=datetime(2025, 1, 1)).generate_batch(3)
    print(f"Deterministic with a fixed seed: {a == b}")


if __name__ == "__main__":
    main()
