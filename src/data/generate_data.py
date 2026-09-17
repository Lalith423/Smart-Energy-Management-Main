"""
Synthetic development dataset generator.

IMPORTANT
---------
Everything produced by this module is a **synthetic development dataset**.
It is generated from a mathematical load model, not measured with a real
energy meter. It exists so that the data pipeline, the machine-learning
models and the dashboard can be developed and tested without hardware.

Load model
----------
For each hourly timestamp the generator builds an *active power* target from
four multiplicative components and then derives the electrical quantities
that a real meter would report:

    P(t) = base_load * hour_factor(t) * day_factor(t) * season_factor(t) * noise

    I    = P / (V * PF)                (single phase)
    E    = P * interval_hours / 1000   (kWh per sample)

Daily shape (hour_factor): night low -> morning rise -> moderate afternoon
-> evening peak -> night decay, which is the classic domestic load curve.

Run directly:
    python -m src.data.generate_data
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)

# Relative load multiplier for each hour of the day (index 0..23).
# Night trough ~0.35, morning shoulder ~0.9, afternoon ~0.8, evening peak ~1.6.
HOURLY_PROFILE: tuple[float, ...] = (
    0.40, 0.35, 0.32, 0.30, 0.32, 0.45,   # 00:00 - 05:00  night
    0.70, 0.95, 1.00, 0.85, 0.75, 0.72,   # 06:00 - 11:00  morning
    0.80, 0.78, 0.72, 0.70, 0.80, 1.00,   # 12:00 - 17:00  afternoon
    1.35, 1.60, 1.55, 1.25, 0.90, 0.60,   # 18:00 - 23:00  evening peak
)

BASE_LOAD_W: float = 900.0      # mean household active power, watts
WEEKEND_FACTOR: float = 1.12    # people stay home -> slightly higher usage


def _season_factor(day_of_year: np.ndarray) -> np.ndarray:
    """Yearly seasonality: hotter months (cooling load) draw more power."""
    # Peak around day 150 (end of May, Indian summer), trough in winter.
    return 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 60) / 365.0)


def _power_factor_from_load(relative_load: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Power factor improves as the load increases.

    Light load is dominated by inductive stand-by devices (poor PF); at heavy
    load the resistive share grows and PF rises towards ~0.97.
    """
    pf = 0.80 + 0.14 * np.clip(relative_load, 0.0, 1.6) / 1.6
    pf = pf + rng.normal(0.0, 0.010, size=pf.shape)
    return np.clip(pf, 0.70, 0.99)


def _inject_abnormal_events(
    power: np.ndarray, rng: np.random.Generator, n_events: int
) -> np.ndarray:
    """
    Insert a small number of abnormal consumption episodes.

    These are *not* labelled in the output file. They exist so the Isolation
    Forest has genuinely unusual patterns to find. Two kinds are injected:
    sudden surges (e.g. a large appliance switched on at an odd hour) and
    sudden collapses (e.g. a feeder trip).
    """
    power = power.copy()
    if n_events <= 0 or power.size < 48:
        return power

    # Candidate start positions: never inside the first day (the lag warm-up
    # window) and never so late that the episode runs past the end.
    candidates = np.arange(24, power.size - 6)
    n_events = min(n_events, candidates.size)
    if n_events <= 0:
        return power

    starts = rng.choice(candidates, size=n_events, replace=False)
    for start in starts:
        length = int(rng.integers(2, 5))
        if rng.random() < 0.65:
            power[start : start + length] *= rng.uniform(2.4, 3.4)   # surge
        else:
            power[start : start + length] *= rng.uniform(0.08, 0.25)  # collapse
    return power


def generate_energy_data(
    start: str = "2025-01-01",
    end: str = "2025-09-30 23:00:00",
    freq_minutes: int | None = None,
    seed: int = config.RANDOM_SEED,
    n_abnormal_events: int = 40,
) -> pd.DataFrame:
    """
    Generate a synthetic development dataset.

    Parameters
    ----------
    start, end : str
        Inclusive timestamp range.
    freq_minutes : int, optional
        Sampling interval. Defaults to ``config.SAMPLING_INTERVAL_MINUTES``.
    seed : int
        Seed for the random generator - the dataset is fully reproducible.
    n_abnormal_events : int
        Number of abnormal consumption episodes to inject (unlabelled).

    Returns
    -------
    pandas.DataFrame
        Columns: timestamp, voltage, current, power_factor, frequency,
        active_power, energy.
    """
    freq_minutes = freq_minutes or config.SAMPLING_INTERVAL_MINUTES
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    n = len(timestamps)
    logger.info("Generating %d synthetic samples (%s -> %s)", n, timestamps[0], timestamps[-1])

    hours = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    hour_factor = np.asarray(HOURLY_PROFILE, dtype=float)[hours]
    day_factor = np.where(day_of_week >= 5, WEEKEND_FACTOR, 1.0)
    season = _season_factor(day_of_year)

    # Multiplicative log-normal noise keeps power strictly positive.
    noise = rng.lognormal(mean=0.0, sigma=0.12, size=n)

    active_power = BASE_LOAD_W * hour_factor * day_factor * season * noise
    active_power = _inject_abnormal_events(active_power, rng, n_abnormal_events)
    active_power = np.clip(active_power, 40.0, 6000.0)

    # Supply voltage sags slightly when the local load is high.
    relative_load = active_power / BASE_LOAD_W
    voltage = (
        config.NOMINAL_VOLTAGE
        - 3.5 * (relative_load - 1.0)
        + rng.normal(0.0, 1.8, size=n)
    )
    voltage = np.clip(voltage, 200.0, 250.0)

    power_factor = _power_factor_from_load(relative_load, rng)

    frequency = config.NOMINAL_FREQUENCY + rng.normal(0.0, 0.035, size=n)
    frequency = np.clip(frequency, 49.5, 50.5)

    # Current is derived, not invented:  I = P / (V * PF)
    current = active_power / (voltage * power_factor)

    # Energy per sample:  E[kWh] = P[W] * interval[h] / 1000
    interval_hours = freq_minutes / 60.0
    energy = active_power * interval_hours / 1000.0

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "voltage": np.round(voltage, 2),
            "current": np.round(current, 3),
            "power_factor": np.round(power_factor, 3),
            "frequency": np.round(frequency, 3),
            "active_power": np.round(active_power, 2),
            "energy": np.round(energy, 5),
        }
    )
    return df


def save_dataset(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the dataset to CSV, creating parent folders as needed."""
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def summarise(df: pd.DataFrame) -> str:
    """Human-readable summary used by the CLI and the notebooks."""
    lines = [
        f"Rows           : {len(df):,}",
        f"Time range     : {df['timestamp'].min()}  ->  {df['timestamp'].max()}",
        f"Columns        : {', '.join(df.columns)}",
        f"Missing values : {int(df.isna().sum().sum())}",
        f"Duplicate ts   : {int(df['timestamp'].duplicated().sum())}",
        "",
        "Numeric ranges (min / mean / max):",
    ]
    for col in config.NUMERIC_COLUMNS:
        lines.append(
            f"  {col:<13} {df[col].min():>10.3f} {df[col].mean():>10.3f} {df[col].max():>10.3f}"
        )
    total_kwh = float(df["energy"].sum())
    days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
    lines += [
        "",
        f"Total energy   : {total_kwh:,.2f} kWh over {days} days",
        f"Average daily  : {total_kwh / days:,.2f} kWh/day",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: generate the raw dataset plus a small sample file."""
    config.ensure_directories()
    df = generate_energy_data()
    save_dataset(df, config.RAW_DATA_PATH)

    # A committed 500-row sample so the repository is useful straight after clone.
    sample = df.head(500)
    save_dataset(sample, config.SAMPLE_DATA_PATH)

    print("\n=== SYNTHETIC DEVELOPMENT DATASET ===")
    print(f"Generated at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(summarise(df))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
