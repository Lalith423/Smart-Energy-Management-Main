"""
Central configuration for the Smart Energy Monitoring system.

All paths, electrical constants and environment-driven settings live here so
that no other module has to hard-code a path or a magic number.

Values can be overridden through a `.env` file in the project root
(see `.env.example`). Nothing secret is ever hard-coded in source.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:  # python-dotenv is a declared dependency, but the app must not die without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit in a broken install
    pass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# src/utils/config.py -> src/utils -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_DATA_PATH: Path = RAW_DATA_DIR / "energy_data.csv"
PROCESSED_DATA_PATH: Path = PROCESSED_DATA_DIR / "energy_processed.csv"
SAMPLE_DATA_PATH: Path = SAMPLE_DATA_DIR / "energy_sample.csv"

MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best_model.pkl"))
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH: Path = MODELS_DIR / "model_comparison.csv"
ANOMALY_MODEL_PATH: Path = MODELS_DIR / "anomaly_model.pkl"

DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", DATA_DIR / "energy.db"))


# --------------------------------------------------------------------------
# Electrical constants and assumptions
# --------------------------------------------------------------------------
# Single-phase domestic supply (Indian grid nominal values).
NOMINAL_VOLTAGE: float = 230.0          # volts (RMS)
NOMINAL_FREQUENCY: float = 50.0         # hertz

# The historical dataset is sampled once per hour. Energy for one sample is
# therefore:  E[kWh] = P[W] * (SAMPLING_INTERVAL_MINUTES / 60) / 1000
SAMPLING_INTERVAL_MINUTES: int = 60

# Plausibility limits used by the data-validation layer. Readings outside these
# bands are reported (never silently deleted) as physically implausible.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "voltage": (180.0, 270.0),        # wide brown-out / over-voltage band
    "current": (0.0, 40.0),           # single-phase domestic feeder
    "power_factor": (0.30, 1.0),
    "frequency": (47.0, 53.0),        # grid frequency excursion band
    "active_power": (0.0, 10_000.0),  # watts
    "energy": (0.0, 10.0),            # kWh per sampling interval
}

REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "voltage",
    "current",
    "power_factor",
    "frequency",
    "active_power",
    "energy",
]

NUMERIC_COLUMNS: list[str] = [c for c in REQUIRED_COLUMNS if c != "timestamp"]


# --------------------------------------------------------------------------
# Machine-learning settings
# --------------------------------------------------------------------------
TARGET_COLUMN: str = "active_power"

# Chronological split: everything before this timestamp trains the model,
# everything from it onwards is the untouched test period.
# (Shipped dataset: Jan-Aug 2025 -> training, Sep 2025 -> testing.)
TRAIN_TEST_SPLIT_DATE: str = "2025-09-01"

# Model selection criterion. Documented in the README: the model with the
# lowest test RMSE wins, because RMSE penalises the large errors that matter
# most for peak-load planning.
MODEL_SELECTION_METRIC: str = "rmse"

RANDOM_SEED: int = 42

LAG_PERIODS: list[int] = [1, 2, 3, 24]
ROLLING_WINDOWS: list[int] = [3, 24]

ANOMALY_FEATURES: list[str] = ["active_power", "current", "voltage", "power_factor"]
ANOMALY_CONTAMINATION: float = 0.02  # expected share of abnormal samples


# --------------------------------------------------------------------------
# IoT / MQTT settings (optional - the app runs fine without a broker)
# --------------------------------------------------------------------------
MQTT_BROKER: str = os.getenv("MQTT_BROKER", "")
MQTT_PORT: int = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC: str = os.getenv("MQTT_TOPIC", "energy/data")
MQTT_KEEPALIVE: int = int(os.getenv("MQTT_KEEPALIVE", "60"))
DEVICE_ID: str = os.getenv("DEVICE_ID", "ENERGY_NODE_01")

SIMULATOR_INTERVAL_SECONDS: float = float(os.getenv("SIMULATOR_INTERVAL_SECONDS", "2"))


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------
DEFAULT_TARIFF_PER_KWH: float = float(os.getenv("TARIFF_PER_KWH", "8.0"))
CURRENCY_SYMBOL: str = os.getenv("CURRENCY_SYMBOL", "Rs.")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger


def ensure_directories() -> None:
    """Create every directory the pipeline writes to, if missing."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        DATABASE_PATH.parent,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)


def sampling_interval_hours() -> float:
    """Sampling interval expressed in hours (used for energy integration)."""
    return SAMPLING_INTERVAL_MINUTES / 60.0
