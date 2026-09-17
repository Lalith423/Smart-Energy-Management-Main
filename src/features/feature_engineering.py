"""
Time-series feature engineering.

Leakage rule
------------
Every feature must be computable *at prediction time*, using only information
that already exists when the prediction is made.

* Calendar features (hour, day, month, ...) are known in advance -> safe.
* Lag features use ``shift(k)`` with ``k >= 1`` -> only past values.
* Rolling statistics are computed on the **shifted** series
  (``shift(1).rolling(w)``), so the current sample never contributes to its
  own rolling mean. Computing ``rolling(w)`` directly on the target would
  leak the target into its own feature - a classic and easy-to-miss bug.

The first ``max(lag)`` rows have undefined lags and are dropped by
``build_training_frame``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add calendar features derived from the timestamp (no leakage possible).

    Cyclical encoding
    -----------------
    ``hour`` and ``month`` are also encoded as sine/cosine pairs. Two reasons:

    1. Hour 23 and hour 0 are one hour apart in reality but 23 units apart as
       raw integers. The sin/cos pair restores that adjacency.
    2. A decision tree can only split on values it saw while training. The test
       month (September) never appears in the training months (Jan-Aug), so a
       raw ``month`` integer of 9 falls outside the learned range and the tree
       has to extrapolate - which tree ensembles cannot do. The sin/cos values
       for September lie *inside* the range already seen, so the model
       interpolates instead. A raw ``day_of_year`` column has the same problem
       and is deliberately not used.
    """
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])

    out["hour"] = ts.dt.hour.astype("int16")
    out["day"] = ts.dt.day.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["day_of_week"] = ts.dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COLUMN,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add ``lag_k`` columns: the target value k sampling intervals earlier.

    With hourly sampling, ``lag_1`` is one hour ago and ``lag_24`` is the same
    hour on the previous day - the strongest predictor in a daily load curve.
    """
    lags = lags or config.LAG_PERIODS
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COLUMN,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean/std of the target over past windows only.

    ``shift(1)`` is applied *before* ``rolling`` so the window ends at the
    previous sample. Window 3 captures the short-term trend; window 24 gives
    the previous day's average level.
    """
    windows = windows or config.ROLLING_WINDOWS
    out = df.copy()
    past = out[target].shift(1)
    for window in windows:
        out[f"rolling_mean_{window}"] = past.rolling(window=window, min_periods=window).mean()
        out[f"rolling_std_{window}"] = past.rolling(window=window, min_periods=window).std()
    return out


def create_features(
    df: pd.DataFrame,
    target: str = config.TARGET_COLUMN,
    dropna: bool = True,
) -> pd.DataFrame:
    """Apply every feature step in order. Rows with undefined lags are dropped."""
    out = df.sort_values("timestamp").reset_index(drop=True)
    out = add_time_features(out)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)

    if dropna:
        before = len(out)
        out = out.dropna().reset_index(drop=True)
        logger.info("Dropped %d warm-up rows with undefined lag/rolling values", before - len(out))
    return out


def get_feature_columns(df: pd.DataFrame, target: str = config.TARGET_COLUMN) -> list[str]:
    """
    Return the model input columns.

    Excluded on purpose:
      * ``timestamp``  - not numeric, already encoded as calendar features
      * ``active_power`` (target)
      * ``energy``     - a linear transform of the target (E = P*dt/1000);
                         including it would hand the model the answer
      * ``voltage`` / ``current`` / ``power_factor`` / ``frequency`` -
                         these are measured *at the same instant* as the target
                         and satisfy P = V*I*PF exactly, so they are unavailable
                         when forecasting a future hour. Excluding them is the
                         single most important leakage decision in this project.
    """
    excluded = {
        "timestamp",
        target,
        "energy",
        "voltage",
        "current",
        "power_factor",
        "frequency",
        "anomaly",
        "device_id",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_training_frame(
    df: pd.DataFrame, target: str = config.TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """
    Convenience wrapper used by training and evaluation.

    Returns
    -------
    X : pandas.DataFrame       feature matrix
    y : pandas.Series          target vector
    feature_names : list[str]  ordered feature column names
    timestamps : pandas.Series aligned timestamps (for chronological splitting)
    """
    featured = create_features(df, target=target)
    feature_names = get_feature_columns(featured, target=target)
    return featured[feature_names], featured[target], feature_names, featured["timestamp"]


def build_prediction_row(history: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Build a one-row feature frame for a future ``timestamp``.

    ``history`` must be ordered and contain the target column for the samples
    immediately preceding ``timestamp``. Only past target values are touched,
    so this mirrors exactly what the model saw during training.
    """
    target = config.TARGET_COLUMN
    series = history[target].reset_index(drop=True)
    max_lag = max(config.LAG_PERIODS + config.ROLLING_WINDOWS)
    if len(series) < max_lag:
        raise ValueError(
            f"Need at least {max_lag} historical samples to build features, got {len(series)}"
        )

    ts = pd.Timestamp(timestamp)
    row: dict[str, float] = {
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24.0)),
        "month_sin": float(np.sin(2 * np.pi * ts.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * ts.month / 12.0)),
    }
    for lag in config.LAG_PERIODS:
        row[f"lag_{lag}"] = float(series.iloc[-lag])
    for window in config.ROLLING_WINDOWS:
        window_values = series.iloc[-window:]
        row[f"rolling_mean_{window}"] = float(window_values.mean())
        row[f"rolling_std_{window}"] = float(window_values.std(ddof=1))
    return pd.DataFrame([row])
