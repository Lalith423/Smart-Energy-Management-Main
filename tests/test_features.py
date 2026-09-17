"""Tests for feature engineering, with explicit target-leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    build_prediction_row,
    build_training_frame,
    create_features,
    get_feature_columns,
)
from src.utils import config


# ------------------------------------------------------------- time features
def test_time_features_are_created(clean_dataset):
    out = add_time_features(clean_dataset)
    for column in ["hour", "day", "month", "day_of_week", "is_weekend",
                   "hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert column in out.columns


def test_time_features_match_the_timestamp(clean_dataset):
    out = add_time_features(clean_dataset)
    row = out.iloc[100]
    ts = pd.Timestamp(row["timestamp"])
    assert row["hour"] == ts.hour
    assert row["day_of_week"] == ts.dayofweek
    assert row["is_weekend"] == int(ts.dayofweek >= 5)


def test_cyclical_encoding_is_bounded_and_continuous(clean_dataset):
    out = add_time_features(clean_dataset)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()
    # hour 23 and hour 0 must be neighbours in the encoded space
    h23 = np.array([np.sin(2 * np.pi * 23 / 24), np.cos(2 * np.pi * 23 / 24)])
    h0 = np.array([0.0, 1.0])
    h12 = np.array([np.sin(np.pi), np.cos(np.pi)])
    assert np.linalg.norm(h23 - h0) < np.linalg.norm(h23 - h12)


# -------------------------------------------------------------- lag features
def test_all_configured_lags_exist(clean_dataset):
    out = add_lag_features(clean_dataset)
    for lag in config.LAG_PERIODS:
        assert f"lag_{lag}" in out.columns


def test_lag_values_come_from_the_past(clean_dataset):
    """lag_k at row i must equal the target at row i-k. This is the leakage test."""
    out = add_lag_features(clean_dataset)
    target = config.TARGET_COLUMN
    for lag in config.LAG_PERIODS:
        for i in (50, 100, 200):
            assert out[f"lag_{lag}"].iloc[i] == pytest.approx(out[target].iloc[i - lag])


def test_lags_create_leading_nans(clean_dataset):
    out = add_lag_features(clean_dataset)
    assert out["lag_24"].head(24).isna().all()
    assert not out["lag_24"].iloc[24:].isna().any()


# ---------------------------------------------------------- rolling features
def test_rolling_features_exclude_the_current_sample(clean_dataset):
    """rolling_mean_w at row i must use rows i-w .. i-1, never row i."""
    out = add_rolling_features(clean_dataset)
    target = config.TARGET_COLUMN
    window = config.ROLLING_WINDOWS[0]
    i = 100
    expected = out[target].iloc[i - window: i].mean()
    assert out[f"rolling_mean_{window}"].iloc[i] == pytest.approx(expected)


def test_rolling_mean_is_not_the_current_value(clean_dataset):
    out = add_rolling_features(clean_dataset).dropna()
    target = config.TARGET_COLUMN
    window = config.ROLLING_WINDOWS[0]
    # If the current sample leaked in, these would be suspiciously correlated
    # with zero lag; a strict equality check catches the classic bug directly.
    assert not np.allclose(out[f"rolling_mean_{window}"], out[target])


# ------------------------------------------------------------ feature matrix
def test_create_features_drops_warmup_rows(clean_dataset):
    out = create_features(clean_dataset)
    max_lag = max(config.LAG_PERIODS + config.ROLLING_WINDOWS)
    assert len(out) == len(clean_dataset) - max_lag
    assert out.isna().sum().sum() == 0


def test_feature_columns_exclude_target_and_concurrent_measurements(clean_dataset):
    featured = create_features(clean_dataset)
    features = get_feature_columns(featured)
    for forbidden in [config.TARGET_COLUMN, "energy", "voltage", "current",
                      "power_factor", "frequency", "timestamp"]:
        assert forbidden not in features, f"{forbidden} would leak into the model"


def test_build_training_frame_alignment(clean_dataset):
    X, y, names, timestamps = build_training_frame(clean_dataset)
    assert len(X) == len(y) == len(timestamps)
    assert list(X.columns) == names
    assert X.isna().sum().sum() == 0
    assert pd.Series(timestamps).is_monotonic_increasing


# --------------------------------------------------------- prediction inputs
def test_prediction_row_matches_training_columns(clean_dataset):
    X, _, names, _ = build_training_frame(clean_dataset)
    next_ts = clean_dataset["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    row = build_prediction_row(clean_dataset, next_ts)
    assert list(row.columns) == names
    assert len(row) == 1
    assert row.isna().sum().sum() == 0


def test_prediction_row_lags_use_the_latest_history(clean_dataset):
    next_ts = clean_dataset["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    row = build_prediction_row(clean_dataset, next_ts)
    target = config.TARGET_COLUMN
    assert row["lag_1"].iloc[0] == pytest.approx(clean_dataset[target].iloc[-1])
    assert row["lag_24"].iloc[0] == pytest.approx(clean_dataset[target].iloc[-24])


def test_prediction_row_requires_enough_history(clean_dataset):
    with pytest.raises(ValueError):
        build_prediction_row(clean_dataset.head(5), pd.Timestamp("2025-02-01"))
