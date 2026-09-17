"""Tests for training, evaluation metrics and the prediction interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.features.feature_engineering import build_training_frame
from src.models.evaluate import (
    evaluate_model,
    evaluate_predictions,
    format_comparison_table,
    mae,
    mape,
    r2,
    rmse,
    select_best_model,
)
from src.models.predict import ModelNotAvailableError, load_model_bundle, predict
from src.models.train import build_model_zoo, chronological_split, train_models
from src.utils import config


# ------------------------------------------------------------------- metrics
def test_metrics_are_zero_for_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    assert mae(y, y) == pytest.approx(0.0)
    assert rmse(y, y) == pytest.approx(0.0)
    assert r2(y, y) == pytest.approx(1.0)
    assert mape(y, y) == pytest.approx(0.0)


def test_metric_values_are_correct():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    assert mae(y_true, y_pred) == pytest.approx(20.0 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((100 + 100 + 0) / 3))


def test_rmse_penalises_large_errors_more_than_mae():
    y_true = np.zeros(10)
    few_big = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 100.0])
    many_small = np.full(10, 10.0)
    assert mae(y_true, few_big) == pytest.approx(mae(y_true, many_small))
    assert rmse(y_true, few_big) > rmse(y_true, many_small)


def test_evaluate_predictions_returns_all_metrics():
    result = evaluate_predictions([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(result) == {"mae", "rmse", "r2", "mape"}
    assert all(isinstance(v, float) for v in result.values())


def test_select_best_model_uses_the_configured_metric():
    results = {
        "A": {"mae": 10, "rmse": 20, "r2": 0.8, "mape": 5},
        "B": {"mae": 12, "rmse": 15, "r2": 0.9, "mape": 6},
    }
    assert select_best_model(results, "rmse") == "B"
    assert select_best_model(results, "mae") == "A"
    assert select_best_model(results, "r2") == "B"


def test_select_best_model_rejects_empty_results():
    with pytest.raises(ValueError):
        select_best_model({})


def test_comparison_table_renders(small_dataset):
    table = format_comparison_table(
        {"LinearRegression": {"mae": 1.0, "rmse": 2.0, "r2": 0.9, "mape": 3.0}}
    )
    assert "LinearRegression" in table and "RMSE" in table


# ------------------------------------------------------------------ training
def test_model_zoo_contains_the_three_required_models():
    zoo = build_model_zoo()
    for name in ("LinearRegression", "RandomForest", "GradientBoosting"):
        assert name in zoo


def test_chronological_split_never_shuffles(clean_dataset):
    X, y, _, timestamps = build_training_frame(clean_dataset)
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te = chronological_split(
        X, y, timestamps, split_date="2025-01-08"
    )
    assert len(X_tr) > 0 and len(X_te) > 0
    assert ts_tr.max() < ts_te.min(), "training data must precede test data"
    assert len(X_tr) + len(X_te) == len(X)


def test_split_falls_back_when_date_is_outside_the_range(clean_dataset):
    X, y, _, timestamps = build_training_frame(clean_dataset)
    X_tr, X_te, *_ = chronological_split(X, y, timestamps, split_date="2099-01-01")
    assert len(X_tr) > 0 and len(X_te) > 0


def test_training_produces_metrics_and_selects_a_model(clean_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT_DATE", "2025-01-08")
    results, best_name, metadata = train_models(clean_dataset, save=False)

    assert len(results) >= 3
    assert best_name in results
    for metrics in results.values():
        assert metrics["mae"] >= 0 and metrics["rmse"] >= metrics["mae"]
    assert metadata["split_strategy"].startswith("chronological")
    assert metadata["n_train_samples"] > metadata["n_test_samples"]


def test_a_trained_model_beats_predicting_the_mean(clean_dataset, monkeypatch):
    """Sanity check: the pipeline must add value over a naive baseline."""
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT_DATE", "2025-01-08")
    X, y, _, timestamps = build_training_frame(clean_dataset)
    X_tr, X_te, y_tr, y_te, *_ = chronological_split(X, y, timestamps, split_date="2025-01-08")

    model = LinearRegression().fit(X_tr, y_tr)
    model_rmse = rmse(y_te, model.predict(X_te))
    baseline_rmse = rmse(y_te, np.full(len(y_te), y_tr.mean()))
    assert model_rmse < baseline_rmse


def test_evaluate_model_on_a_fitted_estimator(clean_dataset):
    X, y, _, _ = build_training_frame(clean_dataset)
    model = LinearRegression().fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert 0.0 <= metrics["r2"] <= 1.0


# ---------------------------------------------------------------- prediction
def test_saved_model_bundle_is_well_formed():
    """Runs against the artefact produced by `python -m src.models.train`."""
    if not config.MODEL_PATH.exists():
        pytest.skip("no trained model artefact; run python -m src.models.train")
    bundle = load_model_bundle()
    assert {"model", "model_name", "feature_names", "metadata"} <= set(bundle)
    assert len(bundle["feature_names"]) > 0


def test_missing_model_raises_a_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        load_model_bundle(str(tmp_path / "nope.pkl"))


def test_predict_rejects_a_frame_with_missing_columns():
    if not config.MODEL_PATH.exists():
        pytest.skip("no trained model artefact")
    with pytest.raises(ValueError):
        predict(pd.DataFrame({"hour": [1]}))


def test_next_hour_and_forecast_are_plausible():
    if not config.MODEL_PATH.exists() or not config.PROCESSED_DATA_PATH.exists():
        pytest.skip("pipeline artefacts not built")

    from src.data.preprocess import load_processed_data
    from src.models.predict import forecast, predict_next_hour

    df = load_processed_data()
    nxt = predict_next_hour(df)
    low, high = config.VALID_RANGES["active_power"]
    assert low <= nxt["predicted_power_w"] <= high
    assert nxt["timestamp"] > df["timestamp"].iloc[-1]

    fc = forecast(df, horizon=6)
    assert len(fc) == 6
    assert fc["timestamp"].is_monotonic_increasing
    assert (fc["predicted_power_w"] > 0).all()
    assert fc["predicted_energy_kwh"].sum() == pytest.approx(
        fc["predicted_power_w"].sum() * config.sampling_interval_hours() / 1000.0
    )


def test_forecast_rejects_a_non_positive_horizon():
    if not config.MODEL_PATH.exists() or not config.PROCESSED_DATA_PATH.exists():
        pytest.skip("pipeline artefacts not built")
    from src.data.preprocess import load_processed_data
    from src.models.predict import forecast

    with pytest.raises(ValueError):
        forecast(load_processed_data(), horizon=0)
