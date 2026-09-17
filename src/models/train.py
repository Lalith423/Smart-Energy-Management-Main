"""
Model training, comparison and selection.

Chronological splitting
-----------------------
Time-series data must **never** be shuffled. Shuffling lets the model train on
samples that occur after the test samples, so it effectively sees the future -
the reported score would be optimistic and meaningless in deployment.

The split is by calendar date (``config.TRAIN_TEST_SPLIT_DATE``):

    2025-01-01 .. 2025-08-31   ->  training   (~87 %)
    2025-09-01 .. 2025-09-30   ->  testing    (~13 %, never seen in training)

If the dataset does not cover that date, the code falls back to a positional
80/20 chronological split so the pipeline still works on custom data.

Model selection
---------------
The model with the lowest **test RMSE** wins (``config.MODEL_SELECTION_METRIC``).
Nothing is hard-coded: the winner is whatever the metrics say on the day.

Run directly:
    python -m src.models.train
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data.preprocess import load_processed_data
from src.features.feature_engineering import build_training_frame
from src.models.evaluate import evaluate_predictions, format_comparison_table, select_best_model
from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)


def build_model_zoo() -> dict[str, object]:
    """
    Return the candidate estimators.

    XGBoost is included only if it is installed; the project never requires it.
    """
    models: dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=config.RANDOM_SEED,
        ),
    }
    try:  # optional 4th model
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        logger.info("XGBoost detected - included in the comparison")
    except ImportError:
        logger.info("XGBoost not installed - skipping (optional model)")
    return models


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    split_date: str | None = None,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split by time, never at random.

    Returns ``(X_train, X_test, y_train, y_test, ts_train, ts_test)``.
    """
    timestamps = pd.to_datetime(timestamps)
    split_date = split_date or config.TRAIN_TEST_SPLIT_DATE
    boundary = pd.Timestamp(split_date)

    mask = timestamps < boundary
    n_train, n_test = int(mask.sum()), int((~mask).sum())

    if n_train == 0 or n_test == 0:
        cut = int(len(X) * (1 - test_fraction))
        mask = pd.Series(np.arange(len(X)) < cut, index=X.index)
        logger.warning(
            "Split date %s lies outside the data range - falling back to a "
            "positional %d/%d chronological split",
            split_date,
            int(100 * (1 - test_fraction)),
            int(100 * test_fraction),
        )

    return (
        X[mask.to_numpy()],
        X[~mask.to_numpy()],
        y[mask.to_numpy()],
        y[~mask.to_numpy()],
        timestamps[mask.to_numpy()],
        timestamps[~mask.to_numpy()],
    )


def train_models(
    df: pd.DataFrame | None = None, save: bool = True
) -> tuple[dict[str, dict[str, float]], str, dict]:
    """
    Train every candidate model, compare them and persist the winner.

    Returns ``(results, best_model_name, metadata)``.
    """
    config.ensure_directories()
    df = load_processed_data() if df is None else df

    X, y, feature_names, timestamps = build_training_frame(df)
    X_train, X_test, y_train, y_test, ts_train, ts_test = chronological_split(X, y, timestamps)

    logger.info(
        "Train: %d samples (%s -> %s) | Test: %d samples (%s -> %s)",
        len(X_train), ts_train.min(), ts_train.max(),
        len(X_test), ts_test.min(), ts_test.max(),
    )

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}

    for name, model in build_model_zoo().items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        results[name] = evaluate_predictions(y_test, model.predict(X_test))
        fitted[name] = model
        logger.info(
            "%-18s MAE=%8.2f  RMSE=%8.2f  R2=%.4f",
            name, results[name]["mae"], results[name]["rmse"], results[name]["r2"],
        )

    best_name = select_best_model(results, metric=config.MODEL_SELECTION_METRIC)
    best_model = fitted[best_name]
    logger.info(
        "Selected '%s' (lowest test %s = %.2f)",
        best_name, config.MODEL_SELECTION_METRIC.upper(),
        results[best_name][config.MODEL_SELECTION_METRIC],
    )

    metadata = {
        "best_model": best_name,
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "selection_rule": "lowest test RMSE (large errors penalised)",
        "target": config.TARGET_COLUMN,
        "target_unit": "watts",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "train_period": [str(ts_train.min()), str(ts_train.max())],
        "test_period": [str(ts_test.min()), str(ts_test.max())],
        "split_strategy": "chronological (no shuffling)",
        "split_date": config.TRAIN_TEST_SPLIT_DATE,
        "sampling_interval_minutes": config.SAMPLING_INTERVAL_MINUTES,
        "metrics": results,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "synthetic development dataset",
    }

    if save:
        bundle = {
            "model": best_model,
            "model_name": best_name,
            "feature_names": feature_names,
            "metadata": metadata,
        }
        joblib.dump(bundle, config.MODEL_PATH)
        Path(config.MODEL_METADATA_PATH).write_text(json.dumps(metadata, indent=2))
        pd.DataFrame(results).T.rename_axis("model").to_csv(config.MODEL_COMPARISON_PATH)
        logger.info("Saved model -> %s", config.MODEL_PATH)
        logger.info("Saved metadata -> %s", config.MODEL_METADATA_PATH)
        logger.info("Saved comparison -> %s", config.MODEL_COMPARISON_PATH)

    return results, best_name, metadata


def main() -> None:
    results, best_name, metadata = train_models()
    print("\n=== MODEL COMPARISON (test period, chronological split) ===")
    print(f"Train: {metadata['train_period'][0]} -> {metadata['train_period'][1]}"
          f"  ({metadata['n_train_samples']} samples)")
    print(f"Test : {metadata['test_period'][0]} -> {metadata['test_period'][1]}"
          f"  ({metadata['n_test_samples']} samples)")
    print()
    print(format_comparison_table(results))
    print(
        f"\nSelected model: {best_name} "
        f"(criterion: lowest test {config.MODEL_SELECTION_METRIC.upper()})"
    )


if __name__ == "__main__":
    main()
