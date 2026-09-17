"""
Prediction interface.

Architecture rule enforced here:

    Dashboard  ->  this module  ->  saved model (models/best_model.pkl)

The dashboard never trains. It only asks this module for predictions, so the
UI stays fast and the model in production is exactly the artefact that was
evaluated during training.

Multi-step forecasting is **recursive**: the prediction for hour t+1 is fed
back in as the ``lag_1`` value when predicting t+2. Errors therefore compound
with horizon, which is why the dashboard labels long horizons as indicative.

Run directly:
    python -m src.models.predict
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import build_prediction_row, build_training_frame
from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained model artefact is missing or unreadable."""


@lru_cache(maxsize=4)
def load_model_bundle(path: str | None = None) -> dict:
    """
    Load and cache the model bundle.

    The bundle is a dict: ``{model, model_name, feature_names, metadata}``.
    Caching means the dashboard deserialises the pickle once per session.
    """
    model_path = Path(path) if path else config.MODEL_PATH
    if not model_path.exists():
        raise ModelNotAvailableError(
            f"No trained model at {model_path}. Run: python -m src.models.train"
        )
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - corrupt artefact
        raise ModelNotAvailableError(f"Could not load model from {model_path}: {exc}") from exc

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelNotAvailableError(f"Unexpected model artefact format in {model_path}")
    logger.info("Loaded model '%s' from %s", bundle.get("model_name", "?"), model_path)
    return bundle


def model_is_available(path: str | None = None) -> bool:
    """Non-throwing check used by the dashboard's system-health panel."""
    try:
        load_model_bundle(path)
        return True
    except ModelNotAvailableError:
        return False


def get_model_metadata(path: str | None = None) -> dict:
    """Return the metadata dictionary stored alongside the model."""
    return load_model_bundle(path).get("metadata", {})


def predict(features: pd.DataFrame, path: str | None = None) -> pd.Series:
    """
    Predict active power (watts) for a prepared feature frame.

    Columns are reordered to the training order; a missing column is a hard
    error rather than a silent zero-fill.
    """
    bundle = load_model_bundle(path)
    feature_names = bundle["feature_names"]

    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")

    predictions = bundle["model"].predict(features[feature_names])
    return pd.Series(predictions, index=features.index, name="predicted_power")


def predict_on_history(df: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    """
    Run the model across a historical dataframe (actual vs predicted).

    Returns columns: timestamp, actual, predicted, error, abs_error.
    """
    X, y, _, timestamps = build_training_frame(df)
    y_pred = predict(X, path=path)
    out = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "actual": y.to_numpy(),
            "predicted": y_pred.to_numpy(),
        }
    )
    out["error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out


def predict_next_hour(history: pd.DataFrame, path: str | None = None) -> dict:
    """
    Predict the next sampling interval after the end of ``history``.

    Returns ``{"timestamp": Timestamp, "predicted_power_w": float,
    "predicted_energy_kwh": float}``.
    """
    history = history.sort_values("timestamp").reset_index(drop=True)
    step = pd.Timedelta(minutes=config.SAMPLING_INTERVAL_MINUTES)
    next_ts = pd.Timestamp(history["timestamp"].iloc[-1]) + step

    features = build_prediction_row(history, next_ts)
    power = float(predict(features, path=path).iloc[0])
    return {
        "timestamp": next_ts,
        "predicted_power_w": power,
        "predicted_energy_kwh": power * config.sampling_interval_hours() / 1000.0,
    }


def forecast(history: pd.DataFrame, horizon: int = 24, path: str | None = None) -> pd.DataFrame:
    """
    Recursive multi-step forecast.

    Each prediction is appended to a working copy of the history so that the
    next step can build its lag features from it.

    Returns columns: timestamp, predicted_power_w, predicted_energy_kwh, step.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    target = config.TARGET_COLUMN
    step = pd.Timedelta(minutes=config.SAMPLING_INTERVAL_MINUTES)
    max_lag = max(config.LAG_PERIODS + config.ROLLING_WINDOWS)

    working = history.sort_values("timestamp").reset_index(drop=True)[["timestamp", target]]
    working = working.tail(max(max_lag * 2, 96)).reset_index(drop=True)

    rows = []
    for i in range(1, horizon + 1):
        next_ts = pd.Timestamp(working["timestamp"].iloc[-1]) + step
        features = build_prediction_row(working, next_ts)
        power = float(predict(features, path=path).iloc[0])
        rows.append(
            {
                "timestamp": next_ts,
                "predicted_power_w": power,
                "predicted_energy_kwh": power * config.sampling_interval_hours() / 1000.0,
                "step": i,
            }
        )
        working = pd.concat(
            [working, pd.DataFrame([{"timestamp": next_ts, target: power}])],
            ignore_index=True,
        )
    return pd.DataFrame(rows)


def forecast_daily_energy(history: pd.DataFrame, path: str | None = None) -> float:
    """Total predicted energy (kWh) over the next 24 hours."""
    horizon = int(round(24 * 60 / config.SAMPLING_INTERVAL_MINUTES))
    return float(forecast(history, horizon=horizon, path=path)["predicted_energy_kwh"].sum())


def main() -> None:
    from src.data.preprocess import load_processed_data

    df = load_processed_data()
    meta = get_model_metadata()
    print(f"Model: {meta.get('best_model')}  |  trained {meta.get('trained_at')}")

    nxt = predict_next_hour(df)
    print(f"\nNext-interval forecast @ {nxt['timestamp']}: "
          f"{nxt['predicted_power_w']:.1f} W  ({nxt['predicted_energy_kwh']:.4f} kWh)")

    fc = forecast(df, horizon=24)
    print(f"\n24-hour recursive forecast (first 5 steps):\n{fc.head().to_string(index=False)}")
    print(f"\nPredicted energy for the next 24 h: {fc['predicted_energy_kwh'].sum():.2f} kWh")

    hist = predict_on_history(df.tail(800))
    print(f"\nBack-test on last {len(hist)} samples: "
          f"MAE = {hist['abs_error'].mean():.2f} W")


if __name__ == "__main__":
    main()
