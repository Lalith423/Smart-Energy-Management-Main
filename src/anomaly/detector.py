"""
Anomaly detection with Isolation Forest.

Why Isolation Forest
--------------------
It is unsupervised (we have no labelled faults), it handles multivariate data,
and it is cheap. The idea is simple enough to defend in a viva: the algorithm
builds random trees by splitting on random features at random thresholds.
Points that sit far from the bulk of the data get isolated after very few
splits, so a *short average path length* means *anomalous*.

Features used: active_power, current, voltage, power_factor. They are scaled
with a StandardScaler first so that watts (hundreds) do not dominate power
factor (around 1).

Interpretation - important
--------------------------
A flagged sample is an **abnormal consumption pattern**, i.e. a reading that
does not look like the rest of the data. It is *not* evidence of equipment
failure. Proving failure would need labelled fault data, which this project
does not have.

Run directly:
    python -m src.anomaly.detector
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)

NORMAL_LABEL = "NORMAL"
ANOMALY_LABEL = "ANOMALY"


class AnomalyDetector:
    """Isolation Forest wrapper with scaling, persistence and labelling."""

    def __init__(
        self,
        features: list[str] | None = None,
        contamination: float = config.ANOMALY_CONTAMINATION,
        random_state: int = config.RANDOM_SEED,
        n_estimators: int = 200,
    ) -> None:
        self.features = features or list(config.ANOMALY_FEATURES)
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    # ---------------------------------------------------------------- fit
    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for anomaly detection: {missing}")
        return df[self.features].to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the scaler and the Isolation Forest on historical readings."""
        X = self.scaler.fit_transform(self._matrix(df))
        self.model.fit(X)
        self.is_fitted = True
        logger.info(
            "Fitted IsolationForest on %d samples, features=%s, contamination=%.3f",
            len(df), self.features, self.contamination,
        )
        return self

    # ------------------------------------------------------------ predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score readings.

        Returns a frame with:
          anomaly_score : higher = more normal (scikit-learn convention)
          is_anomaly    : bool
          anomaly_status: "NORMAL" / "ANOMALY"
        """
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted. Call fit() or load a saved model.")

        X = self.scaler.transform(self._matrix(df))
        raw = self.model.predict(X)              # +1 normal, -1 anomaly
        scores = self.model.score_samples(X)     # higher = more normal

        return pd.DataFrame(
            {
                "anomaly_score": scores,
                "is_anomaly": raw == -1,
                "anomaly_status": np.where(raw == -1, ANOMALY_LABEL, NORMAL_LABEL),
            },
            index=df.index,
        )

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with the anomaly columns attached."""
        return pd.concat([df.copy(), self.predict(df)], axis=1)

    def check_reading(self, reading: dict) -> dict:
        """
        Score a single reading dict (used by the real-time monitor).

        Returns ``{"anomaly_status": ..., "is_anomaly": ..., "anomaly_score": ...}``.
        """
        row = pd.DataFrame([{k: reading[k] for k in self.features}])
        result = self.predict(row).iloc[0]
        return {
            "anomaly_status": str(result["anomaly_status"]),
            "is_anomaly": bool(result["is_anomaly"]),
            "anomaly_score": float(result["anomaly_score"]),
        }

    # --------------------------------------------------------- persistence
    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else config.ANOMALY_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "features": self.features,
                "contamination": self.contamination,
            },
            path,
        )
        logger.info("Saved anomaly detector -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "AnomalyDetector":
        path = Path(path) if path else config.ANOMALY_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No anomaly model at {path}. Run: python -m src.anomaly.detector"
            )
        bundle = joblib.load(path)
        detector = cls(features=bundle["features"], contamination=bundle["contamination"])
        detector.model = bundle["model"]
        detector.scaler = bundle["scaler"]
        detector.is_fitted = True
        return detector


# ------------------------------------------------------------------ helpers
def detector_is_available(path: Path | None = None) -> bool:
    """Non-throwing availability check for the dashboard health panel."""
    return Path(path or config.ANOMALY_MODEL_PATH).exists()


def train_detector(df: pd.DataFrame | None = None, save: bool = True) -> AnomalyDetector:
    """Fit a detector on the processed dataset and optionally persist it."""
    from src.data.preprocess import load_processed_data

    df = load_processed_data() if df is None else df
    detector = AnomalyDetector().fit(df)
    if save:
        detector.save()
    return detector


def summarise_anomalies(labelled: pd.DataFrame) -> dict:
    """Aggregate statistics used by the dashboard's anomaly page."""
    total = len(labelled)
    count = int(labelled["is_anomaly"].sum())
    return {
        "total_samples": total,
        "anomaly_count": count,
        "normal_count": total - count,
        "anomaly_percentage": (100.0 * count / total) if total else 0.0,
        "mean_power_normal": float(labelled.loc[~labelled["is_anomaly"], "active_power"].mean())
        if total - count
        else 0.0,
        "mean_power_anomaly": float(labelled.loc[labelled["is_anomaly"], "active_power"].mean())
        if count
        else 0.0,
    }


def main() -> None:
    from src.data.preprocess import load_processed_data

    config.ensure_directories()
    df = load_processed_data()
    detector = train_detector(df)
    labelled = detector.label_dataframe(df)
    stats = summarise_anomalies(labelled)

    print("\n=== ANOMALY DETECTION (Isolation Forest) ===")
    print(f"Features        : {detector.features}")
    print(f"Samples scored  : {stats['total_samples']:,}")
    print(f"Flagged ANOMALY : {stats['anomaly_count']:,} "
          f"({stats['anomaly_percentage']:.2f} %)")
    print(f"Mean power - normal   : {stats['mean_power_normal']:.1f} W")
    print(f"Mean power - abnormal : {stats['mean_power_anomaly']:.1f} W")
    print("\nNote: flagged samples are abnormal consumption patterns, "
          "not confirmed equipment failures.")

    # Sanity check on hand-built readings.
    normal_reading = {
        "active_power": 800.0, "current": 3.8, "voltage": 230.0, "power_factor": 0.92
    }
    extreme_reading = {
        "active_power": 5500.0, "current": 26.0, "voltage": 205.0, "power_factor": 0.72
    }
    print(f"\nTypical reading  -> {detector.check_reading(normal_reading)['anomaly_status']}")
    print(f"Extreme reading  -> {detector.check_reading(extreme_reading)['anomaly_status']}")

    print("\nTop 5 most abnormal samples:")
    top = labelled.nsmallest(5, "anomaly_score")[
        ["timestamp", "voltage", "current", "power_factor", "active_power", "anomaly_status"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
