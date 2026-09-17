"""Tests for the Isolation Forest anomaly-detection module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.anomaly.detector import (
    ANOMALY_LABEL,
    NORMAL_LABEL,
    AnomalyDetector,
    summarise_anomalies,
)


@pytest.fixture(scope="module")
def fitted_detector(request):
    dataset = request.getfixturevalue("clean_dataset")
    return AnomalyDetector(contamination=0.02).fit(dataset)


def test_detector_fits_and_reports_state(clean_dataset):
    detector = AnomalyDetector()
    assert detector.is_fitted is False
    detector.fit(clean_dataset)
    assert detector.is_fitted is True


def test_predict_before_fit_raises(clean_dataset):
    with pytest.raises(RuntimeError):
        AnomalyDetector().predict(clean_dataset)


def test_missing_feature_columns_raise(clean_dataset):
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.fit(clean_dataset.drop(columns=["power_factor"]))


def test_output_shape_and_labels(fitted_detector, clean_dataset):
    result = fitted_detector.predict(clean_dataset)
    assert len(result) == len(clean_dataset)
    assert set(result.columns) == {"anomaly_score", "is_anomaly", "anomaly_status"}
    assert set(result["anomaly_status"].unique()) <= {NORMAL_LABEL, ANOMALY_LABEL}


def test_flag_rate_is_close_to_contamination(fitted_detector, clean_dataset):
    result = fitted_detector.predict(clean_dataset)
    rate = result["is_anomaly"].mean()
    assert 0.0 < rate < 0.10, f"unexpected anomaly rate {rate:.3f}"


def test_status_and_boolean_agree(fitted_detector, clean_dataset):
    result = fitted_detector.predict(clean_dataset)
    assert (result.loc[result["is_anomaly"], "anomaly_status"] == ANOMALY_LABEL).all()
    assert (result.loc[~result["is_anomaly"], "anomaly_status"] == NORMAL_LABEL).all()


def test_typical_reading_is_normal(fitted_detector, clean_dataset):
    typical = clean_dataset.median(numeric_only=True)
    reading = {
        "active_power": float(typical["active_power"]),
        "current": float(typical["current"]),
        "voltage": float(typical["voltage"]),
        "power_factor": float(typical["power_factor"]),
    }
    assert fitted_detector.check_reading(reading)["anomaly_status"] == NORMAL_LABEL


def test_extreme_reading_is_flagged(fitted_detector):
    extreme = {
        "active_power": 5800.0,
        "current": 30.0,
        "voltage": 202.0,
        "power_factor": 0.71,
    }
    result = fitted_detector.check_reading(extreme)
    assert result["anomaly_status"] == ANOMALY_LABEL
    assert result["is_anomaly"] is True


def test_abnormal_samples_score_lower_than_normal_ones(fitted_detector, clean_dataset):
    """Isolation Forest convention: a lower score means more anomalous."""
    result = fitted_detector.predict(clean_dataset)
    assert result.loc[result["is_anomaly"], "anomaly_score"].max() <= (
        result.loc[~result["is_anomaly"], "anomaly_score"].min()
    )


def test_label_dataframe_preserves_original_columns(fitted_detector, clean_dataset):
    labelled = fitted_detector.label_dataframe(clean_dataset)
    for column in clean_dataset.columns:
        assert column in labelled.columns
    assert len(labelled) == len(clean_dataset)


def test_summary_statistics_are_consistent(fitted_detector, clean_dataset):
    labelled = fitted_detector.label_dataframe(clean_dataset)
    stats = summarise_anomalies(labelled)
    assert stats["anomaly_count"] + stats["normal_count"] == stats["total_samples"]
    assert 0.0 <= stats["anomaly_percentage"] <= 100.0


def test_save_and_load_roundtrip(fitted_detector, clean_dataset, tmp_path):
    path = fitted_detector.save(tmp_path / "anomaly.pkl")
    reloaded = AnomalyDetector.load(path)
    assert reloaded.is_fitted
    pd.testing.assert_frame_equal(
        fitted_detector.predict(clean_dataset), reloaded.predict(clean_dataset)
    )


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AnomalyDetector.load(tmp_path / "missing.pkl")


def test_detection_is_deterministic(clean_dataset):
    a = AnomalyDetector(random_state=1).fit(clean_dataset).predict(clean_dataset)
    b = AnomalyDetector(random_state=1).fit(clean_dataset).predict(clean_dataset)
    pd.testing.assert_frame_equal(a, b)
