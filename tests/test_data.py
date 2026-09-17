"""Tests for dataset generation, validation and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import generate_energy_data, save_dataset, summarise
from src.data.preprocess import (
    clean_data,
    count_power_equation_mismatches,
    load_raw_data,
    preprocess,
    validate_data,
)
from src.utils import config


# ------------------------------------------------------------------ generation
def test_generated_dataset_has_required_columns(small_dataset):
    for column in config.REQUIRED_COLUMNS:
        assert column in small_dataset.columns


def test_generated_dataset_is_hourly_and_ordered(small_dataset):
    ts = small_dataset["timestamp"]
    assert ts.is_monotonic_increasing
    deltas = ts.diff().dropna().unique()
    assert len(deltas) == 1
    assert pd.Timedelta(deltas[0]) == pd.Timedelta(minutes=config.SAMPLING_INTERVAL_MINUTES)


def test_generated_values_are_physically_plausible(small_dataset):
    for column, (low, high) in config.VALID_RANGES.items():
        values = small_dataset[column]
        assert values.min() >= low, f"{column} below {low}"
        assert values.max() <= high, f"{column} above {high}"


def test_power_equation_holds_in_generated_data(small_dataset):
    expected = (
        small_dataset["voltage"] * small_dataset["current"] * small_dataset["power_factor"]
    )
    assert np.allclose(small_dataset["active_power"], expected, rtol=0.02)


def test_energy_matches_sampling_interval(small_dataset):
    expected = small_dataset["active_power"] * config.sampling_interval_hours() / 1000.0
    assert np.allclose(small_dataset["energy"], expected, rtol=1e-3)


def test_generation_is_reproducible():
    a = generate_energy_data(start="2025-01-01", end="2025-01-02 23:00:00", seed=7)
    b = generate_energy_data(start="2025-01-01", end="2025-01-02 23:00:00", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_evening_load_exceeds_night_load(small_dataset):
    """The synthetic profile must reproduce the classic domestic load curve."""
    by_hour = small_dataset.groupby(small_dataset["timestamp"].dt.hour)["active_power"].mean()
    assert by_hour.loc[19] > by_hour.loc[3]


def test_save_and_reload_roundtrip(small_dataset, tmp_path):
    path = save_dataset(small_dataset, tmp_path / "raw.csv")
    reloaded = load_raw_data(path)
    assert len(reloaded) == len(small_dataset)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["timestamp"])


def test_summarise_returns_text(small_dataset):
    text = summarise(small_dataset)
    assert "Rows" in text and "Total energy" in text


# ------------------------------------------------------------------ validation
def test_validation_passes_on_clean_data(small_dataset):
    report = validate_data(small_dataset)
    assert report.is_valid
    assert report.missing_values == {}
    assert report.duplicate_timestamps == 0


def test_validation_detects_missing_columns(small_dataset):
    report = validate_data(small_dataset.drop(columns=["voltage"]))
    assert not report.is_valid
    assert "voltage" in report.missing_columns


def test_validation_detects_duplicates_and_missing_values(small_dataset):
    corrupted = pd.concat([small_dataset, small_dataset.iloc[[0]]], ignore_index=True)
    corrupted.loc[5, "current"] = np.nan
    report = validate_data(corrupted)
    assert report.duplicate_timestamps == 1
    assert report.missing_values.get("current") == 1


def test_validation_detects_out_of_range_values(small_dataset):
    corrupted = small_dataset.copy()
    corrupted.loc[3, "voltage"] = 999.0
    report = validate_data(corrupted)
    assert report.out_of_range.get("voltage") == 1


def test_power_equation_mismatch_counter(small_dataset):
    corrupted = small_dataset.copy()
    corrupted.loc[0, "active_power"] = 99999.0
    assert count_power_equation_mismatches(corrupted) == 1


# ------------------------------------------------------------------ cleaning
def test_cleaning_removes_duplicates_and_reports_it(small_dataset):
    corrupted = pd.concat([small_dataset, small_dataset.iloc[[0]]], ignore_index=True)
    report = validate_data(corrupted)
    cleaned = clean_data(corrupted, report)
    assert cleaned["timestamp"].duplicated().sum() == 0
    assert any("duplicate" in action for action in report.actions)


def test_cleaning_fills_missing_values(small_dataset):
    corrupted = small_dataset.copy()
    corrupted.loc[10, "voltage"] = np.nan
    cleaned = clean_data(corrupted)
    assert cleaned["voltage"].isna().sum() == 0
    assert len(cleaned) == len(corrupted)


def test_cleaning_clips_impossible_values(small_dataset):
    corrupted = small_dataset.copy()
    corrupted.loc[2, "frequency"] = 120.0
    cleaned = clean_data(corrupted)
    low, high = config.VALID_RANGES["frequency"]
    assert cleaned["frequency"].max() <= high
    assert cleaned["frequency"].min() >= low


def test_cleaning_sorts_chronologically(small_dataset):
    shuffled = small_dataset.sample(frac=1.0, random_state=1).reset_index(drop=True)
    cleaned = clean_data(shuffled)
    assert cleaned["timestamp"].is_monotonic_increasing


def test_cleaning_never_silently_drops_rows(small_dataset):
    """Row loss must always be accompanied by a recorded action."""
    report = validate_data(small_dataset)
    cleaned = clean_data(small_dataset, report)
    if report.rows_out < report.rows_in:
        assert report.actions
    assert report.rows_out == len(cleaned)


def test_preprocess_end_to_end(small_dataset, tmp_path):
    raw_path = save_dataset(small_dataset, tmp_path / "raw.csv")
    out_path = tmp_path / "processed.csv"
    cleaned, report = preprocess(input_path=raw_path, output_path=out_path)
    assert out_path.exists()
    assert report.is_valid
    assert len(cleaned) == len(small_dataset)


def test_load_raw_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")
