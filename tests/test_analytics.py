"""Tests for consumption analytics, peak analysis and cost estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils import analytics, config


# --------------------------------------------------------------- consumption
def test_daily_totals_sum_to_the_dataset_total(clean_dataset):
    daily = analytics.daily_consumption(clean_dataset)
    assert daily["energy_kwh"].sum() == pytest.approx(clean_dataset["energy"].sum(), rel=1e-9)


def test_hourly_profile_has_all_24_hours(clean_dataset):
    hourly = analytics.hourly_consumption(clean_dataset)
    assert len(hourly) == 24
    assert set(hourly["hour"]) == set(range(24))


def test_weekly_and_monthly_aggregations_preserve_energy(clean_dataset):
    total = clean_dataset["energy"].sum()
    assert analytics.weekly_consumption(clean_dataset)["energy_kwh"].sum() == pytest.approx(
        total, rel=1e-9
    )
    assert analytics.monthly_consumption(clean_dataset)["energy_kwh"].sum() == pytest.approx(
        total, rel=1e-9
    )


def test_consumption_summary_is_internally_consistent(clean_dataset):
    summary = analytics.consumption_summary(clean_dataset)
    assert summary["min_daily_kwh"] <= summary["avg_daily_kwh"] <= summary["max_daily_kwh"]
    assert summary["n_days"] > 0
    assert summary["total_energy_kwh"] == pytest.approx(
        summary["avg_daily_kwh"] * summary["n_days"], rel=1e-6
    )


def test_resample_handles_an_unordered_frame(clean_dataset):
    shuffled = clean_dataset.sample(frac=1.0, random_state=3)
    daily = analytics.daily_consumption(shuffled)
    assert daily["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------- peak logic
def test_peak_analysis_ordering(clean_dataset):
    peaks = analytics.peak_analysis(clean_dataset)
    assert peaks["min_load_w"] <= peaks["avg_load_w"] <= peaks["max_load_w"]
    assert 0 <= peaks["peak_hour_of_day"] <= 23
    assert 0.0 < peaks["load_factor"] <= 1.0


def test_peak_matches_the_maximum_sample(clean_dataset):
    peaks = analytics.peak_analysis(clean_dataset)
    assert peaks["max_load_w"] == pytest.approx(clean_dataset["active_power"].max())
    row = clean_dataset.loc[clean_dataset["active_power"].idxmax()]
    assert peaks["peak_timestamp"] == row["timestamp"]


def test_peak_hour_is_in_the_evening_for_a_domestic_profile(clean_dataset):
    """The synthetic load curve peaks in the evening; analytics must find it."""
    assert analytics.peak_analysis(clean_dataset)["peak_hour_of_day"] in range(17, 23)


def test_peak_periods_respect_the_quantile_threshold(clean_dataset):
    peaks = analytics.peak_periods(clean_dataset, threshold_quantile=0.9)
    expected = int(len(clean_dataset) * 0.1)
    assert abs(len(peaks) - expected) <= max(2, expected * 0.05)
    assert (peaks["active_power"] >= peaks["threshold_w"].iloc[0]).all()


def test_daily_weekly_monthly_peaks_are_reported(clean_dataset):
    peaks = analytics.peak_analysis(clean_dataset)
    for key in ("daily_peak", "weekly_peak", "monthly_peak"):
        assert peaks[key]["energy_kwh"] > 0


# ---------------------------------------------------------------------- cost
def test_cost_is_energy_times_tariff():
    assert analytics.estimate_cost(100.0, 8.0) == pytest.approx(800.0)
    assert analytics.estimate_cost(0.0, 8.0) == 0.0


def test_cost_scales_linearly_with_the_tariff():
    assert analytics.estimate_cost(50.0, 16.0) == pytest.approx(
        2 * analytics.estimate_cost(50.0, 8.0)
    )


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError):
        analytics.estimate_cost(-1.0, 8.0)
    with pytest.raises(ValueError):
        analytics.estimate_cost(10.0, -8.0)


def test_cost_summary_periods_are_proportional(clean_dataset):
    cost = analytics.cost_summary(clean_dataset, tariff_per_kwh=8.0)
    assert cost["estimated_weekly_cost"] == pytest.approx(cost["estimated_daily_cost"] * 7)
    assert cost["estimated_monthly_cost"] == pytest.approx(cost["estimated_daily_cost"] * 30)
    assert cost["estimated_yearly_cost"] == pytest.approx(cost["estimated_daily_cost"] * 365)


def test_cost_summary_uses_the_default_tariff_when_unspecified(clean_dataset):
    cost = analytics.cost_summary(clean_dataset)
    assert cost["tariff_per_kwh"] == pytest.approx(config.DEFAULT_TARIFF_PER_KWH)


def test_daily_cost_table_matches_daily_energy(clean_dataset):
    table = analytics.daily_cost_table(clean_dataset, tariff_per_kwh=8.0)
    assert np.allclose(table["estimated_cost"], table["energy_kwh"] * 8.0)
    assert isinstance(table, pd.DataFrame)
