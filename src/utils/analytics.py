"""
Energy analytics: consumption aggregation, peak-load analysis and cost estimation.

This module holds every calculation the dashboard displays. Keeping it out of
the Streamlit file means the numbers are unit-testable and reusable from the
notebooks, and the UI stays a thin rendering layer.

All results are reproducible from the dataset - nothing is hard-coded.
"""

from __future__ import annotations

import pandas as pd

from src.utils import config


# --------------------------------------------------------------------------
# Consumption aggregation
# --------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


def resample_energy(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Sum energy and average power over a pandas offset alias.

    ``rule`` examples: ``'h'`` hourly, ``'D'`` daily, ``'W'`` weekly,
    ``'MS'`` monthly.
    """
    out = _prepare(df).set_index("timestamp")
    agg = out.resample(rule).agg(
        energy_kwh=("energy", "sum"),
        avg_power_w=("active_power", "mean"),
        max_power_w=("active_power", "max"),
        samples=("active_power", "count"),
    )
    return agg[agg["samples"] > 0].reset_index()


def hourly_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """Energy per hour of the day (averaged across all days) - the load profile."""
    out = _prepare(df)
    out["hour"] = out["timestamp"].dt.hour
    return (
        out.groupby("hour")
        .agg(avg_power_w=("active_power", "mean"), avg_energy_kwh=("energy", "mean"))
        .reset_index()
    )


def daily_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """Total energy per calendar day."""
    return resample_energy(df, "D")


def weekly_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """Total energy per calendar week."""
    return resample_energy(df, "W")


def monthly_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """Total energy per calendar month."""
    return resample_energy(df, "MS")


def consumption_summary(df: pd.DataFrame) -> dict:
    """
    Headline consumption statistics.

    Includes a simple linear trend (kWh/day per day) fitted to the daily totals,
    which tells the user whether consumption is drifting up or down.
    """
    daily = daily_consumption(df)
    total = float(daily["energy_kwh"].sum())

    trend = 0.0
    if len(daily) >= 3:
        x = pd.Series(range(len(daily)), dtype=float)
        y = daily["energy_kwh"].astype(float)
        variance = float(((x - x.mean()) ** 2).sum())
        if variance > 0:
            trend = float(((x - x.mean()) * (y - y.mean())).sum() / variance)

    return {
        "total_energy_kwh": total,
        "n_days": int(len(daily)),
        "avg_daily_kwh": float(daily["energy_kwh"].mean()) if len(daily) else 0.0,
        "max_daily_kwh": float(daily["energy_kwh"].max()) if len(daily) else 0.0,
        "min_daily_kwh": float(daily["energy_kwh"].min()) if len(daily) else 0.0,
        "avg_weekly_kwh": float(weekly_consumption(df)["energy_kwh"].mean())
        if len(daily)
        else 0.0,
        "avg_monthly_kwh": float(monthly_consumption(df)["energy_kwh"].mean())
        if len(daily)
        else 0.0,
        "trend_kwh_per_day": trend,
    }


# --------------------------------------------------------------------------
# Peak-load analysis
# --------------------------------------------------------------------------
def peak_analysis(df: pd.DataFrame) -> dict:
    """
    Maximum / average / minimum load plus when the peaks occurred.

    ``peak_hour`` is the hour of day with the highest *average* power, which is
    the number a distribution engineer would use for load-shifting decisions -
    it is more robust than the single highest instantaneous sample.
    """
    out = _prepare(df)
    power = out["active_power"]
    idx_max = int(power.idxmax())

    hourly = hourly_consumption(out)
    peak_hour = int(hourly.loc[hourly["avg_power_w"].idxmax(), "hour"])

    daily = daily_consumption(out)
    weekly = weekly_consumption(out)
    monthly = monthly_consumption(out)

    def _peak_row(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"timestamp": None, "energy_kwh": 0.0, "max_power_w": 0.0}
        row = frame.loc[frame["energy_kwh"].idxmax()]
        return {
            "timestamp": row["timestamp"],
            "energy_kwh": float(row["energy_kwh"]),
            "max_power_w": float(row["max_power_w"]),
        }

    return {
        "max_load_w": float(power.max()),
        "avg_load_w": float(power.mean()),
        "min_load_w": float(power.min()),
        "peak_timestamp": out.loc[idx_max, "timestamp"],
        "peak_hour_of_day": peak_hour,
        "peak_hour_avg_power_w": float(hourly["avg_power_w"].max()),
        "load_factor": float(power.mean() / power.max()) if power.max() else 0.0,
        "daily_peak": _peak_row(daily),
        "weekly_peak": _peak_row(weekly),
        "monthly_peak": _peak_row(monthly),
    }


def peak_periods(df: pd.DataFrame, threshold_quantile: float = 0.9) -> pd.DataFrame:
    """Return the samples above a load quantile - the 'peak periods' to plot."""
    out = _prepare(df)
    threshold = float(out["active_power"].quantile(threshold_quantile))
    peaks = out[out["active_power"] >= threshold].copy()
    peaks["threshold_w"] = threshold
    return peaks


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------
def estimate_cost(
    energy_kwh: float, tariff_per_kwh: float = config.DEFAULT_TARIFF_PER_KWH
) -> float:
    """
    Estimated electricity cost = energy (kWh) x tariff.

    This is an **estimate only**. A real utility bill adds fixed charges, slab
    rates, taxes, duties and sometimes demand charges, none of which are
    modelled here.
    """
    if energy_kwh < 0:
        raise ValueError("energy_kwh cannot be negative")
    if tariff_per_kwh < 0:
        raise ValueError("tariff_per_kwh cannot be negative")
    return float(energy_kwh) * float(tariff_per_kwh)


def cost_summary(
    df: pd.DataFrame, tariff_per_kwh: float = config.DEFAULT_TARIFF_PER_KWH
) -> dict:
    """Cost estimates derived from the dataset at the given tariff."""
    summary = consumption_summary(df)
    avg_daily = summary["avg_daily_kwh"]
    return {
        "tariff_per_kwh": float(tariff_per_kwh),
        "total_energy_kwh": summary["total_energy_kwh"],
        "total_cost": estimate_cost(summary["total_energy_kwh"], tariff_per_kwh),
        "avg_daily_kwh": avg_daily,
        "estimated_daily_cost": estimate_cost(avg_daily, tariff_per_kwh),
        "estimated_weekly_cost": estimate_cost(avg_daily * 7, tariff_per_kwh),
        "estimated_monthly_cost": estimate_cost(avg_daily * 30, tariff_per_kwh),
        "estimated_yearly_cost": estimate_cost(avg_daily * 365, tariff_per_kwh),
        "n_days": summary["n_days"],
    }


def daily_cost_table(
    df: pd.DataFrame, tariff_per_kwh: float = config.DEFAULT_TARIFF_PER_KWH
) -> pd.DataFrame:
    """Per-day energy and estimated cost, ready for charting or export."""
    daily = daily_consumption(df).copy()
    daily["estimated_cost"] = daily["energy_kwh"] * float(tariff_per_kwh)
    return daily
