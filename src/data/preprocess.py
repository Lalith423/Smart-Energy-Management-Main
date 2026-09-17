"""
Data validation and preprocessing.

Design rule: **nothing is removed silently.** Every validation finding and
every cleaning action is recorded in a :class:`ValidationReport`, which is
logged, printed by the CLI and displayed on the dashboard.

Pipeline
--------
1. Load raw CSV, parse timestamps.
2. Validate: required columns, dtypes, timestamps, duplicates, missing values,
   physically implausible values, power-equation consistency.
3. Clean: sort chronologically, drop duplicate timestamps (keep first),
   interpolate short gaps, clip implausible values to the valid band.
4. Recompute derived quantities (active_power, energy) so the physics holds.
5. Save to data/processed/energy_processed.csv.

Run directly:
    python -m src.data.preprocess
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Structured record of what validation found and what cleaning changed."""

    rows_in: int = 0
    rows_out: int = 0
    missing_columns: list[str] = field(default_factory=list)
    non_numeric_columns: list[str] = field(default_factory=list)
    invalid_timestamps: int = 0
    duplicate_timestamps: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    power_equation_mismatches: int = 0
    actions: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the frame is structurally usable (schema + timestamps)."""
        return not self.missing_columns and not self.non_numeric_columns

    def log(self) -> None:
        for line in self.as_text().splitlines():
            logger.info(line)

    def as_text(self) -> str:
        lines = [
            "--- DATA VALIDATION REPORT ---",
            f"Rows in / out        : {self.rows_in} / {self.rows_out}",
            f"Missing columns      : {self.missing_columns or 'none'}",
            f"Non-numeric columns  : {self.non_numeric_columns or 'none'}",
            f"Invalid timestamps   : {self.invalid_timestamps}",
            f"Duplicate timestamps : {self.duplicate_timestamps}",
            f"Missing values       : {self.missing_values or 'none'}",
            f"Out-of-range values  : {self.out_of_range or 'none'}",
            f"P != V*I*PF samples  : {self.power_equation_mismatches}",
            "Actions taken:",
        ]
        lines += [f"  - {a}" for a in (self.actions or ["none"])]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load a raw energy CSV and parse the timestamp column."""
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.data.generate_data"
        )
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> ValidationReport:
    """Inspect a dataframe and return findings. Never modifies the input."""
    report = ValidationReport(rows_in=len(df), rows_out=len(df))

    report.missing_columns = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if report.missing_columns:
        logger.error("Missing required columns: %s", report.missing_columns)
        return report

    for col in config.NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.non_numeric_columns.append(col)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    report.invalid_timestamps = int(ts.isna().sum())
    report.duplicate_timestamps = int(ts.dropna().duplicated().sum())

    missing = {c: int(df[c].isna().sum()) for c in config.NUMERIC_COLUMNS}
    report.missing_values = {c: n for c, n in missing.items() if n > 0}

    for col, (low, high) in config.VALID_RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            bad = int(((df[col] < low) | (df[col] > high)).sum())
            if bad:
                report.out_of_range[col] = bad

    report.power_equation_mismatches = count_power_equation_mismatches(df)
    return report


def count_power_equation_mismatches(df: pd.DataFrame, tolerance: float = 0.02) -> int:
    """
    Count rows where the reported active power disagrees with P = V * I * PF.

    `tolerance` is a relative tolerance (default 2 %).
    """
    needed = {"voltage", "current", "power_factor", "active_power"}
    if not needed.issubset(df.columns):
        return 0
    expected = df["voltage"] * df["current"] * df["power_factor"]
    denominator = expected.abs().clip(lower=1.0)
    relative_error = (df["active_power"] - expected).abs() / denominator
    return int((relative_error > tolerance).sum())


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, report: ValidationReport | None = None) -> pd.DataFrame:
    """
    Return a cleaned copy of ``df``; record every action in ``report``.

    Cleaning steps, in order:
      1. drop rows with unparseable timestamps
      2. sort chronologically
      3. drop duplicate timestamps (keep the first)
      4. interpolate short gaps in numeric columns, then forward/back fill edges
      5. clip physically implausible values into the valid band
      6. recompute active_power and energy from V, I, PF
    """
    report = report or ValidationReport(rows_in=len(df))
    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    n_bad_ts = int(out["timestamp"].isna().sum())
    if n_bad_ts:
        out = out.dropna(subset=["timestamp"])
        report.actions.append(f"dropped {n_bad_ts} rows with unparseable timestamps")

    out = out.sort_values("timestamp").reset_index(drop=True)

    n_dupes = int(out["timestamp"].duplicated().sum())
    if n_dupes:
        out = out.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        report.actions.append(f"dropped {n_dupes} duplicate timestamps (kept first)")

    for col in config.NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        n_missing = int(out[col].isna().sum())
        if n_missing:
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            out[col] = out[col].ffill().bfill()
            report.actions.append(f"interpolated {n_missing} missing values in '{col}'")

    for col, (low, high) in config.VALID_RANGES.items():
        if col not in out.columns:
            continue
        n_bad = int(((out[col] < low) | (out[col] > high)).sum())
        if n_bad:
            out[col] = out[col].clip(lower=low, upper=high)
            report.actions.append(f"clipped {n_bad} out-of-range values in '{col}'")

    # Enforce the physics: P = V * I * PF, E = P * dt / 1000
    out["active_power"] = (out["voltage"] * out["current"] * out["power_factor"]).round(2)
    out["energy"] = (out["active_power"] * config.sampling_interval_hours() / 1000.0).round(5)
    report.actions.append("recomputed active_power = V*I*PF and energy = P*dt/1000")

    # Drop any row that still carries a NaN in a required column.
    before = len(out)
    out = out.dropna(subset=config.REQUIRED_COLUMNS).reset_index(drop=True)
    if len(out) != before:
        report.actions.append(f"dropped {before - len(out)} rows still containing NaN")

    report.rows_out = len(out)
    return out


def preprocess(
    input_path: Path | None = None,
    output_path: Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load -> validate -> clean -> (optionally) save. Returns (df, report)."""
    df = load_raw_data(input_path)
    report = validate_data(df)
    if not report.is_valid:
        raise ValueError(
            "Dataset failed structural validation:\n" + report.as_text()
        )
    cleaned = clean_data(df, report)

    if save:
        output_path = Path(output_path) if output_path else config.PROCESSED_DATA_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)
        logger.info("Saved processed dataset -> %s", output_path)

    return cleaned, report


def load_processed_data(path: Path | None = None) -> pd.DataFrame:
    """Load the processed dataset (used by training, dashboard and notebooks)."""
    path = Path(path) if path is not None else config.PROCESSED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run: python -m src.data.preprocess"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main() -> None:
    config.ensure_directories()
    cleaned, report = preprocess()
    print(report.as_text())
    print("\nProcessed dataset head:")
    print(cleaned.head().to_string(index=False))
    print(f"\nRows: {len(cleaned):,}   Columns: {list(cleaned.columns)}")


if __name__ == "__main__":
    main()
