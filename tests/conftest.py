"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generate_data import generate_energy_data


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    """A deterministic 10-day hourly dataset (fast to build, enough for lags)."""
    return generate_energy_data(
        start="2025-01-01", end="2025-01-10 23:00:00", seed=123, n_abnormal_events=5
    )


@pytest.fixture(scope="session")
def clean_dataset(small_dataset) -> pd.DataFrame:
    """The same dataset after the preprocessing pipeline."""
    from src.data.preprocess import clean_data

    return clean_data(small_dataset)


@pytest.fixture
def temp_db(tmp_path):
    """Path to an isolated SQLite file for database tests."""
    return tmp_path / "test_energy.db"
