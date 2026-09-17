"""Tests for the SQLite persistence layer."""

from __future__ import annotations

import pytest

from src.database.db import (
    clear_readings,
    count_readings,
    database_is_available,
    get_connection,
    get_energy_summary,
    get_readings_by_date,
    get_recent_readings,
    initialize_database,
    insert_reading,
    insert_readings,
)
from src.iot.simulator import EnergySimulator
from src.utils import config


@pytest.fixture
def db_file(temp_db):
    initialize_database(temp_db)
    return temp_db


def _reading(**overrides) -> dict:
    base = {
        "device_id": "TEST_NODE",
        "timestamp": "2025-01-01T10:00:00",
        "voltage": 230.0,
        "current": 2.5,
        "power_factor": 0.95,
        "frequency": 50.0,
    }
    base.update(overrides)
    return base


def test_initialize_creates_the_file_and_table(temp_db):
    assert not temp_db.exists()
    initialize_database(temp_db)
    assert temp_db.exists()
    assert database_is_available(temp_db)


def test_initialize_is_idempotent(temp_db):
    initialize_database(temp_db)
    insert_reading(_reading(), temp_db)
    initialize_database(temp_db)
    assert count_readings(temp_db) == 1


def test_insert_returns_row_id_and_increments_count(db_file):
    row_id = insert_reading(_reading(), db_file)
    assert row_id >= 1
    assert count_readings(db_file) == 1


def test_derived_fields_are_computed_when_absent(db_file):
    insert_reading(_reading(), db_file)
    row = get_recent_readings(1, db_file).iloc[0]
    assert row["active_power"] == pytest.approx(230.0 * 2.5 * 0.95)
    assert row["energy"] == pytest.approx(
        row["active_power"] * config.sampling_interval_hours() / 1000.0
    )


def test_supplied_power_is_not_overwritten(db_file):
    insert_reading(_reading(active_power=1234.5, energy=1.2345), db_file)
    row = get_recent_readings(1, db_file).iloc[0]
    assert row["active_power"] == pytest.approx(1234.5)
    assert row["energy"] == pytest.approx(1.2345)


def test_missing_required_field_raises(db_file):
    broken = _reading()
    del broken["voltage"]
    with pytest.raises(ValueError):
        insert_reading(broken, db_file)


def test_bulk_insert(db_file):
    readings = EnergySimulator(seed=5).generate_batch(25)
    assert insert_readings(readings, db_file) == 25
    assert count_readings(db_file) == 25


def test_bulk_insert_of_nothing_is_a_no_op(db_file):
    assert insert_readings([], db_file) == 0
    assert count_readings(db_file) == 0


def test_recent_readings_are_returned_oldest_first(db_file):
    for hour in range(5):
        insert_reading(_reading(timestamp=f"2025-01-01T0{hour}:00:00"), db_file)
    recent = get_recent_readings(3, db_file)
    assert len(recent) == 3
    assert recent["timestamp"].is_monotonic_increasing
    assert recent["timestamp"].iloc[-1].hour == 4


def test_query_by_date_range(db_file):
    for day in range(1, 6):
        insert_reading(_reading(timestamp=f"2025-01-0{day}T10:00:00"), db_file)
    window = get_readings_by_date("2025-01-02", "2025-01-04T23:59:59", db_file)
    assert len(window) == 3
    assert window["timestamp"].min().day == 2
    assert window["timestamp"].max().day == 4


def test_empty_query_returns_an_empty_frame(db_file):
    result = get_readings_by_date("2030-01-01", "2030-12-31", db_file)
    assert result.empty


def test_energy_summary_aggregates_correctly(db_file):
    insert_reading(_reading(active_power=1000.0, energy=1.0), db_file)
    insert_reading(_reading(timestamp="2025-01-01T11:00:00",
                            active_power=2000.0, energy=2.0), db_file)
    summary = get_energy_summary(db_file)
    assert summary["n_readings"] == 2
    assert summary["total_energy_kwh"] == pytest.approx(3.0)
    assert summary["avg_power_w"] == pytest.approx(1500.0)
    assert summary["max_power_w"] == pytest.approx(2000.0)


def test_anomaly_status_is_persisted_and_counted(db_file):
    insert_reading(_reading(anomaly_status="ANOMALY"), db_file)
    insert_reading(_reading(timestamp="2025-01-01T11:00:00", anomaly_status="NORMAL"), db_file)
    assert get_energy_summary(db_file)["anomaly_count"] == 1


def test_clear_readings(db_file):
    insert_readings(EnergySimulator(seed=1).generate_batch(4), db_file)
    assert clear_readings(db_file) == 4
    assert count_readings(db_file) == 0


def test_connection_rolls_back_on_error(db_file):
    insert_reading(_reading(), db_file)
    with pytest.raises(RuntimeError):
        with get_connection(db_file) as conn:
            conn.execute("DELETE FROM readings")
            raise RuntimeError("simulated failure")
    assert count_readings(db_file) == 1


def test_availability_check_on_a_nonexistent_database(tmp_path):
    assert database_is_available(tmp_path / "never_created.db") is False
