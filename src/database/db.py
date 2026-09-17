"""
SQLite persistence layer.

Everything the dashboard, the simulator and the MQTT client need to store or
read lives behind these functions. No SQL is written anywhere else in the
project, and no database server is required - SQLite is a single local file.

Schema (table ``readings``)
---------------------------
    id             INTEGER PRIMARY KEY
    timestamp      TEXT    ISO-8601, indexed
    device_id      TEXT    e.g. ENERGY_NODE_01
    voltage        REAL    volts
    current        REAL    amperes
    power_factor   REAL    0..1
    frequency      REAL    hertz
    active_power   REAL    watts
    energy         REAL    kWh for this sampling interval
    anomaly_status TEXT    'NORMAL' / 'ANOMALY' / NULL when not scored

Run directly (creates the DB and shows a summary):
    python -m src.database.db
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from src.utils import config
from src.utils.config import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    device_id      TEXT NOT NULL DEFAULT 'UNKNOWN',
    voltage        REAL NOT NULL,
    current        REAL NOT NULL,
    power_factor   REAL NOT NULL,
    frequency      REAL NOT NULL,
    active_power   REAL NOT NULL,
    energy         REAL NOT NULL,
    anomaly_status TEXT,
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_readings_device    ON readings(device_id);
"""

READING_COLUMNS = [
    "timestamp",
    "device_id",
    "voltage",
    "current",
    "power_factor",
    "frequency",
    "active_power",
    "energy",
    "anomaly_status",
]


@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """
    Context manager yielding a SQLite connection with row access by name.

    Commits on success, rolls back on exception, always closes.
    """
    path = Path(db_path) if db_path else config.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(db_path: Path | str | None = None) -> Path:
    """Create the database file and schema if they do not already exist."""
    path = Path(db_path) if db_path else config.DATABASE_PATH
    with get_connection(path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database ready at %s", path)
    return path


def database_is_available(db_path: Path | str | None = None) -> bool:
    """Non-throwing health check used by the dashboard status panel."""
    try:
        with get_connection(db_path) as conn:
            conn.execute("SELECT 1 FROM readings LIMIT 1")
        return True
    except sqlite3.Error:
        return False


def _normalise(reading: dict) -> tuple:
    """Validate and coerce a reading dict into an ordered tuple for insertion."""
    required = ["voltage", "current", "power_factor", "frequency"]
    missing = [k for k in required if k not in reading]
    if missing:
        raise ValueError(f"Reading is missing required fields: {missing}")

    voltage = float(reading["voltage"])
    current = float(reading["current"])
    power_factor = float(reading["power_factor"])
    frequency = float(reading["frequency"])

    # Derive power/energy when the node did not send them (P = V*I*PF).
    active_power = float(reading.get("active_power") or voltage * current * power_factor)
    energy = float(
        reading.get("energy")
        if reading.get("energy") is not None
        else active_power * config.sampling_interval_hours() / 1000.0
    )

    timestamp = reading.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    if isinstance(timestamp, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(timestamp).isoformat()

    return (
        str(timestamp),
        str(reading.get("device_id", config.DEVICE_ID)),
        voltage,
        current,
        power_factor,
        frequency,
        active_power,
        energy,
        reading.get("anomaly_status"),
    )


def insert_reading(reading: dict, db_path: Path | str | None = None) -> int:
    """Insert one reading. Returns the new row id."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            f"INSERT INTO readings ({', '.join(READING_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(READING_COLUMNS))})",
            _normalise(reading),
        )
        return int(cursor.lastrowid)


def insert_readings(readings: Iterable[dict], db_path: Path | str | None = None) -> int:
    """Bulk insert. Returns the number of rows written."""
    rows = [_normalise(r) for r in readings]
    if not rows:
        return 0
    with get_connection(db_path) as conn:
        conn.executemany(
            f"INSERT INTO readings ({', '.join(READING_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(READING_COLUMNS))})",
            rows,
        )
    logger.info("Inserted %d readings", len(rows))
    return len(rows)


def _to_frame(rows: list[sqlite3.Row]) -> pd.DataFrame:
    """Convert sqlite rows to a dataframe with a parsed timestamp column."""
    if not rows:
        return pd.DataFrame(columns=["id", *READING_COLUMNS, "created_at"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def get_recent_readings(limit: int = 100, db_path: Path | str | None = None) -> pd.DataFrame:
    """Return the most recent ``limit`` readings, oldest first (plot-ready)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM readings ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
    df = _to_frame(rows)
    return df.iloc[::-1].reset_index(drop=True) if not df.empty else df


def get_readings_by_date(
    start: str, end: str, db_path: Path | str | None = None
) -> pd.DataFrame:
    """Return readings in the inclusive timestamp window ``[start, end]``."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM readings WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC",
            (str(start), str(end)),
        ).fetchall()
    return _to_frame(rows)


def get_energy_summary(db_path: Path | str | None = None) -> dict:
    """Aggregate statistics over everything stored in the database."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)            AS n_readings,
                   MIN(timestamp)      AS first_timestamp,
                   MAX(timestamp)      AS last_timestamp,
                   SUM(energy)         AS total_energy_kwh,
                   AVG(active_power)   AS avg_power_w,
                   MAX(active_power)   AS max_power_w,
                   MIN(active_power)   AS min_power_w,
                   AVG(voltage)        AS avg_voltage,
                   AVG(power_factor)   AS avg_power_factor
            FROM readings
            """
        ).fetchone()
        anomalies = conn.execute(
            "SELECT COUNT(*) FROM readings WHERE anomaly_status = 'ANOMALY'"
        ).fetchone()[0]

    summary = {k: row[k] for k in row.keys()}
    summary["anomaly_count"] = int(anomalies)
    summary["total_energy_kwh"] = float(summary["total_energy_kwh"] or 0.0)
    return summary


def count_readings(db_path: Path | str | None = None) -> int:
    """Total number of rows in the readings table."""
    with get_connection(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0])


def clear_readings(db_path: Path | str | None = None) -> int:
    """Delete every reading. Returns the number of rows removed."""
    with get_connection(db_path) as conn:
        n = int(conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0])
        conn.execute("DELETE FROM readings")
    logger.warning("Cleared %d readings", n)
    return n


def main() -> None:
    from src.iot.simulator import EnergySimulator

    initialize_database()
    simulator = EnergySimulator(seed=config.RANDOM_SEED)
    batch = simulator.generate_batch(50)
    insert_readings(batch)

    print(f"Database file : {config.DATABASE_PATH}")
    print(f"Rows stored   : {count_readings():,}")
    print("\nMost recent 5 readings:")
    recent = get_recent_readings(5)
    print(recent[["timestamp", "device_id", "voltage", "current",
                  "active_power", "energy"]].to_string(index=False))
    print("\nEnergy summary:")
    for key, value in get_energy_summary().items():
        print(f"  {key:<20} {value}")


if __name__ == "__main__":
    main()
