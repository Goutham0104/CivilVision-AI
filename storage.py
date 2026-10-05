"""
storage.py - Local SQLite Inspection Persistence for CivilVision AI
Provides deterministic local persistence for InspectionRecord and ObservationRecord
without network or external API dependencies.
"""

import os
import sqlite3
from typing import List, Optional, Tuple, Dict, Any
from schema import (
    InspectionRecord,
    ObservationRecord,
    normalize_category,
    normalize_confidence,
    normalize_risk_priority,
    EVIDENCE_TYPES,
)

# Default local database file location
DEFAULT_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "civilvision_inspections.db")

SCHEMA_VERSION = 1


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Returns a configured SQLite connection with foreign keys enabled and row_factory set.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Optional[str] = None) -> bool:
    """
    Initializes the SQLite schema with parameterized DDL, version tracking, and indexes.
    Safe to call repeatedly (idempotent).
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # Schema version metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            # Inspections table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    site_activity TEXT NOT NULL,
                    executive_summary TEXT NOT NULL,
                    disclaimer TEXT NOT NULL,
                    source_image_name TEXT,
                    source_image_size_kb REAL,
                    source_image_dimensions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Observations table (one-to-many relationship)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    sequence_order INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    visual_confidence TEXT NOT NULL,
                    potential_issue TEXT NOT NULL,
                    physical_verification_required TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    risk_priority TEXT NOT NULL,
                    evidence_type TEXT NOT NULL DEFAULT 'VISIBLE',
                    FOREIGN KEY (inspection_id) REFERENCES inspections (inspection_id) ON DELETE CASCADE
                )
                """
            )

            # Indexes for efficient filtering and sorting
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_inspection_id ON observations(inspection_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_category ON observations(category);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_risk_priority ON observations(risk_priority);"
            )

            # Store schema version
            cursor.execute(
                """
                INSERT INTO schema_metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
            return True
    except Exception:
        return False


def inspection_exists(inspection_id: str, db_path: Optional[str] = None) -> bool:
    """
    Checks if an inspection with the given ID already exists in the database.
    """
    if not inspection_id:
        return False
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM inspections WHERE inspection_id = ? LIMIT 1",
                (inspection_id,),
            )
            return cursor.fetchone() is not None
    except Exception:
        return False


def save_inspection_record(record: InspectionRecord, db_path: Optional[str] = None) -> bool:
    """
    Saves an InspectionRecord and all of its ObservationRecords to SQLite in a single transaction.
    If the inspection_id already exists, updates the inspection metadata and reconciles observations
    without creating duplicates.
    Zero Gemini calls.
    Returns True if successfully committed, False otherwise.
    """
    if not record or not getattr(record, "inspection_id", None):
        return False

    try:
        initialize_database(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # Insert or update inspection row
            cursor.execute(
                """
                INSERT INTO inspections (
                    inspection_id,
                    timestamp,
                    site_activity,
                    executive_summary,
                    disclaimer,
                    source_image_name,
                    source_image_size_kb,
                    source_image_dimensions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(inspection_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    site_activity = excluded.site_activity,
                    executive_summary = excluded.executive_summary,
                    disclaimer = excluded.disclaimer,
                    source_image_name = excluded.source_image_name,
                    source_image_size_kb = excluded.source_image_size_kb,
                    source_image_dimensions = excluded.source_image_dimensions
                """,
                (
                    record.inspection_id,
                    record.timestamp,
                    record.site_activity or "",
                    record.executive_summary or "",
                    record.disclaimer or "",
                    record.source_image_name,
                    record.source_image_size_kb,
                    record.source_image_dimensions,
                ),
            )

            # Reconcile observations: delete existing for this ID to maintain exact sequence and avoid duplication
            cursor.execute(
                "DELETE FROM observations WHERE inspection_id = ?",
                (record.inspection_id,),
            )

            # Insert all observations preserving exact sequence order
            obs_rows = []
            for seq, obs in enumerate(record.observations):
                ev_type = obs.evidence_type if obs.evidence_type in EVIDENCE_TYPES else "VISIBLE"
                obs_rows.append(
                    (
                        record.inspection_id,
                        seq,
                        normalize_category(obs.category),
                        obs.observation,
                        normalize_confidence(obs.visual_confidence),
                        obs.potential_issue,
                        obs.physical_verification_required,
                        obs.recommended_action,
                        normalize_risk_priority(obs.risk_priority),
                        ev_type,
                    )
                )

            if obs_rows:
                cursor.executemany(
                    """
                    INSERT INTO observations (
                        inspection_id,
                        sequence_order,
                        category,
                        observation,
                        visual_confidence,
                        potential_issue,
                        physical_verification_required,
                        recommended_action,
                        risk_priority,
                        evidence_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    obs_rows,
                )

            conn.commit()
            return True
    except Exception:
        return False


def load_inspection_record(inspection_id: str, db_path: Optional[str] = None) -> Optional[InspectionRecord]:
    """
    Loads a single InspectionRecord with its observations in exact original sequence from SQLite.
    Returns None if record does not exist or database read fails.
    Zero Gemini calls.
    """
    if not inspection_id:
        return None

    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT inspection_id, timestamp, site_activity, executive_summary, disclaimer,
                       source_image_name, source_image_size_kb, source_image_dimensions
                FROM inspections
                WHERE inspection_id = ?
                """,
                (inspection_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Fetch observations ordered by sequence_order
            cursor.execute(
                """
                SELECT category, observation, visual_confidence, potential_issue,
                       physical_verification_required, recommended_action, risk_priority, evidence_type
                FROM observations
                WHERE inspection_id = ?
                ORDER BY sequence_order ASC
                """,
                (inspection_id,),
            )
            obs_rows = cursor.fetchall()

            observations = []
            for o in obs_rows:
                observations.append(
                    ObservationRecord(
                        category=o["category"],
                        observation=o["observation"],
                        visual_confidence=o["visual_confidence"],
                        potential_issue=o["potential_issue"],
                        physical_verification_required=o["physical_verification_required"],
                        recommended_action=o["recommended_action"],
                        risk_priority=o["risk_priority"],
                        evidence_type=o["evidence_type"] or "VISIBLE",
                    )
                )

            return InspectionRecord(
                inspection_id=row["inspection_id"],
                timestamp=row["timestamp"],
                site_activity=row["site_activity"],
                executive_summary=row["executive_summary"],
                observations=observations,
                disclaimer=row["disclaimer"],
                source_image_name=row["source_image_name"],
                source_image_size_kb=row["source_image_size_kb"],
                source_image_dimensions=row["source_image_dimensions"],
            )
    except Exception:
        return None


def load_inspection_history(db_path: Optional[str] = None) -> List[InspectionRecord]:
    """
    Loads all InspectionRecords from SQLite in chronological insertion order.
    Returns an empty list if the database is empty or on failure.
    Zero Gemini calls.
    """
    try:
        initialize_database(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT inspection_id
                FROM inspections
                ORDER BY timestamp ASC, created_at ASC
                """
            )
            rows = cursor.fetchall()
            records = []
            for r in rows:
                rec = load_inspection_record(r["inspection_id"], db_path)
                if rec:
                    records.append(rec)
            return records
    except Exception:
        return []


def delete_inspection_record(inspection_id: str, db_path: Optional[str] = None) -> bool:
    """
    Deletes an inspection record and its cascaded observations by inspection_id.
    Zero Gemini calls.
    """
    if not inspection_id:
        return False
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inspections WHERE inspection_id = ?", (inspection_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception:
        return False


def get_database_status(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns a health and status summary dictionary for the local SQLite database.
    """
    path = db_path or DEFAULT_DB_PATH
    exists = os.path.exists(path)
    total_inspections = 0
    total_observations = 0
    size_bytes = 0

    if exists:
        try:
            size_bytes = os.path.getsize(path)
            with get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM inspections")
                total_inspections = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM observations")
                total_observations = cursor.fetchone()[0]
        except Exception:
            pass

    return {
        "database_exists": exists,
        "database_path": path,
        "file_size_bytes": size_bytes,
        "total_inspections": total_inspections,
        "total_observations": total_observations,
        "schema_version": SCHEMA_VERSION,
    }
