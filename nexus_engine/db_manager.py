"""
nexus_engine/db_manager.py — SQLite Audit Trail Database Manager
=================================================================
Lightweight database manager for persisting SAR pipeline audit logs.

Tables:
  - sar_audit_logs: Stores final graph state, audit trail, and SAR output
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


# Default database path (can be overridden)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "sar_audit.db"


def get_db_path() -> Path:
    """Get the database file path, creating parent directories if needed."""
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_audit_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize the SQLite database with the sar_audit_logs table.
    
    Creates the table if it doesn't exist with columns:
      - id: Primary key (AUTOINCREMENT)
      - case_id: Unique case identifier (TEXT, NOT NULL)
      - timestamp: ISO-8601 timestamp when record was created (TEXT, NOT NULL)
      - full_audit_json: Complete graph state serialized as JSON (TEXT)
      - final_sar: The final generated SAR narrative text (TEXT)
    
    Args:
        db_path: Optional custom database file path. Uses default if not provided.
    """
    db_file = db_path or get_db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sar_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            full_audit_json TEXT,
            final_sar TEXT,
            UNIQUE(case_id)
        )
    """)
    
    # Create index on case_id for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_case_id ON sar_audit_logs(case_id)
    """)
    
    conn.commit()
    conn.close()
    print(f"[DB Manager] Audit database initialized at: {db_file}")


def save_pipeline_state(
    case_id: str,
    graph_state: Dict[str, Any],
    final_sar: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """
    Save the final LangGraph pipeline state to the audit database.
    
    Args:
        case_id: Unique case identifier
        graph_state: Complete graph state dictionary from LangGraph
        final_sar: The final generated SAR narrative (optional)
        db_path: Optional custom database file path
        
    Returns:
        True if save was successful, False otherwise
    """
    db_file = db_path or get_db_path()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Serialize full graph state to JSON
    try:
        audit_json = json.dumps(graph_state, default=str, indent=2)
    except (TypeError, ValueError) as e:
        print(f"[DB Manager] Warning: Could not serialize full state: {e}")
        # Fallback: serialize basic info
        audit_json = json.dumps({
            "case_id": case_id,
            "error": f"Full serialization failed: {e}",
            "keys": list(graph_state.keys())
        })
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Use INSERT OR REPLACE to handle duplicate case_ids
        cursor.execute("""
            INSERT OR REPLACE INTO sar_audit_logs (case_id, timestamp, full_audit_json, final_sar)
            VALUES (?, ?, ?, ?)
        """, (case_id, timestamp, audit_json, final_sar or ""))
        
        conn.commit()
        conn.close()
        print(f"[DB Manager] Saved audit log for case: {case_id}")
        return True
        
    except sqlite3.Error as e:
        print(f"[DB Manager] Error saving audit log for {case_id}: {e}")
        return False


def get_audit_log(case_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve audit log for a specific case.
    
    Args:
        case_id: Unique case identifier
        db_path: Optional custom database file path
        
    Returns:
        Dictionary with audit data or None if not found
    """
    db_file = db_path or get_db_path()
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT case_id, timestamp, full_audit_json, final_sar
            FROM sar_audit_logs
            WHERE case_id = ?
        """, (case_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "case_id": row[0],
                "timestamp": row[1],
                "full_audit_json": json.loads(row[2]) if row[2] else {},
                "final_sar": row[3]
            }
        return None
        
    except sqlite3.Error as e:
        print(f"[DB Manager] Error retrieving audit log for {case_id}: {e}")
        return None


def list_cases(db_path: Optional[Path] = None) -> list:
    """
    List all cases in the audit database.
    
    Args:
        db_path: Optional custom database file path
        
    Returns:
        List of dictionaries with case_id and timestamp
    """
    db_file = db_path or get_db_path()
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT case_id, timestamp FROM sar_audit_logs
            ORDER BY timestamp DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"case_id": row[0], "timestamp": row[1]} for row in rows]
        
    except sqlite3.Error as e:
        print(f"[DB Manager] Error listing cases: {e}")
        return []


# Convenience function for direct use
def save_final_state(state: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """
    Convenience wrapper to save final state with automatic extraction.
    
    Extracts case_id and final_sar from graph state automatically.
    
    Args:
        state: Complete LangGraph state dictionary
        db_path: Optional custom database file path
        
    Returns:
        True if save was successful, False otherwise
    """
    case_id = state.get("case_id", "UNKNOWN")
    
    # Try to get final SAR from various possible fields
    final_sar = state.get("final_sar_clean") or state.get("sar_draft") or ""
    
    return save_pipeline_state(case_id, state, final_sar, db_path)
