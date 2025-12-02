# backend/db_utils.py
import sqlite3
import json
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    # timeout helps avoid "database is locked" if multiple readers
    return sqlite3.connect(DB_PATH, timeout=10)


def _row_to_detail(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert a full manifests row to a dict with parsed JSON fields.
    """
    return {
        "run_id": row["run_id"],
        "timestamp": row["timestamp"],
        "git_hash": row["git_hash"],
        "git_diff": row["git_diff"],
        "command": row["command"],
        "python_version": row["python_version"],
        "pip_freeze": row["pip_freeze"],
        "device_info": row["device_info"],
        "dataset_id": row["dataset_id"],
        "params": json.loads(row["params_json"] or "{}"),
        "final_metrics": json.loads(row["final_metrics_json"] or "{}"),
        "artifacts": json.loads(row["artifacts_json"] or "[]"),
        "wandb_url": row["wandb_url"],
        "note": row["note"],
    }


def list_runs(
    limit: int = 50,
    offset: int = 0,
    dataset_id: Optional[str] = None,
    command_substring: Optional[str] = None,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
    has_artifacts: Optional[bool] = None,
    metric_name: Optional[str] = None,
    metric_min: Optional[float] = None,
    metric_max: Optional[float] = None,
    note_substring: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return a list of runs with basic info + metrics snippet.

    Supports filtering by:
    - dataset_id
    - command_substring
    - timestamp range
    - has_artifacts
    - metric_name + numeric thresholds
    - note_substring (for simple tagging, e.g. "PUBLISH")
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    base_query = """
        SELECT
            run_id,
            timestamp,
            git_hash,
            command,
            dataset_id,
            wandb_url,
            final_metrics_json,
            artifacts_json,
            note
        FROM manifests
    """

    clauses: List[str] = []
    params: List[Any] = []

    if dataset_id:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)

    if from_timestamp:
        clauses.append("timestamp >= ?")
        params.append(from_timestamp)

    if to_timestamp:
        clauses.append("timestamp <= ?")
        params.append(to_timestamp)

    if command_substring:
        clauses.append("command LIKE ?")
        params.append(f"%{command_substring}%")

    if note_substring:
        clauses.append("note LIKE ?")
        params.append(f"%{note_substring}%")

    if clauses:
        base_query += " WHERE " + " AND ".join(clauses)

    base_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(base_query, params).fetchall()
    conn.close()

    runs: List[Dict[str, Any]] = []
    for r in rows:
        metrics = {}
        if r["final_metrics_json"]:
            try:
                metrics = json.loads(r["final_metrics_json"])
            except json.JSONDecodeError:
                metrics = {}

        artifacts = []
        if r["artifacts_json"]:
            try:
                artifacts = json.loads(r["artifacts_json"])
            except json.JSONDecodeError:
                artifacts = []

        # filter in Python on has_artifacts + metric ranges
        if has_artifacts is not None:
            cond = len(artifacts) > 0
            if has_artifacts != cond:
                continue

        if metric_name is not None and (metric_min is not None or metric_max is not None):
            val = metrics.get(metric_name)
            if not isinstance(val, (int, float)):
                # skip if metric not present or not numeric
                continue
            if metric_min is not None and val < metric_min:
                continue
            if metric_max is not None and val > metric_max:
                continue

        runs.append(
            {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "git_hash": r["git_hash"],
                "command": r["command"],
                "dataset_id": r["dataset_id"],
                "wandb_url": r["wandb_url"],
                "final_metrics": metrics,
                # "note": r["note"],  # add this if you expose note in RunSummary
            }
        )
    return runs


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Return full details for a given run_id, or None if not found.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM manifests WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None
    return _row_to_detail(row)


def get_runs_by_ids(run_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Return full details for a list of run_ids (for comparisons).
    """
    if not run_ids:
        return []

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" for _ in run_ids)
    query = f"SELECT * FROM manifests WHERE run_id IN ({placeholders})"
    rows = conn.execute(query, run_ids).fetchall()
    conn.close()

    return [_row_to_detail(r) for r in rows]


def leaderboard(
    metric_name: str,
    top_k: int = 20,
    dataset_id: Optional[str] = None,
    has_artifacts: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Return top_k runs ordered by a given numeric metric (descending).

    We do filtering in SQL where possible, then parse metrics and sort in Python.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    base_query = """
        SELECT
            run_id,
            timestamp,
            git_hash,
            command,
            dataset_id,
            wandb_url,
            final_metrics_json,
            artifacts_json
        FROM manifests
    """

    clauses: List[str] = []
    params: List[Any] = []

    if dataset_id:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)

    if clauses:
        base_query += " WHERE " + " AND ".join(clauses)

    # we don't ORDER BY here; we'll sort in Python by the metric
    rows = conn.execute(base_query, params).fetchall()
    conn.close()

    scored: List[Dict[str, Any]] = []
    for r in rows:
        metrics = {}
        if r["final_metrics_json"]:
            try:
                metrics = json.loads(r["final_metrics_json"])
            except json.JSONDecodeError:
                metrics = {}

        artifacts = []
        if r["artifacts_json"]:
            try:
                artifacts = json.loads(r["artifacts_json"])
            except json.JSONDecodeError:
                artifacts = []

        if has_artifacts is not None:
            cond = len(artifacts) > 0
            if has_artifacts != cond:
                continue

        val = metrics.get(metric_name)
        if not isinstance(val, (int, float)):
            continue

        scored.append(
            {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "git_hash": r["git_hash"],
                "command": r["command"],
                "dataset_id": r["dataset_id"],
                "wandb_url": r["wandb_url"],
                "final_metrics": metrics,
            }
        )

    # sort by metric descending, take top_k
    scored.sort(key=lambda x: x["final_metrics"].get(metric_name, float("-inf")), reverse=True)
    return scored[:top_k]


def update_run_note(run_id: str, note: str) -> bool:
    """
    Update the note (tagging) for a run.
    Returns True if a row was updated, False otherwise.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE manifests SET note = ? WHERE run_id = ?",
        (note, run_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated
