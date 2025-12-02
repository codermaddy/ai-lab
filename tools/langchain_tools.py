from langchain_core.tools import tool
import requests
from typing import Any, Dict, List, Optional

BACKEND_BASE_URL = "http://localhost:8000"

@tool
def list_runs(
    limit: int = 50,
    offset: int = 0,
    dataset_id: Optional[str] = None,
    command_substring: Optional[str] = None,
    note_substring: Optional[str] = None,
):
    """
    List runs using the /runs endpoint.

    Useful when user asks:
    - "Show me recent runs on DATASET_X"
    - "List all runs tagged PUBLISH"
    """
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
    }
    if dataset_id:
        params["dataset_id"] = dataset_id
    if command_substring:
        params["command_substring"] = command_substring
    if note_substring:
        params["note_substring"] = note_substring

    r = requests.get(f"{BACKEND_BASE_URL}/runs", params=params, timeout=10)
    r.raise_for_status()
    return r.json()  # list[RunSummary]


@tool
def leaderboard(
    metric_name: str,
    top_k: int = 10,
    dataset_id: Optional[str] = None,
    has_artifacts: Optional[bool] = None,
):
    """
    Get top runs by a metric using /runs/leaderboard.

    Useful when user asks:
    - "What's the best run by val_accuracy on CIFAR-10?"
    - "Top 5 runs by F1 score"
    """
    params: Dict[str, Any] = {
        "metric_name": metric_name,
        "top_k": top_k,
    }
    if dataset_id:
        params["dataset_id"] = dataset_id
    if has_artifacts is not None:
        params["has_artifacts"] = str(has_artifacts).lower()

    r = requests.get(f"{BACKEND_BASE_URL}/runs/leaderboard", params=params, timeout=10)
    r.raise_for_status()
    return r.json()  # list[RunSummary]


@tool
def get_run_details(run_id: str):
    """
    Get full details for a single run via /runs/{run_id}.
    """
    r = requests.get(f"{BACKEND_BASE_URL}/runs/{run_id}", timeout=10)
    r.raise_for_status()
    return r.json()  # RunDetail


@tool
def compare_runs(run_ids: List[str]):
    """
    Compare multiple runs by ID using /runs/compare?ids=...

    The backend returns:
    - run_ids: list[str]
    - metric_keys: list[str]
    - param_keys: list[str]
    - runs: list[RunDetail]
    """
    params: Dict[str, Any] = [("ids", rid) for rid in run_ids]
    r = requests.get(f"{BACKEND_BASE_URL}/runs/compare", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def flag_run_for_publish(run_id: str):
    """
    "Flag for publish" is modelled as updating the note field to include 'PUBLISH'.

    This uses PATCH /runs/{run_id}/note with NoteUpdate{note: "..."}.
    For simplicity, we just set note='PUBLISH'. In a richer version, we could
    append to an existing note instead.
    """
    payload = {"note": "PUBLISH"}
    r = requests.patch(
        f"{BACKEND_BASE_URL}/runs/{run_id}/note",
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()  # RunDetail after update


@tool
def search_run_summaries(
    query: str,
    limit: int = 100,
):
    """
    Simple 'retrieval' over run summaries using /runs.

    For now:
    - Fetch up to `limit` runs.
    - Build a textual summary for each (dataset_id, command, note, metrics snippet).
    - Filter those whose text contains the query substring (case-insensitive).

    This gives the agent high-level context, but for exact numbers it should
    call get_run_details or leaderboard.
    """
    params = {"limit": limit, "offset": 0}
    r = requests.get(f"{BACKEND_BASE_URL}/runs", params=params, timeout=10)
    r.raise_for_status()
    runs = r.json()

    q = query.lower()
    matched = []
    for run in runs:
        # We don't know exact RunSummary fields, so we access them defensively.
        run_id = run.get("run_id") or run.get("id") or "unknown"
        dataset_id = run.get("dataset_id", "unknown_dataset")
        note = run.get("note", "") or ""
        command = run.get("command", "") or ""
        metrics = run.get("final_metrics", {}) or run.get("metrics", {}) or {}

        text = (
            f"Run {run_id} on dataset {dataset_id}. "
            f"Command: {command}. "
            f"Note: {note}. "
            f"Metrics: {metrics}."
        ).lower()

        if q in text:
            matched.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "note": note,
                    "metrics": metrics,
                    "snippet": text,
                }
            )

    return matched