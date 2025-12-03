# backend/api.py
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from backend.config import PROJECT_NAME
from backend import db_utils
from backend.models import RunSummary, RunDetail, RunComparison, NoteUpdate
from agents.chat_agent import router as agent_router


app = FastAPI(title=f"{PROJECT_NAME} Lab Notebook Backend")
app.include_router(agent_router)

@app.get("/health")
def health():
    return {"status": "ok", "project": PROJECT_NAME}


@app.get("/runs", response_model=List[RunSummary])
def api_list_runs(
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
):
    """
    List runs with basic info and metrics snippet.

    Filters:
    - dataset_id
    - command_substring: substring search in the command string
    - from_timestamp / to_timestamp: ISO timestamp or prefix (e.g., "2025-12-02")
    - has_artifacts: true/false to filter runs with/without artifacts
    - metric_name + metric_min/metric_max: numeric thresholds on a metric
    - note_substring: simple tagging filter (e.g., note contains 'PUBLISH')
    """
    runs = db_utils.list_runs(
        limit=limit,
        offset=offset,
        dataset_id=dataset_id,
        command_substring=command_substring,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        has_artifacts=has_artifacts,
        metric_name=metric_name,
        metric_min=metric_min,
        metric_max=metric_max,
        note_substring=note_substring,
    )
    return runs


@app.get("/runs/leaderboard", response_model=List[RunSummary])
def api_leaderboard(
    metric_name: str,
    top_k: int = 20,
    dataset_id: Optional[str] = None,
    has_artifacts: Optional[bool] = None,
):
    """
    Leaderboard endpoint.

    Returns the top_k runs ordered by a given metric (descending).
    You can optionally filter by dataset_id and whether artifacts exist.
    """
    runs = db_utils.leaderboard(
        metric_name=metric_name,
        top_k=top_k,
        dataset_id=dataset_id,
        has_artifacts=has_artifacts,
    )
    return runs


@app.get("/runs/compare", response_model=RunComparison)
def api_compare_runs(
    ids: List[str] = Query(..., description="Run IDs to compare, e.g. ?ids=id1&ids=id2"),
):
    """
    Compare multiple runs.

    Returns:
    - the list of RunDetail objects
    - the union of all metric keys
    - the union of all param keys
    """
    runs_data = db_utils.get_runs_by_ids(ids)
    if not runs_data:
        raise HTTPException(status_code=404, detail="No runs found for given ids")

    run_details: List[RunDetail] = [RunDetail(**r) for r in runs_data]

    metric_keys = sorted(
        {k for run in run_details for k in run.final_metrics.keys()}
    )
    param_keys = sorted(
        {k for run in run_details for k in run.params.keys()}
    )

    return RunComparison(
        run_ids=ids,
        metric_keys=metric_keys,
        param_keys=param_keys,
        runs=run_details,
    )


@app.get("/runs/{run_id}", response_model=RunDetail)
def api_get_run(run_id: str):
    """
    Get full details for a single run.
    """
    run = db_utils.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.patch("/runs/{run_id}/note", response_model=RunDetail)
def api_update_note(run_id: str, payload: NoteUpdate):
    """
    Update note/tag for a run.

    Typical usage:
    - Mark as 'PUBLISH', 'BEST_VAL_ACC', 'IGNORE', etc.
    """
    ok = db_utils.update_run_note(run_id, payload.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")

    run = db_utils.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found after update")
    return RunDetail(**run)
