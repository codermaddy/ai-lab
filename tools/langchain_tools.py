BACKEND_BASE_URL = "http://localhost:8000"

from langchain_core.tools import tool
import requests

@tool("list_runs", return_direct=False)
def list_runs_tool(
    dataset: Optional[str] = None,
    time_range: Optional[str] = None,
    tag: Optional[str] = None,
):
    """
    List experiment runs.

    Use dataset and/or time_range ('today', 'this_week', 'last_week') and/or tag
    when the user asks for runs over a period or with certain properties.
    """
    params: Dict[str, Any] = {}
    if dataset:
        params["dataset"] = dataset
    if time_range:
        params["time_range"] = time_range
    if tag:
        params["tag"] = tag

    resp = requests.get(f"{BACKEND_BASE_URL}/runs", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()
    # Expected JSON (decided by Paras), e.g.:
    # [{"run_id": "...", "dataset": "cifar10", "best_metric_name": "val_accuracy", ...}, ...]


@tool("get_run_details", return_direct=False)
def get_run_details_tool(run_id: str):
    """
    Get detailed information about a single run, including hyperparameters,
    metrics, anomalies and W&B links.

    Use this when the user asks about a specific run or when you need precise
    values for metrics and hyperparameters.
    """
    resp = requests.get(f"{BACKEND_BASE_URL}/runs/{run_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


@tool("compare_runs", return_direct=False)
def compare_runs_tool(run_id_a: str, run_id_b: str):
    """
    Compare two runs by their IDs.

    Returns differences in hyperparameters, metrics, and loss/metric curves.
    Use this for 'compare X and Y' queries.
    """
    resp = requests.get(
        f"{BACKEND_BASE_URL}/runs/compare",
        params={"a": run_id_a, "b": run_id_b},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


@tool("flag_run_for_publish", return_direct=False)
def flag_run_for_publish_tool(run_id: str):
    """
    Mark a run as 'flagged_for_publish'.

    Use when the user asks to prepare a run for publishing to Hugging Face.
    """
    resp = requests.post(
        f"{BACKEND_BASE_URL}/runs/{run_id}/flag",
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
    # Expected JSON example:
    # {"run_id": "...", "status": "flagged_for_publish", "timestamp": "..."}