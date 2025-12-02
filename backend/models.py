# backend/models.py
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class RunSummary(BaseModel):
    run_id: str
    timestamp: str
    git_hash: Optional[str] = None
    command: Optional[str] = None
    dataset_id: Optional[str] = None
    wandb_url: Optional[str] = None
    # optional small snippet of metrics for the list view
    final_metrics: Dict[str, Any] = {}
    # you can also surface note here if you like
    note: Optional[str] = None


class RunDetail(BaseModel):
    run_id: str
    timestamp: str
    git_hash: Optional[str] = None
    git_diff: Optional[str] = None
    command: Optional[str] = None
    python_version: Optional[str] = None
    pip_freeze: Optional[str] = None
    device_info: Optional[str] = None
    dataset_id: Optional[str] = None
    params: Dict[str, Any]
    final_metrics: Dict[str, Any]
    artifacts: List[str]
    wandb_url: Optional[str] = None
    note: Optional[str] = None


class RunComparison(BaseModel):
    run_ids: List[str]
    metric_keys: List[str]
    param_keys: List[str]
    runs: List[RunDetail]


class NoteUpdate(BaseModel):
    note: str
