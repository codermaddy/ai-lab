# mock_backend.py

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

app = FastAPI()

# ---------- Fake data ----------

MOCK_RUNS: Dict[str, Dict[str, Any]] = {
    "run_cifar10_lr1e-3_bs64": {
        "run_id": "run_cifar10_lr1e-3_bs64",
        "dataset": "cifar10",
        "model_name": "resnet18",
        "best_metric_name": "val_accuracy",
        "best_metric_value": 0.86,
        "hyperparams": {"lr": 1e-3, "batch_size": 64},
        "metrics": {"val_accuracy": 0.86, "val_loss": 0.55},
        "anomalies": [],
        "flagged_for_publish": False,
    },
    "run_cifar10_lr3e-4_bs128": {
        "run_id": "run_cifar10_lr3e-4_bs128",
        "dataset": "cifar10",
        "model_name": "resnet18",
        "best_metric_name": "val_accuracy",
        "best_metric_value": 0.89,
        "hyperparams": {"lr": 3e-4, "batch_size": 128},
        "metrics": {"val_accuracy": 0.89, "val_loss": 0.48},
        "anomalies": [],
        "flagged_for_publish": False,
    },
    "run_mnist_lr1e-3_bs32": {
        "run_id": "run_mnist_lr1e-3_bs32",
        "dataset": "mnist",
        "model_name": "cnn_small",
        "best_metric_name": "val_accuracy",
        "best_metric_value": 0.97,
        "hyperparams": {"lr": 1e-3, "batch_size": 32},
        "metrics": {"val_accuracy": 0.97, "val_loss": 0.05},
        "anomalies": ["slight overfitting"],
        "flagged_for_publish": False,
    },
}

MOCK_SUMMARIES: Dict[str, str] = {
    "run_cifar10_lr1e-3_bs64": (
        "CIFAR-10 run with ResNet18, lr=1e-3, bs=64. "
        "Reached val_accuracy=0.86, val_loss=0.55. Stable training."
    ),
    "run_cifar10_lr3e-4_bs128": (
        "CIFAR-10 run with ResNet18, lr=3e-4, bs=128. "
        "Reached val_accuracy=0.89, val_loss=0.48. Best CIFAR-10 run so far."
    ),
    "run_mnist_lr1e-3_bs32": (
        "MNIST run with small CNN, lr=1e-3, bs=32. "
        "Reached val_accuracy=0.97 with mild overfitting."
    ),
}

# ---------- Schemas for responses ----------

class Run(BaseModel):
    run_id: str
    dataset: str
    model_name: str
    best_metric_name: str
    best_metric_value: float
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]
    anomalies: List[str]
    flagged_for_publish: bool


class RunSummary(BaseModel):
    run_id: str
    dataset: str
    summary: str


class CompareResult(BaseModel):
    run_a: Run
    run_b: Run
    metric_diffs: Dict[str, float]
    hyperparam_diffs: Dict[str, Dict[str, Any]]


# ---------- Endpoints mimicking DB contract ----------

@app.get("/runs", response_model=List[Run])
def list_runs(
    dataset: Optional[str] = None,
    time_range: Optional[str] = None,  # ignored in mock
    tag: Optional[str] = None,         # ignored in mock
):
    runs = list(MOCK_RUNS.values())
    if dataset:
        runs = [r for r in runs if r["dataset"] == dataset]
    return runs


@app.get("/runs/{run_id}", response_model=Run)
def get_run_details(run_id: str):
    if run_id not in MOCK_RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    return MOCK_RUNS[run_id]


@app.get("/runs/compare", response_model=CompareResult)
def compare_runs(a: str, b: str):
    if a not in MOCK_RUNS or b not in MOCK_RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    run_a = MOCK_RUNS[a]
    run_b = MOCK_RUNS[b]

    metric_diffs: Dict[str, float] = {}
    for k, va in run_a["metrics"].items():
        vb = run_b["metrics"].get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            metric_diffs[k] = vb - va

    hyperparam_diffs: Dict[str, Dict[str, Any]] = {}
    for k in set(run_a["hyperparams"].keys()) | set(run_b["hyperparams"].keys()):
        hyperparam_diffs[k] = {
            "a": run_a["hyperparams"].get(k),
            "b": run_b["hyperparams"].get(k),
        }

    return {
        "run_a": run_a,
        "run_b": run_b,
        "metric_diffs": metric_diffs,
        "hyperparam_diffs": hyperparam_diffs,
    }


@app.post("/runs/{run_id}/flag", response_model=Run)
def flag_run_for_publish(run_id: str):
    if run_id not in MOCK_RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    MOCK_RUNS[run_id]["flagged_for_publish"] = True
    return MOCK_RUNS[run_id]


@app.get("/runs/summaries", response_model=List[RunSummary])
def run_summaries():
    out = []
    for run_id, summary in MOCK_SUMMARIES.items():
        run = MOCK_RUNS[run_id]
        out.append(
            {
                "run_id": run_id,
                "dataset": run["dataset"],
                "summary": summary,
            }
        )
    return out