# ai-lab

# AI Lab Notebook — Backend + Experiment Logger

This repository implements a lightweight **experiment tracking system** designed for ML workflows.
It consists of:

1. **A logging decorator** (`lab_logger`)
2. **A local SQLite manifest database**
3. **A FastAPI backend** for querying runs
4. **Training scripts** that write metrics + artifacts
5. **Agent/UI-friendly JSON APIs** for comparison, leaderboard, tagging, filtering

Everything is fully local and W&B-optional.

---

# 1. Directory Structure

```
ai-lab/
│
├── libs/lab_logger/              # logging library (editable install)
│    └── lab_logger/core.py       # main logger
│
├── manifests/
│    └── manifests.db             # SQLite database (auto-created)
│
├── backend/
│    ├── api.py                   # FastAPI routes
│    ├── db_utils.py              # DB access helpers
│    ├── models.py                # Pydantic schemas
│    └── config.py                # loads .env, project name, DB path
│
├── examples/
│    └── train_model/
│          ├── run.py             # multi-model, multi-epoch trainer
│          ├── sweep.py           # run many experiments
│          └── checkpoints/       # model.ckpt + metrics.csv are saved here
│
├── scripts/
│    ├── check_latest_runs.py     # verify params + artifacts
│    └── inspect_metrics_artifacts.py   # inspect metrics.csv contents
│
├── .env
└── requirements.txt
```

---

# 2. .env Configuration

Create a `.env` file at repo root:

```env
PROJECT_NAME=ai-lab
DB_PATH=manifests/manifests.db
WANDB_PROJECT=ai-lab
WANDB_API_KEY=YOUR_KEY_HERE   # optional
```

Do **not** commit `.env`.

Load the environment:

```bash
export $(cat .env | xargs)
```

---

# 3. The Lab Logger (core.py)

`@log_run()` decorates any function and logs:

**Automatically captured:**

* `run_id`
* `timestamp`
* `git_hash`
* `git_diff`
* `command` used to launch the script
* `python_version`
* `pip_freeze`
* `device_info`
* `dataset_id` (from env)
* **W&B run URL** if enabled
* **Artifacts** (paths you return)
* **Final metrics** (dict you return)
* **Params** (all kwargs to the decorated function)
* `note` (tagging, modifiable via API)

Everything is written to `manifests/manifests.db`.

---

# 4. Manifest Database Schema

SQLite table: **manifests**

```sql
CREATE TABLE manifests (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT,
    git_hash TEXT,
    git_diff TEXT,
    command TEXT,
    python_version TEXT,
    pip_freeze TEXT,
    device_info TEXT,
    dataset_id TEXT,
    params_json TEXT,          -- {"experiment_name": "...", "model_name": "...", ...}
    final_metrics_json TEXT,   -- {"accuracy": ..., "train_loss": ..., ...}
    artifacts_json TEXT,       -- ["path/to/ckpt", "path/to/metrics.csv"]
    wandb_url TEXT,
    note TEXT
)
```

Any training run = one row.

---

# 5. Training Script (`examples/train_model/run.py`)

Supports:

* **Multiple models**: logistic regression, SVM (hinge), perceptron
* **True multi-epoch training** (via SGD `partial_fit`)
* **Epoch-wise train + val loss logging**
* **CSV artifact**: `metrics_<timestamp>.csv`
* **Checkpoint artifact**: `<model_name>_model_<timestamp>.joblib`
* **Semantic parameters** stored in DB:

  * `experiment_name`
  * `model_name`
  * `task_name`
  * `n_epochs`
  * `learning_rate`
  * etc.

Example run:

```bash
python examples/train_model/run.py
```

Example sweep:

```bash
python examples/train_model/sweep.py
```

Artifacts created:

```
examples/train_model/checkpoints/
    logistic_regression_model_1733xxxx.joblib
    logistic_regression_metrics_1733xxxx.csv
    ...
```

`metrics_*.csv` contains:

| epoch | train_loss | val_loss |
| ----- | ---------- | -------- |
| 1     | ...        | ...      |
| 2     | ...        | ...      |
| ...   | ...        | ...      |

This is what the Streamlit agent will read to plot loss curves and compare models.

---

# 6. FastAPI Backend (for agent/UI)

Start server:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### **1. List runs**

```
GET /runs
```

Filters supported:

* `dataset_id`
* `command_substring`
* `from_timestamp`, `to_timestamp`
* `has_artifacts=true/false`
* `metric_name=accuracy&metric_min=0.9`
* `note_substring=PUBLISH`

#### **2. Run details**

```
GET /runs/{run_id}
```

Returns everything:

* params (model/experiment/task)
* final_metrics
* artifacts (checkpoint, metrics.csv)
* git info
* pip freeze
* note

#### **3. Compare runs**

```
GET /runs/compare?ids=ID1&ids=ID2&ids=ID3
```

Returns:

* union of all metric keys
* union of all param keys
* full RunDetail for each run

Perfect for comparison tables.

#### **4. Leaderboard**

```
GET /runs/leaderboard?metric_name=accuracy&top_k=10
```

Sorts runs (descending) by any metric.

#### **5. Tagging**

```
PATCH /runs/{id}/note
{
  "note": "PUBLISH: best model"
}
```

#### **6. Health**

```
GET /health
```

---

# 7. What the Agent/UI Engineer Needs to Know

Your friend who will build the agent only needs the following facts:

### **1. How to fetch all runs**

```
GET /runs
```

Returns summary of all experiments.

### **2. How to fetch full details for a run**

```
GET /runs/{run_id}
```

From this:

* `params["model_name"]` → which model
* `params["experiment_name"]`
* `params["task_name"]`
* artifacts → metrics csv path, checkpoint path
* final_metrics → accuracy, f1, runtime

### **3. How to get artifacts**

`/runs/{id}` returns:

```json
"artifacts": [
  "examples/train_model/checkpoints/logistic_regression_model_....joblib",
  "examples/train_model/checkpoints/logistic_regression_metrics_....csv"
]
```

The agent must:

* identify metrics CSV → ends with `metrics_*.csv`
* read CSV
* plot loss curves
* compare multiple runs

### **4. How to compare runs**

```
GET /runs/compare?ids=A&ids=B&ids=C
```

Use:

* `metric_keys`
* `param_keys`
* `runs[...]`

to build tables.

### **5. How to create leaderboards**

```
GET /runs/leaderboard?metric_name=accuracy&top_k=20
```

### **6. How to filter runs**

Examples:

* By model:
  Agent inspects `params["model_name"]`.
* By task:
  Agent inspects `params["task_name"]`.
* By experiment:
  Agent inspects `params["experiment_name"]`.
* By date:
  `/runs?from_timestamp=2025-12-01`

### **7. Tag important runs**

```
PATCH /runs/{id}/note
```

The UI can expose “Mark as PUBLISH / BEST / BAD RUN”.

### **8. How to interpret results**

The agent simply needs to:

* Call `/runs` or `/runs/{id}`
* Examine `params` + `final_metrics`
* Look for metrics CSV in artifacts
* Load numbers → plot → answer questions

No further DB access, no local code execution.

---

# 8. Verification Scripts

### Show params, metrics, artifacts for latest runs:

```bash
python scripts/check_latest_runs.py
```

### Inspect metrics.csv for a run:

```bash
python scripts/inspect_metrics_artifacts.py
```

---

# 9. How to Run Everything (Summary)

### Run sweeps / experiments:

```bash
python examples/train_model/sweep.py
```

### Start backend:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Query runs:

```bash
curl http://localhost:8000/runs
curl http://localhost:8000/runs/<run_id>
curl "http://localhost:8000/runs/leaderboard?metric_name=accuracy"
```

---

# 10. What This System Supports (Final Summary)

✔ Multi-model training
✔ Multi-epoch losses (metrics.csv)
✔ Checkpoints
✔ Semantic params for agent (model/experiment/task)
✔ SQLite manifest DB
✔ Rich filtering
✔ Leaderboard
✔ Run comparison
✔ Tagging
✔ W&B optional
✔ Everything agent-friendly (JSON, predictable schema)

---

If you want, I can also write a **shorter README.md** version or a **diagram** showing the flow:

Training Script → Logger → SQLite → FastAPI → Streamlit Agent.

Just tell me.
