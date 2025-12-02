# ai-lab
Below is a **clean, production-quality README** you can paste directly into your GitHub repo.
It explains the workflow, DB schema, endpoints, artifacts, training scripts, and exactly what your friend (agent builder) needs to know.

I’ve kept it crisp, well-structured, and professional.

---

# AI Lab Notebook — Backend + Experiment Logger + LangChain Agent

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
├── app.py                      # FastAPI entrypoint (mounts core API + agent API)
│
├── backend/
│   ├── api.py                  # Core REST endpoints (/runs, /leaderboard, etc.)
│   ├── db_utils.py             # DB access helpers (SQLite)
│   ├── models.py               # Pydantic schemas for runs, metrics, etc.
│   └── config.py               # loads .env, project name, DB path
│
├── libs/
│   └── lab_logger/
│       ├── __init__.py         # exposes log_run()
│       └── core.py             # main logger decorator (writes to manifests.db)
│
├── agents/
│   ├── __init__.py
│   ├── agents.py               # Pydantic models for AgentAnswer, ComparisonResult
│   ├── orchestrator.py         # LangChain-based SimpleAgent (planner + tools + answer)
│   └── chat_agent.py           # FastAPI router exposing POST /agent/query
│
├── tools/
│   ├── __init__.py
│   └── langchain_tools.py      # LangChain @tool wrappers over HTTP backend (/runs, /leaderboard, etc.)
│
├── manifests/
│   └── manifests.db            # SQLite database (auto-created)
│
├── examples/
│   └── train_model/
│       ├── run.py              # multi-model, multi-epoch trainer
│       ├── sweep.py            # run many experiments, logs multiple runs
│       └── checkpoints/        # model_*.joblib + metrics_*.csv
│
├── scripts/
│   ├── check_latest_runs.py        # inspect latest runs from DB
│   ├── inspect_metrics_artifacts.py# inspect metrics_*.csv artifacts
│   └── test_agent.py               # simple CLI tester for /agent/query
│__ app.py  $ The backend app runs from here
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
OLLAMA_BASE_URL=http://localhost:11434
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
'
## Running Experiments

### 6.1 Install dependencies

```bash
conda create -n ai-lab-notebook python=3.10
conda activate ai-lab-notebook
pip install -r requirements.txt
```

### 6.2 Run Ollama server
```bash
python -m servers.ollama_server
```

## 8. LangChain Tools Layer (`tools/langchain_tools.py`)

This module exposes the backend HTTP API as **LangChain tools**, using the `@tool` decorator.  
These tools are the **only** way the LLM interacts with experiment data — the agent never touches SQLite directly.

### Available Tools

| Tool Name | Backend API Called | Purpose |
|----------|--------------------|---------|
| `list_runs(...)` | `GET /runs` | List/filter runs by dataset, command substring, notes, etc. |
| `leaderboard(metric_name, top_k, ...)` | `GET /runs/leaderboard` | Return top runs sorted by a metric (e.g., accuracy). |
| `get_run_details(run_id)` | `GET /runs/{run_id}` | Retrieve a run’s full metadata, params, artifacts, and metrics. |
| `compare_runs(run_ids)` | `GET /runs/compare?ids=...` | Return unified metric/param keys and full details for comparison tables. |
| `flag_run_for_publish(run_id)` | `PATCH /runs/{id}/note` | Tag a run as `"PUBLISH"` (or for later UI display). |
| `search_run_summaries(query, limit)` | `GET /runs` | Build textual summaries and perform simple retrieval over runs. |

These tools convert backend JSON responses into agent-usable structures.

---

## 9. Agent Layer (LangChain)

The agent layer consists of:

- **`agents/agents.py`** — Pydantic schemas for agent output  
- **`agents/orchestrator.py`** — Planner + Answer LLM chains (tool-calling logic)  
- **`agents/chat_agent.py`** — FastAPI wrapper exposing `/agent/query`

---

## 9.1 `agents/agents.py` — Agent Output Schema

Defines structured Pydantic models that validate whatever the LLM produces.

### Key Models

#### **AgentAnswer**
```json
{
  "intent": "best_run",
  "natural_language_answer": "...",
  "used_run_ids": ["..."],
  "comparison": { ... } | null,
  "flagged_run_id": "..." | null
}
```

## 11. Testing the Agent — `scripts/test_agent.py`

This script provides a **quick, automated way to verify that the LangChain agent, its tools, and the FastAPI backend are working correctly together**.

It simulates a client calling the `/agent/query` endpoint and prints the structured JSON response.

---

### Location- scripts/test_agent.py

---

## 🔍 What This Script Does

1. **Sends natural-language queries** to your running FastAPI server (`http://localhost:8000/agent/query`).
2. **Receives the agent’s structured JSON output**, validated by Pydantic.
3. Confirms:
   - The **Planner** correctly selects a tool.
   - The backend **tool execution succeeds**.
   - The **Answer chain** returns valid `AgentAnswer` JSON.
4. Shows clear console output for debugging:
   - HTTP status code  
   - Raw JSON returned by the agent  
   - Any errors if JSON is malformed  

This is your “smoke test” to ensure the entire stack is healthy.

---

## 🧪 What It Tests Internally

| Component | Verified? | How |
|----------|-----------|------|
| **LangChain Planner** | ✔ | Does it pick `leaderboard`, `list_runs`, etc.? |
| **LangChain Tools** | ✔ | Do they hit `/runs`, `/runs/leaderboard`, `/runs/{id}` correctly? |
| **Backend API** | ✔ | Are endpoints returning valid JSON? |
| **Agent Answer Chain** | ✔ | Is the final JSON valid according to `AgentAnswer` schema? |
| **FastAPI router** | ✔ | Does `/agent/query` handle requests properly? |

If *any* part of the stack breaks, the script will either:
- Return a **500 error**, or  
- Fail JSON parsing → meaning the agent returned invalid output.

This makes debugging easy.

---

## 🧾 Example Structure of the Script

A simplified conceptual version:

```python
import requests

def ask(query: str):
    payload = {"query": query}
    resp = requests.post("http://localhost:8000/agent/query", json=payload)
    print("STATUS:", resp.status_code)
    print("RESPONSE:")
    print(resp.json())    # ensures JSON is valid

if __name__ == "__main__":
    ask("List my top runs by accuracy.")
    ask("Compare the best two runs.")
    ask("Flag the best run for publish.")
    ask("Summarize my recent experiments.")
```

Each call prints agent output:

```python
STATUS: 200
RESPONSE:
{
  "intent": "best_run",
  "natural_language_answer": "...",
  "used_run_ids": [...],
  "comparison": null,
  "flagged_run_id": null
}
```

#### How to Run It
Start your backend:
```bash
uvicorn app:app --reload --port 8000
```
Then run the script:
```python
python -m scripts.test_agent
```

You should see:
	•	STATUS: 200
	•	Valid JSON printed from the agent

If you get STATUS: 500 or JSONDecodeError → the agent chain likely crashed.

## 12. UI / Frontend Integration

You can integrate **any UI** (Streamlit, React, Vue, SwiftUI, Flutter, etc.) with this system.  
There are **two recommended integration layers**, depending on how much control you want:  
**(A) Chat-style natural language interface**, or **(B) direct structured API access**.

---

# 12.1 Using the Agent Endpoint (`POST /agent/query`)

This is the easiest and most flexible method.  
Perfect for **chatbots**, **assistant panels**, or **query-driven dashboards**.

### ▶️ Example (Python pseudo-code)

```python
import requests

BASE_URL = "http://localhost:8000"

def query_agent(user_text: str):
    resp = requests.post(
        f"{BASE_URL}/agent/query",
        json={"query": user_text},
        timeout=60,
    )
    data = resp.json()
    print(data["natural_language_answer"])
    print("Runs referenced:", data["used_run_ids"])
 ```
Typical UI Flow
	1.	User types:
“Show me my top 3 runs by accuracy and compare them.”
	2.	UI sends:
```jsunicoderegexp
POST /agent/query
{ "query": "Show me my top 3 runs by accuracy and compare them." }
```

3. **Agent returns JSON:**

- `natural_language_answer` → text to display in the UI  
- `used_run_ids` → list of related runs  

4. **If deeper visualization is needed, the UI can then call:**

- `GET /runs/{id}`  
- `GET /runs/compare?ids=...`  

Using these, the UI can build:

- comparison tables  
- metric charts  
- model card views  
- artifact previews  

---

# 12.2 Using Core Backend Endpoints Directly

This gives **full control**, ideal for:

- Streamlit dashboards  
- custom charts  
- experiment browsers  
- Jupyter analysis  

---

## Example Workflow (Streamlit / Python)

```python
import pandas as pd
import requests

BASE_URL = "http://localhost:8000"

# 1) Fetch leaderboard
runs = requests.get(
    f"{BASE_URL}/runs/leaderboard?metric_name=accuracy&top_k=5"
).json()

# 2) Select one run
run_id = runs[0]["run_id"]
detail = requests.get(f"{BASE_URL}/runs/{run_id}").json()

# 3) Locate metrics CSV for plotting
metrics_csv = [
    p for p in detail["artifacts"]
    if "metrics_" in p and p.endswith(".csv")
][0]

df = pd.read_csv(metrics_csv)

# Now you can plot:
# df["epoch"], df["train_loss"], df["val_loss"]
```

# What You Get From Backend Endpoints

| **Endpoint**                   | **Useful For**                                 |
|-------------------------------|------------------------------------------------|
| `GET /runs`                   | Populate full experiment tables                |
| `GET /runs/{id}`              | Show details, params, metrics, artifacts       |
| `GET /runs/leaderboard`       | Leaderboard charts                             |
| `GET /runs/compare?ids=...`   | Comparison heatmaps, tables                    |
| `PATCH /runs/{id}/note`       | Marking runs as **BEST** / **PUBLISH**         |


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
