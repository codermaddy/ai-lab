# AI-Lab — Lightweight Experiment Tracker + FastAPI Backend + OpenAI-Powered Agent

`ai-lab` is a **simple, local-first experiment tracking stack**.
It contains:

* **Lab Logger** — a decorator that records each ML run into a manifest database
* **SQLite Database** — single-file manifest store (`manifests.db`)
* **FastAPI Backend** — clean JSON endpoints for querying and comparing experiments
* **LangChain Agent (OpenAI)** — uses backend endpoints as tools
* **Streamlit UI** — dashboards, comparisons, AI assistant

---

# 1. Project Structure

```
ai-lab/
│
├── backend/
│   ├── api.py               # FastAPI app + routes
│   ├── db_utils.py          # SQLite querying / filtering
│   ├── models.py            # Pydantic schemas
│   └── config.py            # PROJECT_NAME, DB_PATH, OpenAI keys, ENV loading
│
├── libs/
│   └── lab_logger/
│       ├── __init__.py      # exposes log_run
│       └── core.py          # logger decorator → writes to SQLite
│
├── agents/
│   ├── agents.py            # Pydantic AgentAnswer, ComparisonResult
│   ├── orchestrator.py      # LangChain agent (Planner + Tools + Answer)
│   └── chat_agent.py        # FastAPI router exposing POST /agent/query
│
├── tools/
│   └── langchain_tools.py   # LangChain tools calling backend HTTP APIs
│
├── examples/
│   └── train_model/
│       ├── run.py           # single-run trainer
│       ├── sweep.py         # multi-run sweeps
│       └── checkpoints/     # metrics_*.csv, model_*.joblib
│
├── manifests/
│   └── manifests.db
│
├── scripts/
│   ├── check_latest_runs.py
│   ├── inspect_metrics_artifacts.py
│   └── test_agent.py
│
├── stream.py               #  Streamlit UI
├── .env
└── requirements.txt
```

---

# 2. Environment Setup

### `.env` (minimal)

```env
PROJECT_NAME=ai-lab
DB_PATH=manifests/manifests.db
WANDB_PROJECT=ai-lab
WANDB_API_KEY= <Insert_key_here>
OPENAI_API_KEY= <Insert_key_here> ## Agent LLM uses OpenAI
```

### Install:

```bash
conda create -n ai-lab-notebook python=3.10
conda activate ai-lab-notebook
pip install -r requirements.txt
```

---

# 3. Manifest Database (SQLite)

All experiment metadata is written to a single table: **`manifests`**.

### 3.1 Schema used by `db_utils.py`

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
    params_json TEXT,           -- kwargs of training fn
    final_metrics_json TEXT,    -- final metrics dict
    artifacts_json TEXT,        -- list of artifact paths
    wandb_url TEXT,
    note TEXT
);
```

Everything except scalars is JSON-encoded.

---

# 4. DB Access Layer — `backend/db_utils.py`

This file is the *authoritative* interface to SQLite.

### 4.1 `get_connection()`

Creates SQLite connection with timeout (prevents “database is locked” on multiple readers).

### 4.2 `_row_to_detail(row)`

Internal — converts a row into:

```json
{
  "run_id": "...",
  "timestamp": "...",
  "git_hash": "...",
  "git_diff": "...",
  "command": "...",
  "python_version": "...",
  "pip_freeze": "...",
  "device_info": "...",
  "dataset_id": "...",
  "params": {...},
  "final_metrics": {...},
  "artifacts": [...],
  "wandb_url": "...",
  "note": "..."
}
```

---

## 4.3 `list_runs(...)`

This is the backend for `GET /runs`.

Supports SQL filtering:

* `dataset_id`
* `command_substring`
* `timestamp >= from_timestamp`
* `timestamp <= to_timestamp`
* `note_substring`

Then Python filtering on:

* `has_artifacts`
* numeric filtering:

  * `metric_name + metric_min`
  * `metric_name + metric_max`

Returns **summaries**, not full details:

```python
{
  "run_id": ...,
  "timestamp": ...,
  "git_hash": ...,
  "command": ...,
  "dataset_id": ...,
  "wandb_url": ...,
  "final_metrics": {...}
}
```

---

## 4.4 `get_run(run_id)`

Backend for `GET /runs/{run_id}`.
Returns full detail (via `_row_to_detail`).

---

## 4.5 `get_runs_by_ids(run_ids)`

Used by comparison.
Bulk-fetches multiple runs and returns full details.

---

## 4.6 `leaderboard(metric_name, top_k, dataset_id, has_artifacts)`

Backend for `GET /runs/leaderboard`.

Logic:

* filter by dataset_id (SQL)
* parse metrics JSON
* filter out rows without that metric
* sort **descending** by metric value
* return top_k

---

## 4.7 `update_run_note(run_id, note)`

Backend for `PATCH /runs/{id}/note`.
Updates the `note` column.

---

# 5. Logging Layer — `lab_logger`

`@log_run` is used to decorate training functions.

### What it captures:

* run metadata:

  * command, timestamp, git hash, git diff
  * Python version, pip freeze, device info
* all function kwargs → `params_json`
* return value:

  * `final_metrics`
  * `artifacts` (metrics CSV, checkpoint files)
* dataset ID (if provided)
* writes to SQLite via `INSERT INTO manifests`

This is the *only* write path; everything else is read-only.

---

# 6. Training Scripts — `examples/train_model/`

### `run.py`

* Models: logistic regression, linear SVM, perceptron
* Loops epochs with `partial_fit`
* Logs train/val metrics
* Writes:

  * `metrics_<timestamp>.csv`
  * `<model>_model_<timestamp>.joblib`
* Decorated with `@log_run()` → automatically saved to DB

Run example:

```bash
python examples/train_model/run.py \
  --experiment_name iris_svm \
  --model_name linear_svm \
  --n_epochs 30
```

### `sweep.py`

Loops over multiple models, seeds:

```bash
python examples/train_model/sweep.py
```

Populates DB with many runs.

---

# 7. FastAPI Backend — `backend/api.py`

Start the backend:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints 

#### `GET /health`

Returns `{ "status": "ok", "project": PROJECT_NAME }`.

#### `GET /runs`

* Uses `db_utils.list_runs`
* Supports all filters from section 4.3

#### `GET /runs/{run_id}`

Full run details (params, metrics, artifacts, note).

#### `GET /runs/leaderboard`

Arguments:

* `metric_name`
* `top_k`
* optional: `dataset_id`, `has_artifacts`

#### `POST /compare`

Takes:

```json
{ "run_ids": ["id1", "id2"] }
```

Returns a `RunComparison` Pydantic object:

* union of parameter keys
* union of metric keys
* per-run detail structures

#### `PATCH /runs/{run_id}/note`

* Body: `{ "note": "PUBLISH" }`
* Uses `db_utils.update_run_note`

#### `POST /agent/query`

Handled by LangChain agent.
Input:

```json
{ "query": "Compare my best two SVM runs" }
```

Output is `AgentAnswer`:

```json
{
  "intent": "...",
  "natural_language_answer": "...",
  "used_run_ids": [...],
  "comparison": {...},
  "flagged_run_id": null
}
```

---


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

## 9.2 Testing the Agent — `scripts/test_agent.py`

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

# 10. UI / Frontend Integration

You can integrate **any UI** (Streamlit, React, Vue, SwiftUI, Flutter, etc.) with this system.  
There are **two recommended integration layers**, depending on how much control you want:  
**(A) Chat-style natural language interface**, or **(B) direct structured API access**.

---

## 10.1 Using the Agent Endpoint (`POST /agent/query`)

This is the easiest and most flexible method.  
Perfect for **chatbots**, **assistant panels**, or **query-driven dashboards**.

### Example (Python pseudo-code)

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

## 10.2 Using Core Backend Endpoints Directly

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


# 11. LangChain Tools Layer — `tools/langchain_tools.py`

These tools call backend endpoints via HTTP.

Tools include:

* `list_runs_tool(...)` → GET /runs
* `leaderboard_tool(...)` → GET /runs/leaderboard
* `get_run_details_tool(run_id)` → GET /runs/{run_id}
* `compare_runs_tool(run_ids)` → POST /compare
* `update_note_tool(run_id, note)` → PATCH /runs/{id}/note

Agent **never** touches SQLite.
It only uses these HTTP tools.

---

# 12. LangChain Agent — `agents/`

### `agents/orchestrator.py`

* Registers tools
* Uses **OpenAI** for:

  * Planner (tool-selection)
  * Final Answer generator
* Ensures responses conform to `AgentAnswer`

### `agents/chat_agent.py`

Exposes:

```
POST /agent/query
```

Response always in structured JSON (`AgentAnswer`).

### `agents/agents.py`

Defines schemas:

* `AgentAnswer`
* `ComparisonResult`
* `RunSummaryView`
* etc.

---

# 13. Testing the Agent — `scripts/test_agent.py`

Run backend first:

```bash
uvicorn backend.api:app --reload --port 8000
```

Then:

```bash
python scripts/test_agent.py
```

It will:

* send 4–5 test queries to `/agent/query`
* print status codes
* print parsed JSON responses

If everything is correct → `STATUS: 200` for all.

---

# 14. Streamlit UI — `stream.py`

Tabs include:

1. **Runs Explorer** — uses `/runs`
2. **Run Detail View** — uses `/runs/{id}` and reads metrics CSV
3. **Compare Runs** — uses `/compare` and plots loss curves, other metrices
4. **Assistant** — calls `/agent/query`

Run with:

```bash
streamlit run stream.py
```

---

Here is a **minimal, clean, ready-to-paste** version for your README — no fluff, only the required steps + commands.

---

# 15. Minimal Workflow Summary

1. **Create `.env`**

```env
PROJECT_NAME=ai-lab
DB_PATH=manifests/manifests.db
WANDB_PROJECT=ai-lab
WANDB_API_KEY= <Insert_key_here>
OPENAI_API_KEY= <Insert_key_here> ## Agent LLM uses OpenAI
```

2. **Install dependencies**

```bash
conda create -n ai-lab-notebook python=3.10 -y
conda activate ai-lab-notebook
pip install -r requirements.txt
```

3. **Run experiments (populate DB)**

```bash
python examples/train_model/sweep.py  ## better
# or
python examples/train_model/run.py --experiment_name test --model_name linear_svm --n_epochs 30
```

4. **Start backend**

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

5. **Start Streamlit UI in a fresh terminal**

```bash
streamlit run stream.py
```

6. **Alt you can use the agent via API directly**

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List my top runs"}'
```

7. **Everything is stored in**

```
manifests/manifests.db
```

---


# 16. Features Supported

* Local manifest database
* Complete run metadata
* Final metrics JSON
* Artifact paths (CSV, models)
* Leaderboard and filtering
* Multi-run comparison
* Notes / tagging
* OpenAI-powered tool-using agent
* UI with Dashboard and Agent

---

# 17. Future Scope

Planned and possible future enhancements:

* **Artifacts Browser**
  UI panel to view and download model files, metrics CSVs, logs.

* **Rich Metrics Dashboard**
  Time-series charts, metric aggregation, smoothing, anomaly detection.

* **Background Task Queue**
  For running sweeps asynchronously and tracking progress.

* **Notifications & Alerts**
  Notify when a new best run is logged based on a metric threshold.

* **Search Indexing**
  Embedding-based search over experiment parameters and notes.

* **Extended Agent Abilities**

  * “Find experiments similar to X”
  * “Suggest hyperparameters for best accuracy”
  * Auto-generate result summaries for reports.

* **Authentication + Multi-user mode**
  For teams sharing the backend.

---

## Acknowledgements

* **Inspired by lightweight W&B-like workflows**, but designed to be fully local, simple, and transparent.
* **OpenAI API** is used as the default LLM backend for the agent layer.
* **LangChain** powers the tool-call planning system.
* **FastAPI** provides the backend API layer.
* **SQLite** enables a portable, zero-config experiment store.
* **Streamlit** (optional) enables rapid UI prototyping for dashboards and run comparison.

---


### Architecture Diagram (ASCII)

