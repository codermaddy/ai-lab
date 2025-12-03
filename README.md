Understood.
Here is a **detailed, clean, professional README** — *no fluff, no filler*, exactly matching your real system:

* **SQLite DB**
* **FastAPI backend**
* **OpenAI-powered LangChain agent**
* **lab_logger**
* **db_utils.py** exactly as you pasted
* **runs, comparison, leaderboard, notes endpoints**
* **Streamlit optional**
* Everything documented clearly, technically, and correctly.

This is the final README you can directly paste into your GitHub repo.

---

# AI-Lab — Lightweight Experiment Tracker + FastAPI Backend + OpenAI-Powered Agent

`ai-lab` is a **simple, local-first experiment tracking stack**.
It contains:

* **Lab Logger** — a decorator that records each ML run into a manifest database
* **SQLite Database** — single-file manifest store (`manifests.db`)
* **FastAPI Backend** — clean JSON endpoints for querying and comparing experiments
* **LangChain Agent (OpenAI)** — uses backend endpoints as tools
* **Optional Streamlit UI** — dashboards, run browser, comparisons, assistant

Everything is meant to be small, inspectable, and hackable.
No external infra is required.

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
├── streaml.py               # Optional Streamlit UI
├── .env
└── requirements.txt
```

---

# 2. Environment Setup

### `.env` (minimal)

```env
PROJECT_NAME=ai-lab
DB_PATH=manifests/manifests.db

OPENAI_API_KEY=YOUR_KEY_HERE     # Agent LLM uses OpenAI
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

### Endpoints (exact logic matches your code)

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

# 8. LangChain Tools Layer — `tools/langchain_tools.py`

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

# 9. LangChain Agent — `agents/`

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

# 10. Testing the Agent — `scripts/test_agent.py`

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

# 11. Optional Streamlit UI — `streaml.py`

This UI uses only backend endpoints, not SQLite.

Tabs normally include:

1. **Runs Explorer** — uses `/runs`
2. **Run Detail View** — uses `/runs/{id}` and reads metrics CSV
3. **Compare Runs** — uses `/compare`
4. **Assistant** — calls `/agent/query`

Run with:

```bash
streamlit run streaml.py
```

---

# 12. Minimal Workflow Summary

1. Create `.env`

2. Install dependencies

3. Run experiments (`run.py` / `sweep.py`)

4. Start backend:

   ```bash
   uvicorn backend.api:app --reload --port 8000
   ```

5. (Optional) Start Streamlit

6. Use `/agent/query` for an intelligent interface

7. Everything is stored in `manifests.db`

---

# 13. Features Supported

* Local manifest database
* Complete run metadata
* Final metrics JSON
* Artifact paths (CSV, models)
* Leaderboard and filtering
* Multi-run comparison
* Notes / tagging
* OpenAI-powered tool-using agent
* Optional UI

---

If you want, I can also generate:

✅ a shorter README
✅ an "Architecture Diagram" section (ASCII or image)
✅ a "Quickstart in 30 seconds" section
Just say the word.
