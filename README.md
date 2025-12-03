````markdown
# ai-lab

AI-Lab is a lightweight **experiment tracking + analysis + agentic** framework for ML workflows.

It gives you:

- A **logging decorator** (`lab_logger`) for training scripts  
- A local **SQLite experiment database**  
- A **FastAPI backend** with clean JSON APIs  
- A **LangChain agent** that uses those APIs as tools  
- Optional **Streamlit / other UI** that talks to the backend or the agent

Everything is local-first and W&B-optional.

---

# 1. Directory Structure

```bash
ai-lab/
│
├── app.py                      # FastAPI entrypoint (mounts core API + agent API)
│
├── backend/
│   ├── api.py                  # Core REST endpoints (/runs, /compare, /agent/query, etc.)
│   ├── db_utils.py             # DB access helpers (SQLite)
│   ├── models.py               # Pydantic schemas for runs, metrics, comparisons, notes
│   └── config.py               # Loads .env, project name, DB path, base URLs
│
├── libs/
│   └── lab_logger/
│       ├── __init__.py         # Exposes log_run()
│       └── core.py             # Main logger decorator (writes to SQLite)
│
├── agents/
│   ├── __init__.py
│   ├── agents.py               # Pydantic models for AgentAnswer, ComparisonResult, etc.
│   ├── orchestrator.py         # LangChain agent orchestration (planner + tools + answer)
│   └── chat_agent.py           # FastAPI router exposing POST /agent/query
│
├── tools/
│   ├── __init__.py
│   └── langchain_tools.py      # LangChain @tool wrappers over HTTP backend (/runs, /leaderboard, etc.)
│
├── backend_db/
│   └── lab.db                  # SQLite database (auto-created)
│
├── streaml.py                  # Optional Streamlit app (dashboard + comparison + assistant)
│
├── examples/
│   └── train_model/
│       ├── run.py              # Multi-model, multi-epoch trainer
│       ├── sweep.py            # Runs many experiments, logs multiple runs
│       └── checkpoints/        # model_*.joblib + metrics_*.csv
│
├── scripts/
│   ├── check_latest_runs.py        # Inspect latest runs from DB
│   ├── inspect_metrics_artifacts.py# Inspect metrics_*.csv artifacts
│   └── test_agent.py               # Simple CLI tester for /agent/query
│
├── .env
└── requirements.txt
````

> **Note:** Names like `backend_db/lab.db` are illustrative; check `backend/config.py` for the exact default path.

---

# 2. Environment & Configuration

Create a `.env` file at repo root:

```env
PROJECT_NAME=ai-lab
DB_PATH=backend_db/lab.db

# Optional: W&B integration
WANDB_PROJECT=ai-lab
WANDB_API_KEY=YOUR_KEY_HERE

# Optional: local LLM / Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

Do **not** commit `.env`.

On Linux/macOS you can export:

```bash
export $(cat .env | xargs)
```

## 2.1 Python Environment

```bash
conda create -n ai-lab-notebook python=3.10
conda activate ai-lab-notebook
pip install -r requirements.txt
```

## 2.2 (Optional) Run Ollama Server

If your agent uses a local Ollama model:

```bash
python -m servers.ollama_server
```

(Adjust this if you changed the server module path.)

---

# 3. Experiment Database (SQLite)

The backend uses a normalized SQLite schema for experiments.

## 3.1 Tables

### `runs` — one row per experiment

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT,
    experiment_name TEXT,
    model_name TEXT,
    task_name TEXT,
    dataset_id TEXT,
    command TEXT,
    git_hash TEXT,
    git_diff TEXT,
    python_version TEXT,
    pip_freeze TEXT,
    device_info TEXT,
    params_json TEXT,          -- full kwargs of the run
    final_metrics_json TEXT,   -- final metrics (accuracy, f1, loss, etc.)
    artifacts_json TEXT,       -- ["path/to/ckpt", "path/to/metrics.csv"]
    wandb_url TEXT,
    note TEXT                  -- free-form tag, e.g. "PUBLISH", "BAD", etc.
);
```

### `run_metrics` — per-epoch / per-step metrics

```sql
CREATE TABLE run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    epoch INTEGER,
    metric_name TEXT,   -- "train_loss", "val_loss", "accuracy", etc.
    metric_value REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
```

> The exact schema may contain indexes (on `experiment_name`, timestamps, etc.) and extra fields; see `backend/db_utils.py` for details.

---

# 4. Lab Logger (`libs/lab_logger/core.py`)

`@log_run()` is a decorator that you place on your training function.
It automatically logs metadata, metrics, and artifacts into SQLite.

## 4.1 What `log_run()` Captures

**Automatically captured:**

* `run_id` (UUID)
* `timestamp`
* `git_hash` / `git_diff`
* `command` used to launch the script
* `python_version`
* `pip_freeze`
* `device_info`
* `dataset_id` (if provided via env or kwargs)
* **Params** (all kwargs to the decorated function, stored in `params_json`)
* **Final metrics** (whatever your function returns, stored in `final_metrics_json`)
* **Artifacts** (paths returned from your function, stored in `artifacts_json`)
* **W&B run URL** (if enabled)
* `note` (taggable later via API)

Each call to a logged training function:

* Inserts 1 row into `runs`
* Optionally inserts many rows into `run_metrics` (for epoch-wise metrics)

---

# 5. Training Scripts (`examples/train_model/`)

## 5.1 `run.py`

A single training run with:

* Multiple models:

  * logistic regression
  * linear SVM (hinge)
  * perceptron
* Multi-epoch training via `partial_fit`
* Train/validation split
* Epoch-wise metric logging
* Artifact saving:

  * `metrics_<timestamp>.csv`
  * `<model_name>_model_<timestamp>.joblib`
* Parameters stored in DB:

  * `experiment_name`
  * `model_name`
  * `task_name`
  * `n_epochs`
  * `learning_rate`
  * etc.

Run:

```bash
python examples/train_model/run.py \
  --experiment_name iris_svm \
  --model_name linear_svm \
  --n_epochs 30 \
  --learning_rate 0.01
```

Artifacts:

```text
examples/train_model/checkpoints/
    linear_svm_model_1733xxxx.joblib
    linear_svm_metrics_1733xxxx.csv
    ...
```

`metrics_*.csv` typically looks like:

| epoch | train_loss | val_loss | accuracy |
| ----- | ---------: | -------: | -------: |
| 1     |        ... |      ... |      ... |
| 2     |        ... |      ... |      ... |
| ...   |        ... |      ... |      ... |

## 5.2 `sweep.py`

Loops over model types, seeds, and hyperparameters:

```python
MODELS = ["logistic_regression", "linear_svm", "perceptron"]

for model_name in MODELS:
    for seed in [0, 1, 2]:
        train_model(
            test_size=0.25,
            random_state=seed,
            n_epochs=30,
            learning_rate=0.01,
            model_name=model_name,
            experiment_name=f"iris_{model_name}_sweep_v1",
        )
```

Run:

```bash
python examples/train_model/sweep.py
```

This will populate the DB with many `runs` and artifact files, which is what the backend, agent, and UI operate on.

---

# 6. FastAPI Backend

There are two typical entrypoints:

```bash
# Option 1: if app.py mounts everything
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Option 2: if backend.api directly defines app
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Check your `app.py` / `backend/api.py` to confirm.

## 6.1 Core Endpoints

### 1. List runs

```http
GET /runs
```

Query parameters:

* `dataset_id`
* `experiment_name`
* `command_substring`
* `from_timestamp`, `to_timestamp`
* `metric_name`, `metric_min`, `metric_max`
* `note_substring`
* pagination: `limit`, `offset`

Returns a list of run summaries (Pydantic `RunSummary`).

---

### 2. Run details

```http
GET /runs/{run_id}
```

Returns (Pydantic `RunDetail`):

* `run_id`, timestamps
* `params`
* `final_metrics`
* `artifacts`
* `wandb_url`
* `note`
* maybe inlined `metrics` or a pointer to them (depending on implementation)

---

### 3. Compare runs

```http
GET /runs/compare?ids=ID1&ids=ID2&ids=ID3
```

Returns (Pydantic `RunComparison`):

* `metric_keys` — union of all metric keys
* `param_keys` — union of parameter keys
* `runs[...]` — details per run, structured for table/heatmap visuals

---

### 4. Leaderboard

```http
GET /runs/leaderboard?metric_name=accuracy&top_k=10
```

Returns top-K runs sorted by the given metric (descending by default).

---

### 5. Tag / update note

```http
PATCH /runs/{run_id}/note
Content-Type: application/json

{
  "note": "PUBLISH: best validation accuracy"
}
```

Uses Pydantic `NoteUpdate`. Useful for marking best runs.

---

### 6. Health checks

```http
GET /health
```

Simple diagnostics:

```json
{
  "status": "ok",
  "project": "ai-lab"
}
```

---

### 7. Agent endpoint

```http
POST /agent/query
Content-Type: application/json

{
  "query": "List my top runs by accuracy and compare the best two."
}
```

Handled by `agents/chat_agent.py`, returns a structured `AgentAnswer`.

---

# 7. LangChain Tools Layer (`tools/langchain_tools.py`)

This module exposes backend HTTP APIs as **LangChain tools** using the `@tool` decorator.
The agent must use these tools — it never touches SQLite directly.

## 7.1 Available Tools (Typical)

| Tool Name                   | Backend API Called            | Purpose                                                |
| --------------------------- | ----------------------------- | ------------------------------------------------------ |
| `list_runs(...)`            | `GET /runs`                   | List/filter runs by dataset, model, note, etc.         |
| `leaderboard(...)`          | `GET /runs/leaderboard`       | Get top-K runs by any metric (e.g., accuracy).         |
| `get_run_details(run_id)`   | `GET /runs/{run_id}`          | Get the full metadata, params, artifacts, metrics.     |
| `compare_runs(run_ids)`     | `GET /runs/compare?ids=...`   | Compute comparison-ready stats for multiple runs.      |
| `flag_run_for_publish(...)` | `PATCH /runs/{id}/note`       | Mark / annotate a run as "PUBLISH", "BAD", etc.        |
| `search_run_summaries(...)` | `GET /runs` (plus some logic) | Lightweight retrieval / semantic search over run text. |

Each tool:

1. Calls the appropriate HTTP endpoint.
2. Parses JSON into Python dataclasses / Pydantic models.
3. Returns data in a stable shape that the agent can reason about.

---

# 8. Agent Layer (LangChain)

The agent turns **natural language questions** into **tool calls + structured answers**.

## 8.1 Components

* **`agents/agents.py`**
  Pydantic schemas for agent I/O, especially `AgentAnswer`.

* **`agents/orchestrator.py`**
  Builds the LangChain agent:

  * Adds tools from `tools/langchain_tools.py`
  * Defines the planner / reasoning LLM
  * Defines the answer LLM (which outputs `AgentAnswer` JSON)

* **`agents/chat_agent.py`**
  Wraps the agent in a FastAPI router and exposes:

  ```http
  POST /agent/query
  {
    "query": "Compare the SVM and logistic regression runs on iris."
  }
  ```

## 8.2 `AgentAnswer` Schema

A typical `AgentAnswer` in `agents/agents.py` looks like:

```python
class AgentAnswer(BaseModel):
    intent: str
    natural_language_answer: str
    used_run_ids: List[str]
    comparison: Optional[ComparisonResult] = None
    flagged_run_id: Optional[str] = None
```

Where `ComparisonResult` might include:

* `metric_keys`
* `param_keys`
* `rows` or `runs` with per-run values

This enforced schema ensures the agent always returns parseable JSON for the UI.

---

# 9. Testing the Agent — `scripts/test_agent.py`

This script is a **smoke test** for:

* FastAPI router (`/agent/query`)
* LangChain tools
* DB → API → tools → agent → JSON

## 9.1 What It Does

* Sends several test prompts to `http://localhost:8000/agent/query`
* Prints:

  * HTTP status code
  * Parsed JSON response

Conceptual structure:

```python
import requests

BASE_URL = "http://localhost:8000"

def ask(query: str):
    resp = requests.post(f"{BASE_URL}/agent/query", json={"query": query}, timeout=300)
    print("STATUS:", resp.status_code)
    print("RESPONSE:")
    print(resp.json())

if __name__ == "__main__":
    ask("List my top runs by val_accuracy.")
    ask("Compare two best runs and tell me which is better.")
    ask("Flag the best run for publishing.")
    ask("Summarize my experiments.")
```

Run:

```bash
uvicorn app:app --reload --port 8000
python scripts/test_agent.py
```

You should see `STATUS: 200` and a valid `AgentAnswer` JSON for each query.

---

# 10. Streamlit App (Optional UI) — `streaml.py`

If you include a Streamlit UI, it typically has 3–4 main tabs:

1. **Dashboard**

   * High-level counts (total runs, experiments)
   * Quick leaderboard by a chosen metric
   * Small run history plots

2. **Runs / Explorer**

   * Table of runs retrieved via `GET /runs`
   * Filters by model, experiment, dataset, note
   * Clicking a row fetches `GET /runs/{run_id}` and shows:

     * Params, final metrics
     * Notes
     * Links to artifacts
     * (Optional) charts from `metrics_*.csv`

3. **Compare**

   * Select 2–3 run IDs
   * Calls `GET /runs/compare?ids=...`
   * Shows:

     * Side-by-side metrics table
     * Highlighted best per metric
     * Plots of training curves (by reading CSVs)

4. **Assistant**

   * Chat box that calls `POST /agent/query`
   * Displays `natural_language_answer`
   * Uses `used_run_ids` to drive UI components (e.g., highlight referenced runs)

### 10.1 Running Streamlit

From your (SSH) terminal:

```bash
streamlit run streaml.py --server.port 8501 --server.address 0.0.0.0
```

Then open in your browser:

* Local machine (if running locally):
  `http://localhost:8501`
* Remote server (e.g., college server):

  * Use the IP/hostname, e.g. `http://<server-ip>:8501`
  * Or set up SSH port forwarding:
    `ssh -L 8501:localhost:8501 user@server`

---

# 11. UI / Frontend Integration (General)

You can plug in **any** frontend: Streamlit, React, Vue, Jupyter, etc.

There are two main integration styles:

## 11.1 Using the Agent Endpoint (`POST /agent/query`)

Best for chat-style assistants.

**Flow:**

1. User types:

   > “Show me my top 3 runs by accuracy and compare them.”

2. UI sends:

   ```http
   POST /agent/query
   {
     "query": "Show me my top 3 runs by accuracy and compare them."
   }
   ```

3. Agent returns:

   ```json
   {
     "intent": "compare_best_runs",
     "natural_language_answer": "...",
     "used_run_ids": ["...", "...", "..."],
     "comparison": { ... }
   }
   ```

4. UI:

   * Displays `natural_language_answer`
   * Uses `used_run_ids` to fetch details via `/runs/{id}` or `/runs/compare?...`
   * Builds tables / charts if desired

Example (Python):

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

---

## 11.2 Using Backend Endpoints Directly

Best for fully custom dashboards.

Example (Python / Streamlit):

```python
import pandas as pd
import requests

BASE_URL = "http://localhost:8000"

# 1) Fetch leaderboard
runs = requests.get(
    f"{BASE_URL}/runs/leaderboard?metric_name=accuracy&top_k=5"
).json()

# 2) Pick one run
run_id = runs[0]["run_id"]
detail = requests.get(f"{BASE_URL}/runs/{run_id}").json()

# 3) Choose metrics CSV artifact
metrics_csv = [
    p for p in detail["artifacts"]
    if "metrics_" in p and p.endswith(".csv")
][0]

df = pd.read_csv(metrics_csv)
# Now you can plot df["epoch"], df["train_loss"], df["val_loss"], etc.
```

Useful endpoints:

| Endpoint                    | Use Case                     |
| --------------------------- | ---------------------------- |
| `GET /runs`                 | Experiment table / filters   |
| `GET /runs/{id}`            | Run detail page              |
| `GET /runs/leaderboard`     | Leaderboard view             |
| `GET /runs/compare?ids=...` | Comparison charts / tables   |
| `PATCH /runs/{id}/note`     | Mark best / publish / reject |

---

# 12. Running Everything (Quick Summary)

1. **Install dependencies**

   ```bash
   conda create -n ai-lab-notebook python=3.10
   conda activate ai-lab-notebook
   pip install -r requirements.txt
   ```

2. **(Optional) Start Ollama or other LLM backend**

   ```bash
   python -m servers.ollama_server
   ```

3. **Run experiments**

   ```bash
   python examples/train_model/sweep.py
   ```

4. **Start backend**

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   # or: uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
   ```

5. **(Optional) Start Streamlit UI**

   ```bash
   streamlit run streaml.py
   ```

6. **Test agent**

   ```bash
   python scripts/test_agent.py
   ```

---

# 13. Capabilities (What This System Supports)

* ✅ Multi-model training (logistic regression, SVM, perceptron, etc.)
* ✅ Multi-epoch metrics logging (to DB + CSV artifacts)
* ✅ Checkpoints (joblib)
* ✅ Rich run parameters tracked in SQLite
* ✅ Filtering / leaderboard / comparison
* ✅ Tagging / notes (`PUBLISH`, `BAD`, etc.)
* ✅ FastAPI backend with clean JSON APIs
* ✅ LangChain tools and structured `AgentAnswer` responses
* ✅ Easy integration with Streamlit or any other UI
* ✅ W&B integration is optional, not required

This README should be enough for:

* Someone running experiments
* Someone building a dashboard
* Someone building an agent or chat-assistant on top of your lab.

```
```
