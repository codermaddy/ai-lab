# scripts/check_latest_runs.py

import os
import json
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "manifests/manifests.db")

def main(limit: int = 5):
    print(f"Using DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT run_id, timestamp, params_json, final_metrics_json, artifacts_json
        FROM manifests
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    for r in rows:
        run_id = r["run_id"]
        ts = r["timestamp"]
        params = json.loads(r["params_json"] or "{}")
        final_metrics = json.loads(r["final_metrics_json"] or "{}")
        artifacts = json.loads(r["artifacts_json"] or "[]")

        print("=" * 80)
        print(f"RUN: {run_id}")
        print(f"  timestamp: {ts}")
        print(f"  experiment_name: {params.get('experiment_name')}")
        print(f"  model_name:      {params.get('model_name')}")
        print(f"  task_name:       {params.get('task_name')}")
        print(f"  n_epochs:        {params.get('n_epochs')}")
        print(f"  learning_rate:   {params.get('learning_rate')}")

        print("  final_metrics:")
        for k, v in final_metrics.items():
            print(f"    {k}: {v}")

        print("  artifacts:")
        for path in artifacts:
            exists = Path(path).exists()
            print(f"    {path}  [exists={exists}]")

    print("=" * 80)


if __name__ == "__main__":
    main(limit=10)
