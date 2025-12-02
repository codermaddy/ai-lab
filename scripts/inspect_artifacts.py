# scripts/inspect_metrics_artifacts.py

import os
import json
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = os.getenv("DB_PATH", "manifests/manifests.db")

def main(run_id: str):
    print(f"Using DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT artifacts_json FROM manifests WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()

    if row is None:
        print("No such run_id in DB")
        return

    artifacts = json.loads(row["artifacts_json"] or "[]")
    print(f"Artifacts for run {run_id}:")
    metrics_paths = []

    for path in artifacts:
        print(f"  {path}")
        if "metrics_" in path and path.endswith(".csv"):
            metrics_paths.append(path)

    if not metrics_paths:
        print("No metrics_*.csv artifact found.")
        return

    # Just take the first metrics CSV
    metrics_path = Path(metrics_paths[0])
    if not metrics_path.exists():
        print(f"Metrics file does not exist on disk: {metrics_path}")
        return

    print(f"\nReading metrics from {metrics_path}\n")
    df = pd.read_csv(metrics_path)
    print(df.head())
    print("\nSummary:")
    print(df.describe())

if __name__ == "__main__":
    # replace with a real run_id you saw from /runs
    example_run_id = "621e540f-33bb-4387-bb54-fac6e5120243"
    main(example_run_id)
