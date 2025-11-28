import sqlite3, os
DB = os.getenv("DB_PATH", "manifests/manifests.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)
conn = sqlite3.connect(DB)
conn.execute("""
CREATE TABLE IF NOT EXISTS manifests (
  run_id TEXT PRIMARY KEY,
  timestamp TEXT,
  git_hash TEXT,
  git_diff TEXT,
  command TEXT,
  python_version TEXT,
  pip_freeze TEXT,
  device_info TEXT,
  dataset_id TEXT,
  params_json TEXT,
  final_metrics_json TEXT,
  artifacts_json TEXT,
  wandb_url TEXT,
  note TEXT
);""")
conn.commit()
conn.close()
print("DB created at", DB)
