import functools
import sqlite3
import os
import sys
import uuid
import datetime
import subprocess
import json

from git import Repo
import wandb

DB = os.getenv("DB_PATH", "manifests/manifests.db")
os.makedirs(os.path.dirname(DB), exist_ok=True)


def _git_info():
    if Repo is None:
        return None, None
    try:
        repo = Repo(".")
        commit = repo.head.commit.hexsha
        diff = repo.git.diff("--staged") or repo.git.diff() or ""
        return commit, diff
    except Exception:
        return None, None


def _pip_freeze():
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    except Exception:
        return None


def write_manifest(m):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS manifests (
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
    )"""
    )
    conn.execute(
        """INSERT OR REPLACE INTO manifests
      (run_id,timestamp,git_hash,git_diff,command,python_version,pip_freeze,device_info,dataset_id,params_json,final_metrics_json,artifacts_json,wandb_url,note)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            m.get("run_id"),
            m.get("timestamp"),
            m.get("git_hash"),
            m.get("git_diff"),
            m.get("command"),
            m.get("python_version"),
            m.get("pip_freeze"),
            m.get("device_info"),
            m.get("dataset_id"),
            json.dumps(m.get("params", {})),
            json.dumps(m.get("final_metrics", {})),
            json.dumps(m.get("artifacts", [])),
            m.get("wandb_url"),
            m.get("note"),
        ),
    )
    conn.commit()
    conn.close()


def log_run(fn=None, *, project=None, tags=None):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            run_uuid = str(uuid.uuid4())
            git_hash, git_diff = _git_info()

            # Base command: how the script was invoked
            base_cmd = " ".join(sys.argv) if sys.argv else ""

            # ---- NEW: build a richer "command" string that includes key params ----
            # This makes it easier for the agent to filter by model / dataset via command_substring.
            param_keys_for_command = [
                "experiment_name",
                "model_name",
                "task_name",
                "dataset_name",
                "n_epochs",
                "learning_rate",
                "test_size",
                "random_state",
            ]
            param_parts = []
            for key in param_keys_for_command:
                if key in kwargs:
                    value = kwargs[key]
                    param_parts.append(f"{key}={value}")

            if param_parts:
                cmd = base_cmd + " " + " ".join(param_parts) if base_cmd else " ".join(
                    param_parts
                )
            else:
                cmd = base_cmd
            # ----------------------------------------------------------------------

            project_name = project or os.getenv("WANDB_PROJECT")
            wandb_run = None
            if wandb is not None:
                try:
                    wandb_run = wandb.init(project=project_name, config=kwargs, reinit=True)
                    if tags:
                        try:
                            wandb_run.tags = list(tags)
                        except Exception:
                            pass
                except Exception:
                    wandb_run = None

            result = None
            try:
                result = f(*args, **kwargs)
            finally:
                final_metrics = {}
                artifacts = []
                wandb_url = None

                if wandb_run:
                    try:
                        wandb_url = getattr(wandb_run, "get_url", lambda: None)()
                    except Exception:
                        wandb_url = None

                if isinstance(result, dict):
                    # prefer explicit returned metrics over wandb summary when present
                    if "final_metrics" in result and isinstance(result["final_metrics"], dict):
                        final_metrics = result["final_metrics"]
                    if "artifacts" in result and isinstance(result["artifacts"], (list, tuple)):
                        artifacts = list(result["artifacts"])

                # If wandb is active and we have returned final metrics, log them
                if wandb_run:
                    try:
                        if final_metrics:
                            # log metrics as a final step
                            try:
                                wandb_run.log(final_metrics)
                            except Exception:
                                try:
                                    wandb.log(final_metrics)
                                except Exception:
                                    pass
                        # upload artifacts (if any). Use Artifact API when possible, fallback to wandb.save
                        for path in artifacts:
                            try:
                                if not path:
                                    continue
                                if os.path.isdir(path):
                                    art = wandb.Artifact(
                                        f"artifact-{os.path.basename(path)}", type="dataset"
                                    )
                                    art.add_dir(path)
                                else:
                                    art = wandb.Artifact(
                                        f"artifact-{os.path.basename(path)}", type="model"
                                    )
                                    art.add_file(path)
                                try:
                                    wandb_run.log_artifact(art)
                                except Exception:
                                    try:
                                        wandb.log_artifact(art)
                                    except Exception:
                                        try:
                                            wandb.save(path)
                                        except Exception:
                                            pass
                            except Exception:
                                try:
                                    wandb.save(path)
                                except Exception:
                                    pass

                    except Exception:
                        pass
                    finally:
                        try:
                            if final_metrics:
                                for k, v in final_metrics.items():
                                    try:
                                        wandb_run.summary[k] = v
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            wandb.finish()
                        except Exception:
                            pass
                else:
                    pass

                # ---- NEW: derive dataset_id from kwargs, with env fallback ----
                # Prefer explicit dataset_name, then task_name, then DATASET_COMMIT env.
                dataset_name = kwargs.get("dataset_name")
                task_name = kwargs.get("task_name")
                dataset_id = (
                    dataset_name
                    or task_name
                    or os.getenv("DATASET_COMMIT")  # old behavior fallback
                )
                # ----------------------------------------------------------------

                manifest = {
                    "run_id": run_uuid,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "git_hash": git_hash,
                    "git_diff": git_diff,
                    "command": cmd,
                    "python_version": sys.version,
                    "pip_freeze": _pip_freeze(),
                    "device_info": os.uname().sysname if hasattr(os, "uname") else None,
                    "dataset_id": dataset_id,
                    "params": kwargs,
                    "final_metrics": final_metrics,
                    "artifacts": artifacts,
                    "wandb_url": wandb_url,
                    "note": "",
                }
                write_manifest(manifest)
            return result

        return wrapper

    return decorator if fn is None else decorator(fn)


# import functools, sqlite3, os, sys, uuid, datetime, subprocess, json
# from git import Repo
# import wandb

# DB = os.getenv("DB_PATH", "manifests/manifests.db")
# os.makedirs(os.path.dirname(DB), exist_ok=True)

# def _git_info():
#     if Repo is None:
#         return None, None
#     try:
#         repo = Repo('.')
#         commit = repo.head.commit.hexsha
#         diff = repo.git.diff('--staged') or repo.git.diff() or ""
#         return commit, diff
#     except Exception:
#         return None, None

# def _pip_freeze():
#     try:
#         return subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
#     except Exception:
#         return None

# def write_manifest(m):
#     conn = sqlite3.connect(DB, timeout=10)
#     conn.execute("""CREATE TABLE IF NOT EXISTS manifests (
#       run_id TEXT PRIMARY KEY,
#       timestamp TEXT,
#       git_hash TEXT,
#       git_diff TEXT,
#       command TEXT,
#       python_version TEXT,
#       pip_freeze TEXT,
#       device_info TEXT,
#       dataset_id TEXT,
#       params_json TEXT,
#       final_metrics_json TEXT,
#       artifacts_json TEXT,
#       wandb_url TEXT,
#       note TEXT
#     )""")
#     conn.execute("""INSERT OR REPLACE INTO manifests
#       (run_id,timestamp,git_hash,git_diff,command,python_version,pip_freeze,device_info,dataset_id,params_json,final_metrics_json,artifacts_json,wandb_url,note)
#       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
#       (m.get("run_id"),m.get("timestamp"),m.get("git_hash"),m.get("git_diff"),m.get("command"),
#        m.get("python_version"),m.get("pip_freeze"),m.get("device_info"),m.get("dataset_id"),
#        json.dumps(m.get("params",{})), json.dumps(m.get("final_metrics",{})),
#        json.dumps(m.get("artifacts",[])), m.get("wandb_url"), m.get("note")))
#     conn.commit(); conn.close()

# def log_run(fn=None, *, project=None, tags=None):
#     def decorator(f):
#         @functools.wraps(f)
#         def wrapper(*args, **kwargs):
#             run_uuid = str(uuid.uuid4())
#             git_hash, git_diff = _git_info()
#             cmd = " ".join(sys.argv)
#             project_name = project or os.getenv("WANDB_PROJECT")
#             wandb_run = None
#             if wandb is not None:
#                 try:
#                     wandb_run = wandb.init(project=project_name, config=kwargs, reinit=True)
#                 except Exception:
#                     wandb_run = None

#             result = None
#             try:
#                 result = f(*args, **kwargs)
#             finally:
#                 final_metrics = {}
#                 artifacts = []
#                 wandb_url = None

#                 if wandb_run:
#                     try:
#                         wandb_url = getattr(wandb_run, "get_url", lambda: None)()
#                     except Exception:
#                         wandb_url = None

#                 if isinstance(result, dict):
#                     # prefer explicit returned metrics over wandb summary when present
#                     if "final_metrics" in result and isinstance(result["final_metrics"], dict):
#                         final_metrics = result["final_metrics"]
#                     if "artifacts" in result and isinstance(result["artifacts"], (list, tuple)):
#                         artifacts = list(result["artifacts"])

#                 # If wandb is active and we have returned final metrics, log them
#                 if wandb_run:
#                     try:
#                         if final_metrics:
#                             # log metrics as a final step
#                             try:
#                                 wandb_run.log(final_metrics)
#                             except Exception:
#                                 try:
#                                     wandb.log(final_metrics)
#                                 except Exception:
#                                     pass
#                         # upload artifacts (if any). Use Artifact API when possible, fallback to wandb.save
#                         for path in artifacts:
#                             try:
#                                 if not path:
#                                     continue
#                                 if os.path.isdir(path):
#                                     art = wandb.Artifact(f"artifact-{os.path.basename(path)}", type="dataset")
#                                     art.add_dir(path)
#                                 else:
#                                     art = wandb.Artifact(f"artifact-{os.path.basename(path)}", type="model")
#                                     art.add_file(path)
#                                 try:
#                                     wandb_run.log_artifact(art)
#                                 except Exception:
#                                     try:
#                                         wandb.log_artifact(art)
#                                     except Exception:
#                                         try:
#                                             wandb.save(path)
#                                         except Exception:
#                                             pass
#                             except Exception:
#                                 try:
#                                     wandb.save(path)
#                                 except Exception:
#                                     pass

#                     except Exception:
#                         pass
#                     finally:
#                         try:
#                             if final_metrics:
#                                 for k, v in final_metrics.items():
#                                     try:
#                                         wandb_run.summary[k] = v
#                                     except Exception:
#                                         pass
#                         except Exception:
#                             pass
#                         try:
#                             wandb.finish()
#                         except Exception:
#                             pass
#                 else:
#                     pass

#                 manifest = {
#                     "run_id": run_uuid,
#                     "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
#                     "git_hash": git_hash,
#                     "git_diff": git_diff,
#                     "command": cmd,
#                     "python_version": sys.version,
#                     "pip_freeze": _pip_freeze(),
#                     "device_info": os.uname().sysname if hasattr(os, "uname") else None,
#                     "dataset_id": os.getenv("DATASET_COMMIT"),
#                     "params": kwargs,
#                     "final_metrics": final_metrics,
#                     "artifacts": artifacts,
#                     "wandb_url": wandb_url,
#                     "note": ""
#                 }
#                 write_manifest(manifest)
#             return result

#         return wrapper
#     return decorator if fn is None else decorator(fn)
