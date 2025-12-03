import os
import json
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Lab Notebook",
    layout="wide",
)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# -------------------------------------------------------------------
# Simple HTTP helpers
# -------------------------------------------------------------------

def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    url = f"{BACKEND_URL.rstrip('/')}{path}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"GET {url} failed: {e}")
        return None


def patch_json(path: str, body: Dict[str, Any]) -> Optional[Any]:
    url = f"{BACKEND_URL.rstrip('/')}{path}"
    try:
        resp = requests.patch(url, json=body, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"PATCH {url} failed: {e}")
        return None


def post_json(path: str, body: Dict[str, Any]) -> Optional[Any]:
    url = f"{BACKEND_URL.rstrip('/')}{path}"
    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"POST {url} failed: {e}")
        return None


# -------------------------------------------------------------------
# Small backend wrappers
# -------------------------------------------------------------------

def list_runs(
    limit: int = 50,
    dataset_id: Optional[str] = None,
    note_substring: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit, "offset": 0}
    if dataset_id:
        params["dataset_id"] = dataset_id
    if note_substring:
        params["note_substring"] = note_substring

    data = get_json("/runs", params=params)
    return data or []


def leaderboard(
    metric_name: str = "accuracy",
    top_k: int = 20,
    dataset_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"metric_name": metric_name, "top_k": top_k}
    if dataset_id:
        params["dataset_id"] = dataset_id
    data = get_json("/runs/leaderboard", params=params)
    return data or []


def get_run_detail(run_id: str) -> Optional[Dict[str, Any]]:
    return get_json(f"/runs/{run_id}")


def compare_runs(run_ids: List[str]) -> Optional[Dict[str, Any]]:
    # GET /runs/compare?ids=id1&ids=id2
    params = [("ids", rid) for rid in run_ids]
    url = f"{BACKEND_URL.rstrip('/')}/runs/compare"
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"GET {url} failed: {e}")
        return None


def update_run_note(run_id: str, note: str) -> Optional[Dict[str, Any]]:
    return patch_json(f"/runs/{run_id}/note", {"note": note})


def call_agent(query: str) -> Optional[Dict[str, Any]]:
    return post_json("/agent/query", {"query": query})


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def build_run_tag(run: Dict[str, Any]) -> str:
    """
    Build a human-readable tag: task_name | experiment_name | model_name.
    1) Prefer params (task_name, experiment_name, model_name) if present.
    2) Fall back to parsing the command string.
    3) If nothing found, return empty string (caller can display '(no tag)').
    """
    # ---- 1) Try params ----
    params = run.get("params") or {}
    task = params.get("task_name") or params.get("task")
    experiment = params.get("experiment_name") or params.get("experiment")
    model = params.get("model_name") or params.get("model")

    parts = [p for p in (task, experiment, model) if p]
    if parts:
        return " | ".join(parts)

    # ---- 2) Fall back to command parsing ----
    command = run.get("command") or ""

    def get_flag_any(names):
        for name in names:
            for prefix in (f"--{name}=", f"{name}="):
                if prefix in command:
                    return command.split(prefix, 1)[1].split()[0]
        return None

    task = task or get_flag_any(["task_name", "task"])
    experiment = experiment or get_flag_any(["experiment_name", "experiment"])
    model = model or get_flag_any(["model_name", "model"])

    parts = [p for p in (task, experiment, model) if p]
    return " | ".join(parts)


def extract_metric(run: Dict[str, Any], name: str) -> Optional[float]:
    fm = run.get("final_metrics") or {}
    val = fm.get(name)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


# def runs_to_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
#     rows = []
#     for r in runs:
#         rows.append(
#             {
#                 "run_id": r.get("run_id"),
#                 "timestamp": r.get("timestamp"),
#                 "dataset_id": r.get("dataset_id"),
#                 "command": r.get("command"),
#                 "accuracy": extract_metric(r, "accuracy"),
#                 "f1_macro": extract_metric(r, "f1_macro"),
#                 "final_train_loss": extract_metric(r, "final_train_loss"),
#                 "final_val_loss": extract_metric(r, "final_val_loss"),
#                 "runtime_sec": extract_metric(r, "runtime_sec"),
#                 "note": r.get("note"),
#                 "wandb_url": r.get("wandb_url"),
#             }
#         )
#     df = pd.DataFrame(rows)
#     return df

def runs_to_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        cmd = r.get("command") or ""
        rows.append(
            {
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "dataset_id": r.get("dataset_id"),
                "command": cmd,
                "tag": build_run_tag(r),
                "accuracy": extract_metric(r, "accuracy"),
                "f1_macro": extract_metric(r, "f1_macro"),
                "final_train_loss": extract_metric(r, "final_train_loss"),
                "final_val_loss": extract_metric(r, "final_val_loss"),
                "runtime_sec": extract_metric(r, "runtime_sec"),
                "note": r.get("note"),
                "wandb_url": r.get("wandb_url"),
            }
        )
    df = pd.DataFrame(rows)
    return df

def find_metrics_csv(artifacts: List[str]) -> Optional[Path]:
    for a in artifacts:
        if not a:
            continue
        p = Path(a)
        if p.name.endswith(".csv") and "metrics" in p.name and p.exists():
            return p
    return None


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------

st.title("🧪 AI Lab Notebook")

tab_dashboard, tab_compare, tab_assistant = st.tabs(
    ["📊 Runs Dashboard", "📈 Run Comparison", "🤖 Assistant"]
)


# -------------------------------------------------------------------
# TAB 1: Runs Dashboard
# -------------------------------------------------------------------

with tab_dashboard:
    st.subheader("Runs overview")

    col_filters, col_table = st.columns([1, 3])

    with col_filters:
        st.markdown("**Filters**")

        # Fetch a small sample to derive dataset_ids
        sample_runs = list_runs(limit=200)
        all_datasets = sorted(
            {r.get("dataset_id") for r in sample_runs if r.get("dataset_id")}
        )
        dataset_id = st.selectbox(
            "Dataset",
            options=["(all)"] + all_datasets,
            index=0,
        )
        dataset_filter = None if dataset_id == "(all)" else dataset_id

        note_substring = st.text_input(
            "Note contains",
            value="",
            placeholder="PUBLISH, BEST, IGNORE, ...",
        )

        limit = st.slider("Max runs to show", min_value=10, max_value=200, value=50, step=10)
        metric_for_sort = st.selectbox(
            "Sort by metric",
            options=["accuracy", "f1_macro", "runtime_sec", "final_val_loss"],
            index=0,
        )

        if st.button("Refresh runs"):
            st.experimental_rerun()

    # with col_table:
    #     runs = list_runs(limit=limit, dataset_id=dataset_filter, note_substring=note_substring or None)
    #     if not runs:
    #         st.warning("No runs found with current filters.")
    #     else:
    #         df = runs_to_dataframe(runs)
    #         if metric_for_sort in df.columns:
    #             df = df.sort_values(by=metric_for_sort, ascending=(metric_for_sort == "final_val_loss"))

    #         st.markdown("**Runs**")
    #         st.dataframe(
    #             df[
    #                 [
    #                     "run_id",
    #                     "timestamp",
    #                     "dataset_id",
    #                     "accuracy",
    #                     "f1_macro",
    #                     "final_val_loss",
    #                     "runtime_sec",
    #                     "note",
    #                     "wandb_url",
    #                 ]
    #             ],
    #             use_container_width=True,
    #         )

    # st.markdown("---")
    # st.subheader("Tag / update a run note")

    # col_note1, col_note2 = st.columns([2, 1])

    # with col_note1:
    #     all_run_ids = [r.get("run_id") for r in sample_runs]
    #     selected_run_for_note = st.selectbox(
    #         "Select run_id",
    #         options=all_run_ids,
    #         format_func=lambda rid: rid or "",
    #     )

    #     new_note = st.text_input("New note (e.g. PUBLISH)", value="PUBLISH")

    # with col_note2:
    #     if st.button("Update note", type="primary"):
    #         if not selected_run_for_note:
    #             st.error("No run selected.")
    #         else:
    #             updated = update_run_note(selected_run_for_note, new_note)
    #             if updated:
    #                 st.success(f"Updated note for run {selected_run_for_note} to '{new_note}'")
    #                 st.experimental_rerun()
    with col_table:
        runs = list_runs(
            limit=limit,
            dataset_id=dataset_filter,
            note_substring=note_substring or None,
        )
        if not runs:
            st.warning("No runs found with current filters.")
        else:
            df = runs_to_dataframe(runs)
            if metric_for_sort in df.columns:
                df = df.sort_values(
                    by=metric_for_sort,
                    ascending=(metric_for_sort == "final_val_loss"),
                )

            df = df.reset_index(drop=True)

            st.markdown("**Runs (edit notes and save)**")

            display_cols = [
                "run_id",
                "tag",
                "timestamp",
                "dataset_id",
                "accuracy",
                "f1_macro",
                "final_val_loss",
                "runtime_sec",
                "note",
                "wandb_url",
            ]

            edited_df = st.data_editor(
                df[display_cols],
                num_rows="fixed",
                use_container_width=True,
                key="runs_editor",
                column_config={
                    "note": st.column_config.TextColumn(
                        "note",
                        help="Edit and click 'Update notes' to save to DB",
                    )
                },
                disabled=[
                    "run_id",
                    "tag",
                    "timestamp",
                    "dataset_id",
                    "accuracy",
                    "f1_macro",
                    "final_val_loss",
                    "runtime_sec",
                    "wandb_url",
                ],
            )

            if st.button("Update notes", type="primary"):
                # Find rows where note changed
                changes = []
                for idx in range(len(df)):
                    old_note = df.loc[idx, "note"]
                    new_note = edited_df.loc[idx, "note"]
                    if old_note != new_note:
                        changes.append((df.loc[idx, "run_id"], new_note))

                if not changes:
                    st.info("No note changes to update.")
                else:
                    success_count = 0
                    for rid, new_note in changes:
                        updated = update_run_note(rid, new_note or "")
                        if updated:
                            success_count += 1

                    st.success(f"Updated notes for {success_count} runs.")
                    st.experimental_rerun()



# -------------------------------------------------------------------
# TAB 2: Run Comparison
# -------------------------------------------------------------------

with tab_compare:
    st.subheader("Compare runs and visualize training curves")

    # Use leaderboard as a convenient way to get top runs
    metric_for_lb = st.selectbox(
        "Leaderboard metric",
        options=["accuracy", "f1_macro"],
        index=0,
    )
    dataset_for_lb = st.text_input(
        "Optional dataset_id filter (e.g. iris_classification)",
        value="",
    )
    

    # lb_runs = leaderboard(
    #     metric_name=metric_for_lb,
    #     top_k=top_k,
    #     dataset_id=dataset_for_lb or None,
    # )
    
    raw_runs = list_runs(limit=1000)

    # Fetch a large pool
    raw_runs = list_runs(limit=1000)

    # Filter by optional dataset_id (substring)
    ds_filter = (dataset_for_lb or "").strip().lower()
    if ds_filter:
        raw_runs = [
            r for r in raw_runs
            if ds_filter in (r.get("dataset_id") or "").lower()
        ]

    # Sort all matching runs by selected metric
    def metric_value(run: Dict[str, Any]) -> float:
        val = extract_metric(run, metric_for_lb)
        if val is None:
            return -1e9
        return float(val)

    raw_runs.sort(key=metric_value, reverse=True)

    lb_runs = raw_runs


    if not lb_runs:
        st.warning("No runs available for comparison. Try changing filters.")
    else:
        # df_lb = runs_to_dataframe(lb_runs)
        # # Build nice labels for selection
        # options = []
        # label_map = {}
        # for r in lb_runs:
        #     rid = r.get("run_id")
        #     ds = r.get("dataset_id") or "unknown_dataset"
        #     acc = extract_metric(r, "accuracy")
        #     model_hint = (r.get("command") or "").split("model_name=")[-1].split()[0] if "model_name=" in (r.get("command") or "") else "model?"
        #     label = f"{rid[:8]} | {ds} | {model_hint} | acc={acc:.3f}" if acc is not None else f"{rid[:8]} | {ds} | {model_hint}"
        #     options.append(rid)
        #     label_map[rid] = label
        
        df_lb = runs_to_dataframe(lb_runs)
        # Build nice labels for selection: short_id + tag (+ metric)
        options = []
        label_map = {}
        for r in lb_runs:
            rid = r.get("run_id")
            tag = build_run_tag(r)
            if not tag:
                tag = "(no tag)"
            acc = extract_metric(r, "accuracy")

            base = f"{rid[:8]} | {tag}"
            if acc is not None:
                label = f"{base} | acc={acc:.3f}"
            else:
                label = base

            options.append(rid)
            label_map[rid] = label


        selected_run_ids = st.multiselect(
            "Select runs to compare",
            options=options,
            default=options[:2],
            format_func=lambda rid: label_map.get(rid, rid),
        )

        if len(selected_run_ids) < 1:
            st.info("Select at least one run to see details.")
        else:
            # Fetch details
            details = []
            for rid in selected_run_ids:
                d = get_run_detail(rid)
                if d:
                    details.append(d)

            if not details:
                st.error("Failed to fetch run details.")
            else:
                st.markdown("### Metrics table")

                rows = []
                for d in details:
                    fm = d.get("final_metrics") or {}
                    params = d.get("params") or {}
                    rows.append(
                        {
                            "run_id": d.get("run_id"),
                            "dataset_id": d.get("dataset_id"),
                            "experiment_name": params.get("experiment_name"),
                            "model_name": params.get("model_name"),
                            "task_name": params.get("task_name"),
                            "accuracy": fm.get("accuracy"),
                            "f1_macro": fm.get("f1_macro"),
                            "final_train_loss": fm.get("final_train_loss"),
                            "final_val_loss": fm.get("final_val_loss"),
                            "runtime_sec": fm.get("runtime_sec"),
                            "note": d.get("note"),
                        }
                    )

                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df, use_container_width=True)

                # st.markdown("### Training curves (from metrics CSV artifacts)")

                # # Collect loss curves
                # loss_curves = {}
                # for d in details:
                #     rid = d.get("run_id")
                #     artifacts = d.get("artifacts") or []
                #     csv_path = find_metrics_csv(artifacts)
                #     if csv_path is None:
                #         st.warning(f"No metrics CSV artifact found for run {rid}")
                #         continue
                #     try:
                #         df_metrics = pd.read_csv(csv_path)
                #         if {"epoch", "train_loss", "val_loss"}.issubset(df_metrics.columns):
                #             loss_curves[rid] = df_metrics
                #         else:
                #             st.warning(f"Metrics CSV for run {rid} does not contain expected columns.")
                #     except Exception as e:
                #         st.error(f"Failed to read metrics CSV for run {rid}: {e}")

                # if loss_curves:
                #     fig, ax = plt.subplots()
                #     for rid, dfm in loss_curves.items():
                #         short_id = rid[:8]
                #         ax.plot(dfm["epoch"], dfm["train_loss"], label=f"{short_id} train")
                #         ax.plot(dfm["epoch"], dfm["val_loss"], linestyle="--", label=f"{short_id} val")
                #     ax.set_xlabel("Epoch")
                #     ax.set_ylabel("Loss")
                #     ax.set_title("Train / Val loss vs Epoch")
                #     ax.legend()
                #     st.pyplot(fig)
                # else:
                #     st.info("No usable metrics CSVs found for the selected runs.")
                
                st.markdown("### Training curves & metric comparison")

                # Collect loss curves
                loss_curves: Dict[str, pd.DataFrame] = {}
                for d in details:
                    rid = d.get("run_id")
                    artifacts = d.get("artifacts") or []
                    csv_path = find_metrics_csv(artifacts)
                    if csv_path is None:
                        st.warning(f"No metrics CSV artifact found for run {rid}")
                        continue
                    try:
                        df_metrics = pd.read_csv(csv_path)
                        if {"epoch", "train_loss", "val_loss"}.issubset(df_metrics.columns):
                            loss_curves[rid] = df_metrics
                        else:
                            st.warning(
                                f"Metrics CSV for run {rid} does not contain expected columns "
                                f"(need epoch, train_loss, val_loss)."
                            )
                    except Exception as e:
                        st.error(f"Failed to read metrics CSV for run {rid}: {e}")

                # Create two side-by-side plots
                if loss_curves or (metric_for_lb in metrics_df.columns):
                    col_loss, col_bar = st.columns(2)

                    # Left: train vs val loss curves (smaller figsize)
                    with col_loss:
                        if loss_curves:
                            fig_loss, ax_loss = plt.subplots(figsize=(4, 3))
                            for rid, dfm in loss_curves.items():
                                short_id = rid[:8]
                                ax_loss.plot(
                                    dfm["epoch"],
                                    dfm["train_loss"],
                                    label=f"{short_id} train",
                                )
                                ax_loss.plot(
                                    dfm["epoch"],
                                    dfm["val_loss"],
                                    linestyle="--",
                                    label=f"{short_id} val",
                                )
                            ax_loss.set_xlabel("Epoch")
                            ax_loss.set_ylabel("Loss")
                            ax_loss.set_title("Train / Val loss vs Epoch")
                            ax_loss.legend(fontsize=4)
                            fig_loss.tight_layout()
                            st.pyplot(fig_loss)
                        else:
                            st.info("No usable metrics CSVs found for the selected runs.")

                    # Right: bar chart of selected metric across runs
                    with col_bar:
                        if metric_for_lb in metrics_df.columns:
                            metric_df = metrics_df.dropna(subset=[metric_for_lb])
                            if not metric_df.empty:
                                fig_bar, ax_bar = plt.subplots(figsize=(4, 3))

                                # Build labels as "task_name | model_name"
                                x_labels = []
                                for _, row in metric_df.iterrows():
                                    task = row.get("task_name") or "task?"
                                    model = row.get("model_name") or "model?"
                                    x_labels.append(f"{task} | {model}")

                                values = metric_df[metric_for_lb].astype(float)

                                # ---- NEW: Assign a different color per bar ----
                                num_bars = len(values)
                                colors = plt.cm.tab20(range(num_bars))  # tab20 gives 20 distinct colors

                                ax_bar.bar(x_labels, values, color=colors)

                                ax_bar.set_xlabel("Task + model")
                                ax_bar.set_ylabel(metric_for_lb)
                                ax_bar.set_title(f"{metric_for_lb} across selected runs")
                                ax_bar.tick_params(axis="x", rotation=20, labelsize=4)
                                fig_bar.tight_layout()
                                st.pyplot(fig_bar)
                            else:
                                st.info(f"No non-NaN values for metric '{metric_for_lb}' to plot.")
                        else:
                            st.info(f"Metric '{metric_for_lb}' not found in metrics table.")




# -------------------------------------------------------------------
# TAB 3: Assistant
# -------------------------------------------------------------------

# with tab_assistant:
#     st.subheader("Experiment assistant")

#     # if "chat_messages" not in st.session_state:
#     #     st.session_state.chat_messages = []

#     # Show history
#     for msg in st.session_state.chat_messages:
#         if msg["role"] == "user":
#             with st.chat_message("user"):
#                 st.markdown(msg["content"])
#         else:
#             with st.chat_message("assistant"):
#                 st.markdown(msg["content"])

#     user_input = st.chat_input("Ask about your experiments, runs, models, datasets...")

#     if user_input:
#         # Add user message to history
#         # st.session_state.chat_messages.append({"role": "user", "content": user_input})
#         with st.chat_message("assistant"):
#             st.markdown(full_msg)

#         # Call backend agent
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 resp = call_agent(user_input)
#                 if not resp:
#                     st.error("Agent call failed.")
#                 else:
#                     # resp has: intent, natural_language_answer, used_run_ids, comparison, flagged_run_id
#                     answer = resp.get("natural_language_answer") or ""
#                     used_run_ids = resp.get("used_run_ids") or []
#                     flagged_run_id = resp.get("flagged_run_id")
#                     comparison = resp.get("comparison")

#                     # Build a nice markdown message
#                     lines = [answer]

#                     if used_run_ids:
#                         lines.append("")
#                         lines.append("**Runs referenced:**")
#                         for rid in used_run_ids:
#                             lines.append(f"- `{rid}`")

#                     if flagged_run_id:
#                         lines.append("")
#                         lines.append(f"**Flagged run:** `{flagged_run_id}`")

#                     if comparison:
#                         try:
#                             better = comparison.get("better_run_id")
#                             if better:
#                                 lines.append("")
#                                 lines.append(f"**Agent thinks best run:** `{better}`")
#                         except Exception:
#                             pass

#                     full_msg = "\n".join(lines)
#                     st.markdown(full_msg)
#                     st.session_state.chat_messages.append(
#                         {"role": "assistant", "content": full_msg}
#                     )

with tab_assistant:
    st.subheader("Experiment assistant (LLM + tools)")

    # Show history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about your experiments, runs, models, datasets...")

    if user_input:
        # 1) Show the current user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) Save it to history
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })

        # 3) Call backend and show assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_agent(user_input)

                if not resp:
                    answer = "Agent call failed."
                else:
                    answer = resp.get("natural_language_answer") or ""
                    used_run_ids = resp.get("used_run_ids") or []
                    flagged_run_id = resp.get("flagged_run_id")
                    comparison = resp.get("comparison")

                    lines = [answer]

                    if used_run_ids:
                        lines.append("\n**Runs referenced:**")
                        for rid in used_run_ids:
                            lines.append(f"- `{rid}`")

                    if flagged_run_id:
                        lines.append(f"\n**Flagged run:** `{flagged_run_id}`")

                    if comparison and comparison.get("better_run_id"):
                        lines.append(
                            f"\n**Agent thinks best run:** `{comparison['better_run_id']}`"
                        )

                    answer = "\n".join(lines)

                st.markdown(answer)

        # 4) Save assistant reply to history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer
        })


