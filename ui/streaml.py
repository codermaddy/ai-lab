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

def extract_metric(run: Dict[str, Any], name: str) -> Optional[float]:
    fm = run.get("final_metrics") or {}
    val = fm.get(name)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def runs_to_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append(
            {
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "dataset_id": r.get("dataset_id"),
                "command": r.get("command"),
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
            "Dataset (dataset_id filter)",
            options=["(all)"] + all_datasets,
            index=0,
        )
        dataset_filter = None if dataset_id == "(all)" else dataset_id

        note_substring = st.text_input(
            "Note contains (e.g. PUBLISH)",
            value="",
            placeholder="PUBLISH, BEST, IGNORE, ...",
        )

        limit = st.slider("Max runs to show", min_value=10, max_value=200, value=50, step=10)
        metric_for_sort = st.selectbox(
            "Sort by metric (descending)",
            options=["accuracy", "f1_macro", "runtime_sec", "final_val_loss"],
            index=0,
        )

        if st.button("Refresh runs"):
            st.experimental_rerun()

    with col_table:
        runs = list_runs(limit=limit, dataset_id=dataset_filter, note_substring=note_substring or None)
        if not runs:
            st.warning("No runs found with current filters.")
        else:
            df = runs_to_dataframe(runs)
            if metric_for_sort in df.columns:
                df = df.sort_values(by=metric_for_sort, ascending=(metric_for_sort == "final_val_loss"))

            st.markdown("**Runs**")
            st.dataframe(
                df[
                    [
                        "run_id",
                        "timestamp",
                        "dataset_id",
                        "accuracy",
                        "f1_macro",
                        "final_val_loss",
                        "runtime_sec",
                        "note",
                        "wandb_url",
                    ]
                ],
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Tag / update a run note")

    col_note1, col_note2 = st.columns([2, 1])

    with col_note1:
        all_run_ids = [r.get("run_id") for r in sample_runs]
        selected_run_for_note = st.selectbox(
            "Select run_id",
            options=all_run_ids,
            format_func=lambda rid: rid or "",
        )

        new_note = st.text_input("New note (e.g. PUBLISH)", value="PUBLISH")

    with col_note2:
        if st.button("Update note", type="primary"):
            if not selected_run_for_note:
                st.error("No run selected.")
            else:
                updated = update_run_note(selected_run_for_note, new_note)
                if updated:
                    st.success(f"Updated note for run {selected_run_for_note} to '{new_note}'")
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
    top_k = st.slider("Number of runs to fetch for selection", 5, 50, 20, 5)

    lb_runs = leaderboard(
        metric_name=metric_for_lb,
        top_k=top_k,
        dataset_id=dataset_for_lb or None,
    )

    if not lb_runs:
        st.warning("No runs available for comparison. Try changing filters.")
    else:
        df_lb = runs_to_dataframe(lb_runs)
        # Build nice labels for selection
        options = []
        label_map = {}
        for r in lb_runs:
            rid = r.get("run_id")
            ds = r.get("dataset_id") or "unknown_dataset"
            acc = extract_metric(r, "accuracy")
            model_hint = (r.get("command") or "").split("model_name=")[-1].split()[0] if "model_name=" in (r.get("command") or "") else "model?"
            label = f"{rid[:8]} | {ds} | {model_hint} | acc={acc:.3f}" if acc is not None else f"{rid[:8]} | {ds} | {model_hint}"
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
                
                                # --- Metric comparison plot (bar chart) ---
                st.markdown("### Metric comparison")

                # Choose which metric to plot
                available_metrics = [
                    col
                    for col in ["accuracy", "f1_macro", "final_train_loss", "final_val_loss", "runtime_sec"]
                    if col in metrics_df.columns
                ]

                if not available_metrics:
                    st.info("No numeric metrics available to plot.")
                else:
                    # default metric = same as leaderboard metric if present, else first available
                    default_idx = (
                        available_metrics.index(metric_for_lb)
                        if metric_for_lb in available_metrics
                        else 0
                    )
                    metric_to_plot = st.selectbox(
                        "Metric to plot",
                        options=available_metrics,
                        index=default_idx,
                    )

                    plot_df = metrics_df.dropna(subset=[metric_to_plot])
                    if plot_df.empty:
                        st.info(f"No values found for metric '{metric_to_plot}' in selected runs.")
                    else:
                        # Build nice x-axis labels using task_name + model_name (fallback to run_id prefix)
                        labels = []
                        values = []
                        for _, row in plot_df.iterrows():
                            task = row.get("task_name") or ""
                            model = row.get("model_name") or ""
                            if task or model:
                                label = f"{task} / {model}".strip().strip("/")
                            else:
                                label = (row.get("run_id") or "")[:8] or "run"
                            labels.append(label)
                            values.append(row[metric_to_plot])

                        fig, ax = plt.subplots()
                        x_positions = range(len(labels))
                        ax.bar(x_positions, values)
                        ax.set_xticks(list(x_positions))
                        ax.set_xticklabels(labels, rotation=20, ha="right")
                        ax.set_ylabel(metric_to_plot)
                        ax.set_title(f"{metric_to_plot} across selected runs")
                        st.pyplot(fig)



# -------------------------------------------------------------------
# TAB 3: Assistant
# -------------------------------------------------------------------

with tab_assistant:
    st.subheader("Experiment assistant (LLM + tools)")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Show history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your experiments, runs, models, datasets...")

    if user_input:
        # Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Call backend agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = call_agent(user_input)
                if not resp:
                    st.error("Agent call failed.")
                else:
                    # resp has: intent, natural_language_answer, used_run_ids, comparison, flagged_run_id
                    answer = resp.get("natural_language_answer") or ""
                    used_run_ids = resp.get("used_run_ids") or []
                    flagged_run_id = resp.get("flagged_run_id")
                    comparison = resp.get("comparison")

                    # Build a nice markdown message
                    lines = [answer]

                    if used_run_ids:
                        lines.append("")
                        lines.append("**Runs referenced:**")
                        for rid in used_run_ids:
                            lines.append(f"- `{rid}`")

                    if flagged_run_id:
                        lines.append("")
                        lines.append(f"**Flagged run:** `{flagged_run_id}`")

                    if comparison:
                        try:
                            better = comparison.get("better_run_id")
                            if better:
                                lines.append("")
                                lines.append(f"**Agent thinks best run:** `{better}`")
                        except Exception:
                            pass

                    full_msg = "\n".join(lines)
                    st.markdown(full_msg)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": full_msg}
                    )
