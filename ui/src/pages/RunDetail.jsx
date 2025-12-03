// ui/src/pages/RunDetail.jsx
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchRunDetail, updateRunNote } from "../api/runs";

export default function RunDetail() {
  const { id } = useParams();
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [noteDraft, setNoteDraft] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetchRunDetail(id)
      .then((data) => {
        setRun(data);
        setNoteDraft(data.note || "");
        setLoading(false);
      })
      .catch((err) => {
        console.error("fetchRunDetail error:", err);
        setError(err.message || "Failed to load run");
        setLoading(false);
      });
  }, [id]);

  if (loading) return <div className="p-6">Loading run…</div>;
  if (error) return <div className="p-6 text-red-600">Error: {error}</div>;
  if (!run) return <div className="p-6">Run not found.</div>;

  const runId = run.run_id || run.id;
  const ts = run.timestamp || run.created_at;
  const params = run.params || {};
  const finalMetrics = run.final_metrics || run.metrics || {};
  const artifacts = Array.isArray(run.artifacts) ? run.artifacts : [];

  const titleParts = [
    params.task_name,
    params.experiment_name,
    params.model_name,
  ].filter(Boolean);
  const title = titleParts.join(" / ") || runId;

  const handleSaveNote = async () => {
    try {
      setSavingNote(true);
      const updated = await updateRunNote(runId, noteDraft);
      setRun(updated);
    } catch (err) {
      console.error("updateRunNote error:", err);
      alert("Failed to update note: " + err.message);
    } finally {
      setSavingNote(false);
    }
  };

  const describeArtifact = (path) => {
    const fileName = path.split(/[\\/]/).pop() || path;
    const lower = fileName.toLowerCase();

    let kind = "Artifact";
    let hint = "";

    if (lower.endsWith(".csv")) {
      kind = "Metrics CSV";
      hint = "Tabular metrics or logs";
    } else if (
      lower.endsWith(".png") ||
      lower.endsWith(".jpg") ||
      lower.endsWith(".jpeg")
    ) {
      kind = "Plot / Figure";
      hint = "Saved chart or visualization";
    } else if (
      lower.endsWith(".joblib") ||
      lower.endsWith(".pkl") ||
      lower.endsWith(".pt") ||
      lower.endsWith(".bin")
    ) {
      kind = "Model Checkpoint";
      hint = "Serialized model weights or state";
    } else if (lower.endsWith(".json")) {
      kind = "JSON Summary";
      hint = "Config, metrics, or summary data";
    }

    return { fileName, kind, hint };
  };

  // ---------- helpers for pretty key/value rendering ----------
  const prettyLabel = (key) =>
    key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

  const renderKVGrid = (obj) => {
    const entries = Object.entries(obj || {});
    if (!entries.length) {
      return (
        <div className="text-sm text-slate-500">
          No values logged for this section.
        </div>
      );
    }

    return (
      <div className="grid md:grid-cols-2 gap-x-8 gap-y-2">
        {entries.map(([k, v]) => (
          <div key={k} className="flex flex-col text-sm">
            <span className="text-[11px] uppercase tracking-wide text-slate-400">
              {prettyLabel(k)}
            </span>
            <span className="font-mono text-[13px] text-slate-800 break-all">
              {typeof v === "number"
                ? Number.isInteger(v)
                  ? v
                  : v.toFixed(6).replace(/0+$/, "").replace(/\.$/, "")
                : Array.isArray(v) || typeof v === "object"
                ? JSON.stringify(v)
                : String(v)}
            </span>
          </div>
        ))}
      </div>
    );
  };

  // pull out some "headline" metrics
  const headlineMetrics = {
    accuracy:
      typeof finalMetrics.accuracy === "number"
        ? finalMetrics.accuracy
        : null,
    f1_macro:
      typeof finalMetrics.f1_macro === "number"
        ? finalMetrics.f1_macro
        : typeof finalMetrics.f1 === "number"
        ? finalMetrics.f1
        : null,
    final_val_loss:
      typeof finalMetrics.final_val_loss === "number"
        ? finalMetrics.final_val_loss
        : null,
    runtime_sec:
      typeof finalMetrics.runtime_sec === "number"
        ? finalMetrics.runtime_sec
        : null,
  };

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{title}</h1>
          <div className="text-xs text-slate-400 break-all">
            id: {runId}
          </div>
          {ts && (
            <div className="text-sm text-slate-500">
              {new Date(ts).toLocaleString()}
            </div>
          )}
        </div>

        {run.wandb_url && (
          <a
            className="px-3 py-1 rounded border text-sm"
            href={run.wandb_url}
            target="_blank"
            rel="noreferrer"
          >
            Open in W&B
          </a>
        )}
      </div>

      {/* Params (nice grid) */}
      <div className="bg-white border rounded-lg shadow-sm p-4 space-y-3">
        <h2 className="font-semibold text-lg">Params</h2>
        {renderKVGrid(params)}
      </div>

      {/* Final metrics – highlight key ones */}
      <div className="bg-white border rounded-lg shadow-sm p-4 space-y-4">
        <h2 className="font-semibold text-lg">Final Metrics</h2>

        {/* headline badges */}
        <div className="flex flex-wrap gap-3">
          {headlineMetrics.accuracy != null && (
            <MetricBadge
              label="Accuracy"
              value={headlineMetrics.accuracy}
              decimals={3}
            />
          )}
          {headlineMetrics.f1_macro != null && (
            <MetricBadge
              label="F1 Macro"
              value={headlineMetrics.f1_macro}
              decimals={3}
            />
          )}
          {headlineMetrics.final_val_loss != null && (
            <MetricBadge
              label="Final Val Loss"
              value={headlineMetrics.final_val_loss}
              decimals={4}
            />
          )}
          {headlineMetrics.runtime_sec != null && (
            <MetricBadge
              label="Runtime (sec)"
              value={headlineMetrics.runtime_sec}
              decimals={3}
            />
          )}
        </div>

        {/* all metrics as key/value grid */}
        <div className="pt-2 border-t border-slate-100">
          {renderKVGrid(finalMetrics)}
        </div>
      </div>

      {/* Artifacts */}
      <div className="bg-white border rounded-lg shadow-sm p-4">
        <h2 className="font-semibold text-lg mb-2">Artifacts</h2>

        {artifacts.length === 0 ? (
          <div className="text-sm text-slate-500">
            No artifacts logged for this run.
          </div>
        ) : (
          <>
            <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3">
              {artifacts.map((p, idx) => {
                const { fileName, kind, hint } = describeArtifact(p);
                return (
                  <div
                    key={idx}
                    className="border rounded-md bg-slate-50 px-3 py-2 text-xs flex flex-col justify-between"
                  >
                    <div className="font-semibold text-[12px] mb-1">
                      {kind}
                    </div>
                    <div className="text-[11px] text-slate-700 truncate">
                      {fileName}
                    </div>
                    {hint && (
                      <div className="text-[10px] text-slate-500 mt-1">
                        {hint}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              These artifacts are stored on the backend machine (e.g. metrics
              CSVs, plots, model checkpoints). They’re used by the logger/agent
              for deeper analysis, but are not directly downloadable from the
              browser.
            </p>
          </>
        )}
      </div>

      {/* Note editor (flag/publish) */}
      <div className="bg-white border rounded-lg shadow-sm p-4 space-y-2">
        <h2 className="font-semibold text-lg mb-2">Note / Tag</h2>
        <textarea
          className="w-full border rounded p-2 text-sm"
          rows={3}
          value={noteDraft}
          onChange={(e) => setNoteDraft(e.target.value)}
          placeholder='Example: "PUBLISH: best model" or "BAD RUN: wrong seed"'
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleSaveNote}
            disabled={savingNote}
            className="px-3 py-1 rounded bg-slate-800 text-white text-sm hover:opacity-90 disabled:opacity-50"
          >
            {savingNote ? "Saving…" : "Save note"}
          </button>
          {run.note && (
            <div className="text-xs text-slate-500">
              Current note: <span className="font-mono">{run.note}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// small badge component for headline metrics
function MetricBadge({ label, value, decimals = 3 }) {
  const formatted =
    typeof value === "number"
      ? value.toFixed(decimals)
      : String(value);

  return (
    <div className="inline-flex flex-col px-3 py-2 rounded-lg bg-slate-900 text-white text-xs shadow-sm">
      <span className="uppercase tracking-wide text-[10px] text-slate-200">
        {label}
      </span>
      <span className="text-sm font-semibold">{formatted}</span>
    </div>
  );
}
