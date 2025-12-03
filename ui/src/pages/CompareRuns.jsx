// ui/src/pages/CompareRuns.jsx
import React, { useEffect, useMemo, useState } from "react";
import { fetchRuns, fetchRunDetail } from "../api/runs";
import CompareChart from "../components/CompareChart";
import { downloadCSV, downloadJSON } from "../utils/exportUtils";
import { Link } from "react-router-dom";


//  Helper to build a human-friendly title for a run
function buildRunTitle(run, fallbackId) {
  if (!run) return fallbackId;
  const params = run.params || {};

  const parts = [
    run.task_name || params.task_name,
    run.experiment_name || params.experiment_name,
    run.model_name || params.model_name,
  ].filter(Boolean);

  if (parts.length) return parts.join(" / ");
  return run.name || run.run_id || run.id || fallbackId;
}

export default function CompareRuns() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState([]); // array of run ids (run_id)
  const [metricOptions, setMetricOptions] = useState([]);
  const [metric, setMetric] = useState(""); // currently selected metric name
  const [runDetailsCache, setRunDetailsCache] = useState({}); // id -> run detail

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const raw = await fetchRuns();
        // fetchRuns() may return either `[]` or `{ runs: [...] }` depending on implementation.
        const list = Array.isArray(raw) ? raw : (raw?.runs || []);

        // 🔹 fetch full details for each run so we get params / metrics
        const detailed = await Promise.all(
          list.map(async (r) => {
            const id = r.run_id || r.id;
            try {
              const full = await fetchRunDetail(id);
              return { ...r, ...full };
            } catch {
              return r;
            }
          })
        );

        setRuns(detailed);

        // collect metric keys from final_metrics (backend stores final_metrics)
        const allMetrics = new Set();
        detailed.forEach((r) => {
          if (r.final_metrics && typeof r.final_metrics === "object") {
            Object.keys(r.final_metrics).forEach((k) => allMetrics.add(k));
          }
          if (r.metrics && typeof r.metrics === "object") {
            Object.keys(r.metrics).forEach((k) => allMetrics.add(k));
          }
        });

        const metricsArray = Array.from(allMetrics);
        setMetricOptions(metricsArray);
        if (!metric && metricsArray.length) setMetric(metricsArray[0]);
      } catch (err) {
        console.error("fetchRuns error:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // helper: load details for a run id (with cache)
  async function ensureRunDetail(id) {
    if (runDetailsCache[id]) return runDetailsCache[id];
    try {
      const detail = await fetchRunDetail(id);
      setRunDetailsCache((prev) => ({ ...prev, [id]: detail }));
      return detail;
    } catch (e) {
      console.error("failed fetchRunDetail", id, e);
      return null;
    }
  }

  // Build series for the selected runs & chosen metric
  const series = useMemo(() => {
    if (!selected.length || !metric) return [];

    return selected
      .map((id) => {
        const cached =
          runDetailsCache[id] ||
          runs.find((r) => (r.run_id || r.id) === id) ||
          {};

        const runName = buildRunTitle(cached, id);

        // Prefer final_metrics, fallback to metrics
        const runMetrics =
          cached.final_metrics || cached.metrics || cached.metrics_summary || {};

        const m = runMetrics[metric];
        let points = [];

        if (Array.isArray(m) && m.length && typeof m[0] === "object") {
          // assume {step, value, ts}
          points = m
            .map((p) => {
              const x =
                p.step ??
                p.x ??
                (p.ts ? new Date(p.ts).getTime() : undefined);
              return {
                x: x ?? 0,
                y: p.value ?? p.y ?? p.metric ?? null,
                ts: p.ts ?? null,
              };
            })
            .filter((p) => p.y !== null);
        } else if (typeof m === "object" && m !== null) {
          const arr = m.values || m.data || null;
          if (Array.isArray(arr)) {
            points = arr.map((p, idx) => ({
              x: p.step ?? idx + 1,
              y: p.value ?? p,
            }));
          }
        } else if (typeof m === "number" || typeof m === "string") {
          // scalar -> single-point
          points = [{ x: 1, y: Number(m) }];
        } else if (Array.isArray(m)) {
          points = m.map((v, idx) => ({ x: idx + 1, y: v }));
        } else {
          points = [];
        }

        return {
          runId: id,
          runName,
          points,
        };
      })
      .filter((s) => s.points && s.points.length > 0);
  }, [selected, metric, runDetailsCache, runs]);

  // Handler: toggle selection (limit to 5 runs)
  function toggleSelect(id) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 5) return prev; // limit
      return [...prev, id];
    });
  }

  async function handleExportJSON() {
    const payload = { metric, runs: [] };
    for (const id of selected) {
      const detail = await ensureRunDetail(id);
      const title = buildRunTitle(detail, id);
      const metrics =
        detail?.final_metrics ||
        detail?.metrics ||
        detail?.metrics_summary ||
        {};
      payload.runs.push({
        id,
        name: title,
        metric_data: metrics[metric],
      });
    }
    downloadJSON("compare_runs.json", payload);
  }

  async function handleExportCSV() {
    // Build rows: run_id, run_name, step, value, ts
    const rows = [];
    for (const id of selected) {
      const detail = await ensureRunDetail(id);
      const title = buildRunTitle(detail, id);
      const metrics =
        detail?.final_metrics ||
        detail?.metrics ||
        detail?.metrics_summary ||
        {};
      const m = metrics[metric];

      if (Array.isArray(m) && m.length && typeof m[0] === "object") {
        m.forEach((pt) => {
          rows.push({
            run_id: id,
            run_name: title,
            step: pt.step ?? pt.x ?? "",
            value: pt.value ?? pt.y ?? "",
            ts: pt.ts ?? "",
          });
        });
      } else if (typeof m === "number" || typeof m === "string") {
        rows.push({
          run_id: id,
          run_name: title,
          step: 1,
          value: m,
          ts: "",
        });
      } else if (Array.isArray(m)) {
        m.forEach((v, idx) =>
          rows.push({
            run_id: id,
            run_name: title,
            step: idx + 1,
            value: v,
            ts: "",
          }),
        );
      }
    }

    downloadCSV("compare_runs.csv", rows, [
      "run_id",
      "run_name",
      "step",
      "value",
      "ts",
    ]);
  }

  if (loading) return <div className="p-6">Loading runs…</div>;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Compare Runs</h1>

        <div className="flex items-center gap-3">
          <select
            className="border rounded p-2 bg-white"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            <option value="">Select metric</option>
            {metricOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          <button
            onClick={handleExportCSV}
            disabled={!selected.length || !metric}
            className="px-3 py-1 rounded bg-slate-800 text-white disabled:opacity-40"
          >
            Export CSV
          </button>

          <button
            onClick={handleExportJSON}
            disabled={!selected.length || !metric}
            className="px-3 py-1 rounded border"
          >
            Export JSON
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-1 bg-white p-4 rounded shadow-sm">
          <h3 className="font-medium mb-2">
            Available runs ({runs.length})
          </h3>

          <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
            {runs.map((r) => {
              const id = r.run_id || r.id;
              const ts = r.timestamp || r.created_at;
              const title = buildRunTitle(r, id);
              return (
                <div
                  key={id}
                  className="flex items-start gap-2 p-2 border rounded"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(id)}
                    onChange={() => toggleSelect(id)}
                  />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <Link
                        to={`/runs/${encodeURIComponent(id)}`}
                        className="font-medium"
                      >
                        {title}
                      </Link>
                      {ts && (
                        <div className="text-xs text-slate-500">
                          {new Date(ts).toLocaleDateString()}
                        </div>
                      )}
                    </div>

                    <div className="text-xs text-slate-500">
                      {r.tags?.slice?.(0, 3)?.join(", ")}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-3 text-xs text-slate-500">
            Tip: select up to 5 runs. If the metric has time-series, it will be
            overlaid; otherwise, you’ll see scalar values.
          </div>
        </div>

        <div className="md:col-span-2 bg-white p-4 rounded shadow-sm">
          <h3 className="font-medium mb-3">Comparison</h3>

          {selected.length === 0 && (
            <div className="text-slate-500">
              Select runs on the left to compare metrics.
            </div>
          )}

          {selected.length > 0 && !metric && (
            <div className="text-slate-500">
              Select a metric to visualize.
            </div>
          )}

          {selected.length > 0 && metric && (
            <ComparisonView
              selected={selected}
              metric={metric}
              series={series}
              runs={runs}
              runDetailsCache={runDetailsCache}
            />
          )}

        </div>
      </div>
    </div>
  );
}

function ComparisonView({ selected, metric, series }) {
  const hasTimeSeries =
    series.length > 0 && series.some((s) => s.points.length > 1);

  // return (
  //   <div>
  //     <div className="mb-4">
  //       <strong>Metric:</strong> {metric}
  //     </div>

  //     {hasTimeSeries ? (
  //       <CompareChart series={series} xKeyLabel="step" />
  //     ) : (
  //       <ScalarTable series={series} />
  //     )}
  //   </div>
  // );
  return (
    <div>
      <div className="mb-4">
        <strong>Metric:</strong> {metric}
      </div>
      <CompareChart series={series} xKeyLabel="step" />
    </div>
  );
}

function ScalarTable({ series }) {
  if (!series.length)
    return <div className="text-slate-500">No data to show.</div>;
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-left">
          <th className="border-b py-2">Run</th>
          <th className="border-b py-2">Value</th>
        </tr>
      </thead>
      <tbody>
        {series.map((s) => (
          <tr key={s.runId}>
            <td className="py-2 border-b">{s.runName}</td>
            <td className="py-2 border-b">{s.points[0]?.y ?? "-"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
