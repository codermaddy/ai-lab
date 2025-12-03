// ui/src/pages/RunsList.jsx
import React, { useEffect, useState } from "react";
import { fetchRuns, fetchRunDetail } from "../api/runs";
import { Link } from "react-router-dom";

export default function RunsList() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true);
      try {
        const data = await fetchRuns();
        const baseRuns = Array.isArray(data) ? data : (data.runs || []);

        console.debug("fetchRuns =>", baseRuns);

        const list = Array.isArray(baseRuns) ? baseRuns : (baseRuns?.runs || baseRuns?.data || []);
        if (!list || list.length === 0) {
          if (mounted) {
            setRuns([]);
            setLoading(false);
          }
          return;
        }

        // fetch details in parallel but don't fail entirely if a detail fails
        const detailed = await Promise.all(
          list.map(async (r) => {
            const id = r.run_id || r.id;
            if (!id) return r;
            try {
              const full = await fetchRunDetail(id);
              return { ...r, ...full };
            } catch (err) {
              console.warn("detail fetch failed for", id, err);
              return r;
            }
          })
        );

        if (mounted) {
          setRuns(detailed);
          setLoading(false);
        }
      } catch (err) {
        console.error("RunsList load error", err);
        if (mounted) {
          setError(err.message || String(err));
          setLoading(false);
        }
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  if (loading) return <div className="p-6">Loading runs…</div>;
  if (error) return <div className="p-6 text-red-600">Error: {error}</div>;

  if (!runs.length) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-semibold mb-2">Runs</h1>
        <p className="text-sm text-slate-500">
          No runs found. Check backend is running and CORS. Also inspect browser console / network.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-semibold mb-2">Runs</h1>
      <div className="space-y-2">
        {runs.map((run) => {
          const id = run.run_id || run.id;
          const ts = run.timestamp || run.created_at;
          const fm = run.final_metrics || run.metrics || {};
          const acc = fm?.accuracy;
          const f1 = fm?.f1_macro ?? fm?.f1;
          const params = run.params || {};
          const titleParts = [
            run.task_name || params.task_name,
            run.experiment_name || params.experiment_name,
            run.model_name || params.model_name,
          ].filter(Boolean);
          const title = titleParts.join(" / ") || id;

          return (
            <Link key={id} to={`/runs/${encodeURIComponent(id)}`} className="block bg-white border rounded-lg shadow-sm p-4 hover:bg-slate-50">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-sm">Run: {title}</div>
                  {title !== id && <div className="text-[11px] text-slate-400 break-all">id: {id}</div>}
                  {ts && <div className="text-xs text-slate-500">{new Date(ts).toLocaleString()}</div>}
                  {run.note && <div className="text-xs text-amber-700 mt-1">Note: {run.note}</div>}
                </div>
                <div className="text-right text-xs text-slate-600">
                  {acc !== undefined && !Number.isNaN(acc) && <div>accuracy: {Number(acc).toFixed(3)}</div>}
                  {f1 !== undefined && !Number.isNaN(f1) && <div>f1_macro: {Number(f1).toFixed(3)}</div>}
                  {Array.isArray(run.artifacts) && run.artifacts.length > 0 && <div className="mt-1 text-[11px] text-slate-500">{run.artifacts.length} artifact(s)</div>}
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
