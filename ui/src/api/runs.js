// ui/src/api/runs.js
const BASE =
  import.meta.env.VITE_API_BASE ||
  import.meta.env.VITE_BACKEND_URL ||
  "http://localhost:8000";

function _url(path = "") {
  return `${BASE.replace(/\/$/, "")}${path}`;
}

async function _fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const txt = await res.text().catch(() => "<no-body>");
    throw new Error(`${res.status} ${res.statusText} - ${txt}`);
  }
  return res.json();
}

/**
 * fetchRuns - always returns an array of runs (empty array if none)
 * Accepts optional params object -> used to build query string.
 */
export async function fetchRuns(params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `${BASE.replace(/\/$/, "")}/runs${qs ? "?" + qs : ""}`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`fetchRuns ${res.status}: ${body}`);
  }
  const data = await res.json();

  // Normalize: always return { runs: [...] }
  if (Array.isArray(data)) return { runs: data };
  if (Array.isArray(data?.runs)) return { runs: data.runs };
  if (Array.isArray(data?.data)) return { runs: data.data };
  // fallback
  return { runs: [] };
}

export async function fetchRunDetail(id) {
  if (!id) throw new Error("missing id");
  const url = _url(`/runs/${encodeURIComponent(id)}`);
  console.debug("fetchRunDetail ->", url);
  return _fetchJson(url);
}

export async function compareRuns(runIds = []) {
  if (!runIds.length) return null;
  const query = runIds.map(id => `ids=${encodeURIComponent(id)}`).join("&");
  const url = _url(`/runs/compare?${query}`);
  return _fetchJson(url);
}

export async function updateRunNote(runId, note) {
  const url = _url(`/runs/${encodeURIComponent(runId)}/note`);
  return _fetchJson(url, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ note }) });
}

export async function fetchLeaderboard(metricName = "accuracy", topK = 10) {
  const url = _url(`/runs/leaderboard?metric_name=${encodeURIComponent(metricName)}&top_k=${topK}`);
  return _fetchJson(url);
}
