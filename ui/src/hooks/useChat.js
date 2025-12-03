// // ui/src/hooks/useChat.js
// import { useEffect, useState } from "react";

// const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
// const STORAGE_KEY = "ai-lab-chat-history";
// const MAX_MESSAGES = 50; // keep last 50 to avoid huge history

// export function useChat({ persist = true } = {}) {
//   const [messages, setMessages] = useState([]);
//   const [loading, setLoading]   = useState(false);
//   const [error, setError]       = useState(null);

//   // Load persisted chat
//   useEffect(() => {
//     if (!persist) return;
//     try {
//       const raw = window.localStorage.getItem(STORAGE_KEY);
//       if (raw) {
//         const parsed = JSON.parse(raw);
//         if (Array.isArray(parsed)) {
//           setMessages(parsed);
//         }
//       }
//     } catch (e) {
//       console.warn("Failed to load chat history:", e);
//     }
//   }, [persist]);

//   // Save persisted chat
//   useEffect(() => {
//     if (!persist) return;
//     try {
//       window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
//     } catch (e) {
//       console.warn("Failed to save chat history:", e);
//     }
//   }, [messages, persist]);

//   const clear = () => {
//     setMessages([]);
//     setError(null);
//     if (persist) {
//       try {
//         window.localStorage.removeItem(STORAGE_KEY);
//       } catch {}
//     }
//   };

//   // 🔹 helper: keep only last N messages
//   const pushMessage = (msg) => {
//     setMessages((prev) => {
//       const next = [...prev, msg];
//       if (next.length > MAX_MESSAGES) {
//         return next.slice(-MAX_MESSAGES);
//       }
//       return next;
//     });
//   };

//   // 🔹 helper: fetch details for used_run_ids and build a summary
//   const buildRunsSummary = async (ids) => {
//     try {
//       const res = await fetch(`${BASE}/runs`);
//       if (!res.ok) return null;

//       const data = await res.json();
//       const allRuns = Array.isArray(data) ? data : data.runs || [];
//       const byId = new Map(
//         allRuns.map((r) => [(r.run_id || r.id), r])
//       );

//       const lines = [];

//       ids.forEach((id) => {
//         const run = byId.get(id);
//         if (!run) {
//           lines.push(`- ${id}`);
//           return;
//         }
//         const params = run.params || {};
//         const fm = run.final_metrics || run.metrics || {};
//         const acc = fm.accuracy;
//         const f1  = fm.f1_macro || fm.f1;

//         const titleParts = [
//           params.task_name,
//           params.experiment_name,
//           params.model_name,
//         ].filter(Boolean);
//         const title = titleParts.join(" / ") || "(untitled)";

//         const metricStr = [
//           acc != null ? `accuracy=${acc.toFixed(3)}` : null,
//           f1  != null ? `f1_macro=${f1.toFixed(3)}` : null,
//         ].filter(Boolean).join(", ");

//         lines.push(
//           `- ${id}\n  • ${title}` +
//           (metricStr ? `\n  • ${metricStr}` : "")
//         );
//       });

//       return lines.join("\n");
//     } catch (e) {
//       console.warn("Failed to fetch run summaries:", e);
//       return null;
//     }
//   };

//   const sendMessage = async (text) => {
//     if (!text || !text.trim()) return;
//     const clean = text.trim();
//     setError(null);

//     const now = new Date().toISOString();

//     // push user message
//     const userMsg = {
//       id: `u-${Date.now()}`,
//       role: "user",
//       text: clean,
//       created_at: now,
//     };
//     pushMessage(userMsg);

//     setLoading(true);
//     try {
//       const res = await fetch(`${BASE}/agent/query`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query: clean }),
//       });

//       if (!res.ok) {
//         const txt = await res.text();
//         throw new Error(`Agent error: ${res.status} ${txt}`);
//       }

//       const data = await res.json();

//       // main answer from agent
//       const mainText =
//         data.natural_language_answer ||
//         "Agent returned no natural_language_answer.";

//       let extras = "";

//       // 🔹 richer recent runs details
//       if (Array.isArray(data.used_run_ids) && data.used_run_ids.length) {
//         const summary = await buildRunsSummary(data.used_run_ids);
//         if (summary) {
//           extras += `\n\nRuns referenced:\n${summary}`;
//         } else {
//           extras += `\n\nRuns referenced: ${data.used_run_ids.join(", ")}`;
//         }
//       }

//       // 🔹 show comparison object in a compact way if present
//       if (data.comparison) {
//         try {
//           const compact = JSON.stringify(data.comparison, null, 2)
//             .slice(0, 1200); // avoid huge payloads
//           extras += `\n\nComparison:\n${compact}`;
//         } catch {
//           extras += `\n\nComparison: [object]`;
//         }
//       }

//       if (data.intent) {
//         extras += `\n\n(Intent: ${data.intent})`;
//       }
//       if (data.flagged_run_id) {
//         extras += `\nFlagged run: ${data.flagged_run_id}`;
//       }

//       const assistantMsg = {
//         id: `a-${Date.now()}`,
//         role: "assistant",
//         text: mainText + (extras || ""),
//         created_at: new Date().toISOString(),
//       };

//       pushMessage(assistantMsg);
//     } catch (err) {
//       console.error("sendMessage error:", err);
//       setError(err.message || "Failed to contact agent.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return {
//     messages,
//     sendMessage,
//     loading,
//     error,
//     clear,
//   };
// }

import { useEffect, useState } from "react";
import { fetchRunDetail } from "../api/runs";

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const STORAGE_KEY = "ai-lab-chat-history";
const MAX_MESSAGES = 50; // keep last N messages so UI doesn't hang

// Helper: build a human title for a run
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

// Helper: summarize a run (title + id + key metrics + timestamp)
async function summarizeRun(id) {
  try {
    const detail = await fetchRunDetail(id);
    const title = buildRunTitle(detail, id);

    const fm =
      detail.final_metrics || detail.metrics || detail.metrics_summary || {};
    const parts = [];

    if (fm.accuracy != null) parts.push(`accuracy=${fm.accuracy.toFixed(3)}`);
    if (fm.f1_macro != null)
      parts.push(`f1_macro=${fm.f1_macro.toFixed(3)}`);
    if (fm.final_val_loss != null)
      parts.push(`final_val_loss=${fm.final_val_loss.toFixed(4)}`);
    if (fm.runtime_sec != null)
      parts.push(`runtime_sec=${fm.runtime_sec.toFixed(3)}`);

    const metricsLine = parts.length
      ? parts.join(", ")
      : "no key metrics logged";

    const ts = detail.timestamp || detail.created_at;
    const when = ts ? new Date(ts).toLocaleString() : null;

    const lines = [
      `- ${title}`,
      `  id: ${detail.run_id || detail.id || id}`,
      when ? `  date: ${when}` : null,
      `  metrics: ${metricsLine}`,
    ].filter(Boolean);

    return lines.join("\n");
  } catch (e) {
    console.warn("summarizeRun failed for", id, e);
    return `- ${id}`;
  }
}

export function useChat({ persist = true } = {}) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  // Load persisted chat
  useEffect(() => {
    if (!persist) return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setMessages(parsed);
        }
      }
    } catch (e) {
      console.warn("Failed to load chat history:", e);
    }
  }, [persist]);

  // Save persisted chat
  useEffect(() => {
    if (!persist) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.warn("Failed to save chat history:", e);
    }
  }, [messages, persist]);

  const clear = () => {
    setMessages([]);
    setError(null);
    if (persist) {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {}
    }
  };

  // keep history short so UI stays snappy
  const pushMessage = (msg) => {
    setMessages((prev) => {
      const next = [...prev, msg];
      if (next.length > MAX_MESSAGES) return next.slice(-MAX_MESSAGES);
      return next;
    });
  };

  const sendMessage = async (text) => {
    if (!text || !text.trim()) return;
    const clean = text.trim();
    setError(null);

    const now = new Date().toISOString();

    // user message
    const userMsg = {
      id: `u-${Date.now()}`,
      role: "user",
      text: clean,
      created_at: now,
    };
    pushMessage(userMsg);

    setLoading(true);
    try {
      const res = await fetch(`${BASE}/agent/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: clean }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Agent error: ${res.status} ${txt}`);
      }

      const data = await res.json();

      const sections = [];

      // 1) main natural-language answer from the agent
      sections.push(
        data.natural_language_answer ||
          "Agent returned no natural_language_answer."
      );

      // 2) richer details for runs the agent touched
      if (Array.isArray(data.used_run_ids) && data.used_run_ids.length) {
        const summaries = await Promise.all(
          data.used_run_ids.map((id) => summarizeRun(id))
        );
        sections.push(`Runs referenced:\n${summaries.join("\n")}`);
      }

      // 3) if a best / flagged run exists, give a dedicated section
      if (data.flagged_run_id) {
        const bestSummary = await summarizeRun(data.flagged_run_id);
        sections.push(`Best run details:\n${bestSummary}`);
      }

      // 4) comparison object (keep compact)
      if (data.comparison) {
        try {
          const compact = JSON.stringify(data.comparison, null, 2).slice(
            0,
            1200
          );
          sections.push(`Comparison summary:\n${compact}`);
        } catch {
          sections.push("Comparison summary: [unprintable object]");
        }
      }

      // 5) intent tag (small hint at the end)
      if (data.intent) {
        sections.push(`(Intent: ${data.intent})`);
      }

      const assistantMsg = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: sections.join("\n\n"),
        created_at: new Date().toISOString(),
      };

      pushMessage(assistantMsg);
    } catch (err) {
      console.error("sendMessage error:", err);
      setError(err.message || "Failed to contact agent.");
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    sendMessage,
    loading,
    error,
    clear,
  };
}
