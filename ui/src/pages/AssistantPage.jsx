// ui/src/pages/AssistantPage.jsx
import React, { useMemo } from "react";
import Chatbox from "../components/Chatbox";
import { useLocation } from "react-router-dom";

/**
 * AssistantPage
 * - left column: helper info + quick tool buttons
 * - right column: Chatbox component
 */

export default function AssistantPage() {
  const location = useLocation();
  const chatKey = useMemo(() => `chat-${location.key ?? location.pathname}`, [location.key, location.pathname]);

  const suggested = [
    "Show recent runs",
    "Compare best runs",
    "Flag run r1 for publish",
    "Show artifacts for run r1"
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <aside className="lg:col-span-1 bg-white border rounded-lg shadow-sm p-4">
        <h2 className="text-lg font-semibold mb-2">Assistant</h2>
        <p className="text-sm text-slate-600 mb-4">
          Ask the AI about experiments, runs, artifacts, or request actions (flag, compare).
        </p>

        <div className="mb-4">
          <h3 className="text-sm font-medium mb-2">Quick suggestions</h3>
          <div className="flex flex-col gap-2">
            {suggested.map(s => (
              <button
                key={s}
                onClick={() => window.location.href = "/assistant"}
                className="text-sm p-2 border rounded text-left hover:bg-slate-50"
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="text-xs text-slate-500">
          Tip: You can attach artifacts or use suggested chips in the chat composer.
        </div>
      </aside>

      <main className="lg:col-span-2">
        <div className="bg-white border rounded-lg shadow-sm p-4">
          <Chatbox key={chatKey} />
        </div>
      </main>
    </div>
  );
}


