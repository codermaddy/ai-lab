import React, { useState } from "react";
import { flagRun, unflagRun } from "../api/runs";

/**
 * Props:
 * - runId (string, required)
 * - initialFlag (boolean) optional
 * - onUpdate(updatedRun) optional callback (will be passed updated run from API)
 */
export default function FlagButton({ runId, initialFlag = false, onUpdate = () => {} }) {
  const [isFlagged, setIsFlagged] = useState(initialFlag);
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleFlag(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await flagRun(runId, { flagged: true, note });
      setIsFlagged(true);
      setOpen(false);
      onUpdate(resp);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to flag run");
    } finally {
      setLoading(false);
    }
  }

  async function handleUnflag() {
    if (!confirm("Unflag this run for publish?")) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await unflagRun(runId);
      setIsFlagged(false);
      onUpdate(resp);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to unflag run");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="flex items-center gap-2">
        {isFlagged ? (
          <button
            title="Unflag run"
            onClick={handleUnflag}
            disabled={loading}
            className="inline-flex items-center gap-2 px-3 py-1 rounded bg-yellow-400 text-black hover:brightness-95"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M19 21V7M5 7h14l-2 14H7L5 7z" />
            </svg>
            Flagged
          </button>
        ) : (
          <button
            onClick={() => setOpen(true)}
            disabled={loading}
            className="inline-flex items-center gap-2 px-3 py-1 rounded border hover:bg-slate-50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" d="M5 5v14M5 7h12l-2 4 2 4H5" />
            </svg>
            Flag for publish
          </button>
        )}
      </div>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <form
            onSubmit={handleFlag}
            className="relative z-10 w-full max-w-lg bg-white rounded-lg p-6 shadow-lg"
          >
            <h3 className="text-lg font-semibold mb-2">Flag run for publish</h3>
            <p className="text-sm text-slate-600 mb-4">Add optional notes for reviewers or publication metadata.</p>

            <label className="block mb-2 text-sm">
              Notes (optional)
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                className="w-full mt-1 border rounded p-2 min-h-[100px]"
              />
            </label>

            {error && <div className="text-sm text-red-600 mb-2">{error}</div>}

            <div className="flex items-center justify-end gap-2">
              <button type="button" onClick={() => setOpen(false)} className="px-3 py-1 rounded border">Cancel</button>
              <button type="submit" disabled={loading} className="px-3 py-1 rounded bg-yellow-500 text-black">
                {loading ? "Flagging…" : "Flag and request publish"}
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
