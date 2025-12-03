// ui/src/components/Chatbox.jsx
import React, { useEffect, useRef, useState } from "react";
import { useChat } from "../hooks/useChat";

export default function Chatbox({ initialPrompt, clearOnMount = true }) {
  const { messages, sendMessage, loading, error, clear } = useChat({
    persist: true,
  });

  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState([]);

  const listRef = useRef(null);

  // Optionally clear history when this component mounts
  useEffect(() => {
    if (clearOnMount) {
      clear();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearOnMount]);

  // Scroll to bottom when messages update
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight + 500;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return;
    const text = input.trim();
    setInput("");

    // current backend / useChat only expects the text.
    // Attachments UI is local-only for now (backend ignores them).
    await sendMessage(text || "(no message, only attachments)");
    setAttachments([]);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleAttach = (e) => {
    const files = Array.from(e.target.files || []);
    const mapped = files.map((f) => ({
      id: Math.random().toString(36).slice(2),
      name: f.name,
      url: URL.createObjectURL(f),
    }));
    setAttachments((prev) => [...prev, ...mapped]);
    e.target.value = "";
  };

  const removeAttachment = (id) => {
    setAttachments((prev) => prev.filter((x) => x.id !== id));
  };

  const suggestions = [
    "Show recent runs",
    "Compare top 3 runs by accuracy",
    "Show details for the best run",
    "Show artifacts for the latest run",
  ];

  return (
    <div className="flex flex-col h-[70vh] border rounded bg-white shadow-sm">
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between">
        <h2 className="font-medium">AI Assistant</h2>
        <button
          onClick={clear}
          className="text-xs text-slate-500 hover:text-red-500"
          type="button"
        >
          Clear chat
        </button>
      </div>

      {/* Messages */}
      <div ref={listRef} className="flex-1 overflow-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`max-w-[80%] ${
              msg.role === "user" ? "ml-auto text-right" : "mr-auto"
            }`}
          >
            <div
              className={`p-3 rounded-xl whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-slate-800 text-white"
                  : "bg-slate-100 text-slate-900"
              }`}
            >
              {msg.text}
              {msg.streaming && (
                <div className="text-xs text-slate-500 mt-2">...thinking</div>
              )}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {new Date(msg.created_at).toLocaleTimeString()}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-sm text-slate-400 italic">
            Assistant typing…
          </div>
        )}
      </div>

      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="p-2 border-t bg-slate-50 flex gap-2 overflow-x-auto">
          {attachments.map((a) => (
            <div
              key={a.id}
              className="flex items-center gap-2 border rounded px-2 py-1 bg-white shadow-sm"
            >
              <span className="text-xs">{a.name}</span>
              <button
                className="text-xs text-red-500"
                onClick={() => removeAttachment(a.id)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Suggestion chips */}
      <div className="p-2 border-t bg-slate-50 flex gap-2 overflow-x-auto">
        {suggestions.map((s) => (
          <button
            key={s}
            className="px-3 py-1 rounded border text-sm hover:bg-slate-100"
            onClick={() => sendMessage(s)}
            type="button"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Composer */}
      <div className="p-3 border-t">
        <div className="flex gap-3">
          <textarea
            value={input}
            onKeyDown={handleKey}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something…"
            className="flex-1 border rounded p-3 min-h-[60px]"
          />

          <div className="flex flex-col gap-2">
            <label className="px-3 py-2 border rounded cursor-pointer text-sm text-center">
              Attach
              <input
                type="file"
                multiple
                className="hidden"
                onChange={handleAttach}
              />
            </label>

            <button
              onClick={handleSend}
              className="px-4 py-2 rounded bg-slate-800 text-white"
              type="button"
            >
              Send
            </button>
          </div>
        </div>

        {error && <div className="text-sm text-red-600 mt-2">{error}</div>}
      </div>
    </div>
  );
}
