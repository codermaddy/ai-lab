import React, { useEffect, useState } from "react";


function simpleCSVParse(text, maxRows = 50, maxCols = 20) {
  // Very simple CSV parser — splits by lines and commas, does not handle quoted commas robustly.
  const lines = text.split(/\r?\n/).filter(Boolean).slice(0, maxRows);
  return lines.map(line => {
    // naive split; trim whitespace
    const cols = line.split(",").slice(0, maxCols).map(c => c.trim());
    return cols;
  });
}

export default function ArtifactPreview({ artifact, onClose = () => {} }) {
  const [contentUrl, setContentUrl] = useState(artifact?.embed_url || artifact?.url || "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [previewType, setPreviewType] = useState(null); // 'image','pdf','csv','json','text','audio','video','download'
  const [previewData, setPreviewData] = useState(null); // for parsed CSV / text / json

  useEffect(() => {
    async function decide() {
      setError("");
      setLoading(true);
      setPreviewData(null);

      const url = artifact?.embed_url || artifact?.url;
      setContentUrl(url);

      if (!url) {
        setError("No URL available for this artifact.");
        setLoading(false);
        return;
      }

      // Determine type by explicit `artifact.type` if provided, otherwise try HEAD to detect content-type.
      let contentType = artifact?.type || null;
      try {
        if (!contentType) {
          // Try HEAD first — some servers block HEAD; handle gracefully with fallback to GET.
          const headResp = await fetch(url, { method: "HEAD" });
          if (headResp.ok) {
            contentType = headResp.headers.get("content-type");
          } else {
            // fallback to a small GET for text-based types
            const resp = await fetch(url, { method: "GET", headers: { Range: "bytes=0-2048" } });
            if (resp.ok) {
              contentType = resp.headers.get("content-type");
            }
          }
        }
      } catch (e) {
        // network / CORS may block HEAD — ignore and continue to guess from extension
        console.warn("HEAD failed for artifact:", e);
      }

      // fallback: guess from filename
      if (!contentType && artifact?.filename) {
        const ext = artifact.filename.split(".").pop()?.toLowerCase();
        if (ext === "csv" || ext === "tsv") contentType = "text/csv";
        else if (["png", "jpg", "jpeg", "gif", "webp", "bmp"].includes(ext)) contentType = `image/${ext === "jpg" ? "jpeg" : ext}`;
        else if (ext === "pdf") contentType = "application/pdf";
        else if (["mp4", "webm"].includes(ext)) contentType = `video/${ext}`;
        else if (["mp3", "wav"].includes(ext)) contentType = `audio/${ext}`;
        else if (["json"].includes(ext)) contentType = "application/json";
        else contentType = "application/octet-stream";
      }

      // Decide previewType
      if (contentType) {
        if (contentType.startsWith("image/")) setPreviewType("image");
        else if (contentType === "application/pdf") setPreviewType("pdf");
        else if (contentType.startsWith("text/csv") || contentType === "text/csv" || (artifact.filename && artifact.filename.endsWith(".csv"))) setPreviewType("csv");
        else if (contentType.startsWith("application/json") || (artifact.filename && artifact.filename.endsWith(".json"))) setPreviewType("json");
        else if (contentType.startsWith("text/") || contentType === "application/xml") setPreviewType("text");
        else if (contentType.startsWith("audio/")) setPreviewType("audio");
        else if (contentType.startsWith("video/")) setPreviewType("video");
        else setPreviewType("download");
      } else {
        setPreviewType("download");
      }

      // For CSV/JSON/TEXT, fetch to preview
      try {
        if (["csv", "json", "text"].includes(previewType)) {
          // attempt fetch
          const resp = await fetch(url);
          if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
          const txt = await resp.text();

          if (previewType === "csv") {
            const table = simpleCSVParse(txt, 50, 20);
            setPreviewData({ table });
          } else if (previewType === "json") {
            try {
              const obj = JSON.parse(txt);
              setPreviewData({ json: obj });
            } catch (e) {
              // fallback to text
              setPreviewType("text");
              setPreviewData({ text: txt.slice(0, 20000) });
            }
          } else { // text
            setPreviewData({ text: txt.slice(0, 20000) });
          }
        }
      } catch (e) {
        console.warn("Preview fetch failed or blocked by CORS:", e);
        // Fetch might fail due to CORS; fall back to download/iframe where possible
        // If it was CSV/text, fall back to "download" preview.
        if (["csv", "json", "text"].includes(previewType)) {
          setPreviewType("download");
          setPreviewData(null);
        } else {
          // leave other types (images/pdf/video) as-is; they may still work in <img> or <iframe>
        }
      } finally {
        setLoading(false);
      }
    }

    decide();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifact]);

  function renderBody() {
    if (!artifact) return <div>No artifact selected.</div>;
    const url = contentUrl;

    if (loading) return <div className="p-4">Preparing preview…</div>;
    if (error) return <div className="p-4 text-red-600">{error}</div>;

    switch (previewType) {
      case "image":
        return (
          <div className="flex justify-center">
            <img src={url} alt={artifact.filename || artifact.name} className="max-h-[70vh] object-contain" />
          </div>
        );

      case "pdf":
        return (
          <div className="h-[70vh]">
            <iframe title={artifact.filename || artifact.name} src={url} className="w-full h-full" />
          </div>
        );

      case "video":
        return (
          <div className="flex justify-center">
            <video controls className="max-h-[70vh] w-full">
              <source src={url} />
              Your browser does not support the video tag.
            </video>
          </div>
        );

      case "audio":
        return (
          <div>
            <audio controls>
              <source src={url} />
              Your browser does not support the audio element.
            </audio>
          </div>
        );

      case "csv":
        if (previewData?.table?.length) {
          const [headerRow, ...rest] = previewData.table;
          return (
            <div className="max-h-[60vh] overflow-auto">
              <table className="min-w-full border-collapse">
                <thead className="bg-slate-50 sticky top-0">
                  <tr>
                    {headerRow.map((h, i) => <th key={i} className="border px-2 py-1 text-left">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {rest.map((row, rIdx) => (
                    <tr key={rIdx} className={rIdx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      {row.map((cell, cIdx) => <td key={cIdx} className="border px-2 py-1">{cell}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return (
          <div className="p-4">
            Could not preview CSV (maybe CORS or file is large). <a href={url} target="_blank" rel="noreferrer" className="underline">Open in new tab</a>
          </div>
        );

      case "json":
        return (
          <div className="max-h-[60vh] overflow-auto">
            <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(previewData?.json, null, 2)}</pre>
          </div>
        );

      case "text":
        return (
          <div className="max-h-[60vh] overflow-auto">
            <pre className="whitespace-pre-wrap text-sm">{previewData?.text ?? "No preview available."}</pre>
          </div>
        );

      default:
        // download fallback
        return (
          <div className="p-4">
            <div className="mb-2">Unable to preview inline (CORS or unsupported type).</div>
            <a href={url} target="_blank" rel="noreferrer" className="underline">Open in new tab</a>
            <div className="mt-3 text-sm text-slate-500">Or right-click → Save link as to download.</div>
          </div>
        );
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative z-10 w-full max-w-5xl bg-white rounded shadow-lg overflow-hidden">
        <div className="flex items-center justify-between p-3 border-b">
          <div>
            <div className="font-medium">{artifact?.name ?? artifact?.filename}</div>
            <div className="text-xs text-slate-500">{artifact?.filename ?? artifact?.url}</div>
          </div>
          <div className="flex items-center gap-2">
            {artifact?.url && (
              <a
                href={artifact.url}
                target="_blank"
                rel="noreferrer"
                className="text-sm underline"
              >
                Open raw
              </a>
            )}
            <button onClick={onClose} className="px-3 py-1 rounded border">Close</button>
          </div>
        </div>

        <div className="p-4">
          {renderBody()}
        </div>
      </div>
    </div>
  );
}
