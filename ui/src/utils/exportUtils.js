// ui/src/utils/exportUtils.js
export function downloadJSON(filename, obj) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  export function downloadCSV(filename, rows, headers) {
    // rows: array of arrays or array of objects; headers: array of string
    let csv = "";
    if (Array.isArray(rows) && rows.length > 0 && typeof rows[0] === "object" && !Array.isArray(rows[0])) {
      // rows are objects -> use headers order if provided, else use keys of first row
      const keys = headers && headers.length ? headers : Object.keys(rows[0]);
      csv += keys.join(",") + "\n";
      rows.forEach(r => {
        csv += keys.map(k => `"${(r[k] ?? "").toString().replace(/"/g, '""')}"`).join(",") + "\n";
      });
    } else {
      // rows are arrays
      if (headers && headers.length) csv += headers.join(",") + "\n";
      rows.forEach(r => {
        csv += r.map(c => `"${(c ?? "").toString().replace(/"/g, '""')}"`).join(",") + "\n";
      });
    }
  
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
  