// ui/src/components/CompareChart.jsx
import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  BarChart,
  Bar,
  Cell,
} from "recharts";

/**
 * Props:
 * - series: [{ runId, runName, points: [{x, y, ts}], color? }, ...]
 * - xKeyLabel: string label for X axis (for time-series mode)
 * - lossCurves: optional, array of { runId, runName, rows: [{ epoch, train_loss, val_loss }] }
 */
export default function CompareChart({ series, xKeyLabel = "step", lossCurves = [] }) {
  if (!series || series.length === 0) {
    return (
      <div className="p-4 text-slate-500">No data to plot for selected runs.</div>
    );
  }

  const palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  // detect scalar mode
  const allScalar = series.every(s => Array.isArray(s.points) && s.points.length === 1);

  // ---------------- SCALAR BAR MODE ----------------
  if (allScalar) {
    const data = series.map((s, idx) => {
      const fullName = s.runName || s.runId || `Run ${idx + 1}`;
      const parts = fullName.split(" / ").map(p => p.trim()).filter(Boolean);

      let shortName;
      if (parts.length >= 2) {
        const task = parts[0];
        const model = parts[parts.length - 1];
        shortName = `${task} / ${model}`;
      } else {
        shortName = fullName;
      }

      return {
        fullName,
        shortName,
        value: s.points[0]?.y ?? null,
      };
    });

    return (
      <div>
        <div style={{ height: 380 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 20, left: 6, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="shortName" angle={-20} textAnchor="end" interval={0} height={60} />
              <YAxis />
              <Tooltip labelFormatter={(label, payload) => {
                const first = payload && payload[0];
                const full = first && first.payload && first.payload.fullName;
                return full || label;
              }} />
              <Bar dataKey="value">
                {data.map((entry, index) => (
                  <Cell key={entry.fullName || index} fill={palette[index % palette.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* If lossCurves provided, render the train/val loss comparison below */}
        {Array.isArray(lossCurves) && lossCurves.length > 0 && (
          <div style={{ height: 360, marginTop: 20 }}>
            <LossComparisonPlot lossCurves={lossCurves} palette={palette} />
          </div>
        )}
      </div>
    );
  }

  // ---------------- TIME-SERIES MODE ----------------
  // Build union of x values
  const xSet = new Set();
  series.forEach(s => s.points.forEach(p => xSet.add(p.x)));
  const xValues = Array.from(xSet).sort((a, b) => Number(a) - Number(b));

  const rows = xValues.map(x => {
    const row = { x };
    series.forEach(s => {
      const p = s.points.find(pt => String(pt.x) === String(x));
      row[s.runId] = p ? p.y : null;
    });
    return row;
  });

  return (
    <div>
      <div style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 10, right: 28, left: 6, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" label={{ value: xKeyLabel, position: "bottom", offset: 0 }} />
            <YAxis />
            <Tooltip />
            <Legend verticalAlign="top" />
            {series.map((s, idx) => (
              <Line
                key={s.runId}
                type="monotone"
                dataKey={s.runId}
                name={s.runName || s.runId}
                stroke={s.color || palette[idx % palette.length]}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* If lossCurves provided, render them too */}
      {Array.isArray(lossCurves) && lossCurves.length > 0 && (
        <div style={{ height: 360, marginTop: 20 }}>
          <LossComparisonPlot lossCurves={lossCurves} palette={palette} />
        </div>
      )}
    </div>
  );
}

/** Small helper component: render multiple train/val curves (one pair per run). */
function LossComparisonPlot({ lossCurves, palette }) {
  // Build union of epoch values
  const epochSet = new Set();
  lossCurves.forEach(lc => lc.rows.forEach(r => epochSet.add(Number(r.epoch))));
  const epochs = Array.from(epochSet).sort((a, b) => a - b);

  const rows = epochs.map(epoch => {
    const row = { epoch };
    lossCurves.forEach(lc => {
      const runKeyTrain = `${lc.runId}_train`;
      const runKeyVal = `${lc.runId}_val`;
      const found = lc.rows.find(rr => Number(rr.epoch) === Number(epoch));
      row[runKeyTrain] = found ? (Number(found.train_loss) || null) : null;
      row[runKeyVal] = found ? (Number(found.val_loss) || null) : null;
    });
    return row;
  });

  // build legend lines in consistent order (train then val for each run)
  const lines = [];
  lossCurves.forEach((lc, idx) => {
    const trainKey = `${lc.runId}_train`;
    const valKey = `${lc.runId}_val`;
    const color = lc.color || palette[idx % palette.length];
    lines.push({ key: trainKey, name: `${lc.runName} (train)`, color, dash: "solid" });
    lines.push({ key: valKey, name: `${lc.runName} (val)`, color, dash: "3 3" });
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 10, right: 28, left: 6, bottom: 30 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" label={{ value: "epoch", position: "bottom", offset: 0 }} />
        <YAxis />
        <Tooltip />
        <Legend verticalAlign="top" />
        {lines.map((L, i) => (
          <Line
            key={L.key}
            type="monotone"
            dataKey={L.key}
            name={L.name}
            stroke={L.color}
            strokeWidth={2}
            dot={false}
            strokeDasharray={L.dash}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
