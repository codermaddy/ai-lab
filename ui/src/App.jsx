import React from "react";
import { Routes, Route, NavLink, Link } from "react-router-dom";
import RunsList from "./pages/RunsList";
import RunDetail from "./pages/RunDetail";
import CompareRuns from "./pages/CompareRuns";
import Chatbox from "./components/Chatbox";
import AssistantPage from "./pages/AssistantPage";

function Home() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-2">AI-LAB</h1>
      <p className="mb-4 text-slate-600">
        AI-LAB is a small experiment hub for ML workflows. Every training run
          is logged to a central database with its hyperparameters, metrics,
          artifacts (checkpoints, CSVs, plots), and metadata.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="p-4 border rounded-lg bg-white shadow-sm">
          <h3 className="font-semibold">Assistant</h3>
          <p className="text-sm text-slate-500 mb-3">Open the assistant to ask questions, run comparisons, or request actions.</p>
          <Link to="/assistant" className="inline-block px-3 py-2 rounded bg-slate-800 text-white hover:opacity-95">
            Open Assistant
          </Link>
        </div>
      </div>
    </div>
  );
}

function Layout({ children }) {
  const activeClass =
    "block p-2 rounded bg-slate-100 font-medium text-slate-900";
  const baseClass = "block p-2 rounded hover:bg-slate-100";

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-white border-r p-4">
        <div className="mb-6">
          <NavLink to="/" className="text-lg font-bold">AI-LAB</NavLink>
        </div>

        <nav className="space-y-2 text-sm">
          <NavLink to="/" end className={({ isActive }) => (isActive ? activeClass : baseClass)}>
            Dashboard
          </NavLink>

          <NavLink to="/runs" className={({ isActive }) => (isActive ? activeClass : baseClass)}>
            Runs
          </NavLink>

          <NavLink to="/compare" className={({ isActive }) => (isActive ? activeClass : baseClass)}>
            Compare
          </NavLink>

          <NavLink to="/assistant" className={({ isActive }) => (isActive ? activeClass : baseClass)}>
            Assistant
          </NavLink>
        </nav>
      </aside>

      <main className="flex-1 bg-slate-50 p-6">
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/runs" element={<RunsList />} />
        <Route path="/runs/:id" element={<RunDetail />} />
        <Route path="/assistant" element={<AssistantPage />} />
        <Route path="/compare" element={<CompareRuns />} />
        {/* fallback */}
        <Route path="*" element={<div className="p-6">Page not found</div>} />
      </Routes>
    </Layout>
  );
}
