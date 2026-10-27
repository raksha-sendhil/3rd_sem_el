"use client";

import { useEffect, useState } from "react";
import { API_URL, api, type Health } from "@/api/client";
import { AttacksView } from "@/components/AttacksView";
import { CompareView } from "@/components/CompareView";
import { DocumentsView } from "@/components/DocumentsView";

const VIEWS = [
  { id: "compare", label: "Compare" },
  { id: "attacks", label: "Attacks" },
  { id: "documents", label: "Documents" },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

function viewFromHash(): ViewId {
  const hash = window.location.hash.slice(1);
  return VIEWS.some((v) => v.id === hash) ? (hash as ViewId) : "compare";
}

export default function Home() {
  const [view, setView] = useState<ViewId>("compare");
  const [health, setHealth] = useState<Health | null | "down">(null);

  useEffect(() => {
    setView(viewFromHash());
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth("down"));
    check();
    const timer = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const host = API_URL.replace(/^https?:\/\//, "");
  let status: { state: string; text: string };
  if (health === null) status = { state: "wait", text: "Connecting" };
  else if (health === "down") status = { state: "down", text: "Not reachable" };
  else status = { state: "ok", text: `Connected, model ${health.model}` };

  return (
    <>
      <header className="shell-header">
        <div className="brand">
          sanitise<b>Dashboard</b>
        </div>
        <nav className="nav" aria-label="Views">
          {VIEWS.map((v) => (
            <a key={v.id} href={`#${v.id}`} aria-current={view === v.id ? "page" : undefined}>
              {v.label}
            </a>
          ))}
        </nav>
        <div className="api-status" title={API_URL}>
          <span className="status-square" data-state={status.state} aria-hidden />
          <span>{status.text}</span>
          <span className="api-url mono">{host}</span>
        </div>
      </header>
      <main className="page">
        {health === "down" && (
          <div className="notice" role="alert">
            <b>Cannot reach the API</b>
            <span>
              Start it with <code>sanitise serve</code>, or set <code>NEXT_PUBLIC_API_URL</code>{" "}
              in <code>web/.env.local</code> to the address printed by the Colab notebook.
            </span>
          </div>
        )}
        {view === "compare" && <CompareView />}
        {view === "attacks" && <AttacksView />}
        {view === "documents" && <DocumentsView />}
      </main>
    </>
  );
}
