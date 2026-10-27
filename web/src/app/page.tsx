"use client";

import { useEffect, useState } from "react";
import { API_URL, api } from "@/api/client";
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
  const [offline, setOffline] = useState(false);

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
        .then(() => alive && setOffline(false))
        .catch(() => alive && setOffline(true));
    check();
    const timer = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <>
      <header className="masthead">
        <span className="wordmark">sanitise</span>
        <nav className="tabs" aria-label="Views">
          {VIEWS.map((v) => (
            <a key={v.id} href={`#${v.id}`} aria-current={view === v.id ? "page" : undefined}>
              {v.label}
            </a>
          ))}
        </nav>
        {offline && (
          <span className="offline" title={API_URL}>
            API unreachable
          </span>
        )}
      </header>
      <main className="page">
        {view === "compare" && <CompareView />}
        {view === "attacks" && <AttacksView />}
        {view === "documents" && <DocumentsView />}
      </main>
    </>
  );
}
