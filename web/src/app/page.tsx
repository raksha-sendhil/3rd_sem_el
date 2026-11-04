"use client";

import { useEffect, useState } from "react";
import { API_URL, type DomainInfo, api, getDomain, setDomain } from "@/api/client";
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
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [domain, setDomainState] = useState<string | null>(null);

  useEffect(() => {
    api
      .domains()
      .then((list) => {
        setDomains(list.domains);
        const saved = getDomain();
        const chosen = list.domains.some((d) => d.name === saved) ? saved! : list.default;
        setDomain(chosen);
        setDomainState(chosen);
      })
      .catch(() => setDomainState(getDomain() ?? ""));
  }, []);

  const chooseDomain = (name: string) => {
    setDomain(name);
    setDomainState(name); // the views remount and reload for the new domain
  };
  const info = domains.find((d) => d.name === domain);

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
      <header className="top">
        <h1 className="title" aria-label="sanitise">
          {[..."sanitise"].map((ch, i) => (
            <span key={i} aria-hidden>
              {ch}
            </span>
          ))}
        </h1>
        {offline && (
          <span className="offline" title={API_URL}>
            API unreachable
          </span>
        )}
        {domains.length > 0 && (
          <label
            className="domain"
            title={
              info
                ? `Agents: ${info.model} | Answers: ${info.answer_model} (${info.answer_backend}) | Agent 1: ${info.agent1}`
                : undefined
            }
          >
            Domain{" "}
            <select value={domain ?? ""} onChange={(e) => chooseDomain(e.target.value)}>
              {domains.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <nav className="tabs" aria-label="Views">
          {VIEWS.map((v) => (
            <a
              key={v.id}
              className="tab"
              href={`#${v.id}`}
              aria-current={view === v.id ? "page" : undefined}
            >
              {v.label}
            </a>
          ))}
        </nav>
      </header>
      {domain !== null && (
        <main className="page" key={domain}>
          {view === "compare" && <CompareView />}
          {view === "attacks" && <AttacksView />}
          {view === "documents" && <DocumentsView />}
        </main>
      )}
    </>
  );
}
