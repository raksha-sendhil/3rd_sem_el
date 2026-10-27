"use client";

import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ApiError,
  api,
  type DocumentSummary,
  type DocumentView,
  type RoutingDecision,
} from "@/api/client";
import { Notice, Working, formatMs } from "./common";

const ACTION: Record<string, string> = {
  placeholder: "Token",
  generalise: "Vague",
  keep: "Kept",
};

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The original with every changed span marked at its offsets. */
function markOriginal(text: string, decisions: RoutingDecision[]): ReactNode[] {
  const spans = decisions
    .filter((d) => d.action !== "keep" && d.start != null && d.end != null)
    .sort((a, b) => a.start! - b.start!);
  const out: ReactNode[] = [];
  let at = 0;
  for (const d of spans) {
    if (d.start! < at) continue; // overlaps a span already marked
    out.push(<Fragment key={`t${at}`}>{text.slice(at, d.start!)}</Fragment>);
    out.push(
      <mark key={`m${d.start}`} data-action={d.action} title={`${d.category} → ${d.replacement}`}>
        {text.slice(d.start!, d.end!)}
      </mark>,
    );
    at = d.end!;
  }
  out.push(<Fragment key="end">{text.slice(at)}</Fragment>);
  return out;
}

/** The sanitised text with its tokens and vague phrases marked. */
function markSanitised(text: string, decisions: RoutingDecision[]): ReactNode[] {
  const vague = [
    ...new Set(
      decisions
        .filter((d) => d.action === "generalise" && d.replacement)
        .map((d) => d.replacement!.toLowerCase()),
    ),
  ].sort((a, b) => b.length - a.length);
  const parts = [String.raw`\[[A-Z]+_\d+\]`, ...vague.map(escape)];
  return text.split(new RegExp(`(${parts.join("|")})`, "gi")).map((part, i) => {
    if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
    return part.startsWith("[") ? (
      <span key={i} className="token">
        {part}
      </span>
    ) : (
      <mark key={i} data-action="generalise">
        {part}
      </mark>
    );
  });
}

export function DocumentsView() {
  const [docs, setDocs] = useState<DocumentSummary[] | null>(null);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [doc, setDoc] = useState<DocumentView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .documents()
      .then((list) => {
        setDocs(list);
        setSelected(
          (s) => s ?? list.find((d) => d.n_changes > 0)?.doc_id ?? list[0]?.doc_id ?? null,
        );
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    let alive = true;
    setDoc(null);
    api
      .document(selected)
      .then((d) => alive && setDoc(d))
      .catch((e) => alive && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [selected]);

  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (docs ?? []).filter(
      (d) => !f || d.doc_id.includes(f) || d.preview.toLowerCase().includes(f),
    );
  }, [docs, filter]);

  const rec = doc?.record;
  const decisions = rec?.rewriter.decisions ?? [];
  const changed = decisions.filter((d) => d.action !== "keep");

  return (
    <>
      <Notice error={error} />
      <div className="library">
        <div className="doc-list">
          <input
            aria-label="Filter records"
            placeholder="Filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <div className="doc-items" role="listbox" aria-label="Records">
            {docs && shown.length === 0 && <div className="empty" style={{ padding: 14 }}>No match</div>}
            {shown.map((d) => (
              <button
                key={d.doc_id}
                type="button"
                role="option"
                className="doc-item"
                aria-selected={d.doc_id === selected}
                onClick={() => setSelected(d.doc_id)}
              >
                <span className="top">
                  <span className="mono">{d.doc_id}</span>
                  <span className="n">{d.n_changes}</span>
                </span>
                <span className="preview">{d.preview}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          {selected && !doc && !error && <Working seconds={0} />}
          {doc && rec && (
            <>
              <div className="doc-head">
                <h2>{doc.doc_id}</h2>
                <span className="stat">
                  <b>{changed.length}</b>changed
                </span>
                <span className="stat">
                  <b>{rec.extractor.facts.length}</b>facts
                </span>
                <span className="stat">
                  <b>{formatMs(rec.meta.total_ms)}</b>
                </span>
                <span className="keys">
                  <span className="key">
                    <span className="token">[TOKEN_1]</span>removed
                  </span>
                  <span className="key">
                    <mark data-action="generalise">&nbsp;&nbsp;&nbsp;&nbsp;</mark>made vague
                  </span>
                </span>
              </div>

              {(rec.residual_leaks ?? []).length > 0 && (
                <Notice
                  error={`${(rec.residual_leaks ?? []).length} detail(s) survived rewriting and were masked.`}
                />
              )}

              <div className="duo">
                <section className="panel" data-series="baseline">
                  <div className="panel-head">
                    <span className="label">Original</span>
                  </div>
                  <div className="panel-body">
                    <p className="doc-text">{markOriginal(doc.original, decisions)}</p>
                  </div>
                </section>
                <section className="panel" data-series="protected">
                  <div className="panel-head">
                    <span className="label">Sanitised</span>
                  </div>
                  <div className="panel-body">
                    <p className="doc-text">{markSanitised(rec.rewriter.safe_text, decisions)}</p>
                  </div>
                </section>
              </div>

              {decisions.length > 0 && (
                <section className="section" style={{ marginTop: 40 }}>
                  <h2 className="label" style={{ marginBottom: 14 }}>
                    Changes
                  </h2>
                  <div className="table-wrap">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Detail</th>
                          <th>Kind</th>
                          <th>Action</th>
                          <th>Became</th>
                          <th>Why</th>
                        </tr>
                      </thead>
                      <tbody>
                        {decisions.map((d, i) => (
                          <tr key={i}>
                            <td>{d.span_text}</td>
                            <td className="mono">{d.category}</td>
                            <td>{ACTION[d.action] ?? d.action}</td>
                            <td>
                              {d.replacement?.startsWith("[") ? (
                                <span className="token">{d.replacement}</span>
                              ) : (
                                (d.replacement ?? "")
                              )}
                            </td>
                            <td style={{ color: "var(--ink-2)" }}>{d.reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {rec.extractor.facts.length > 0 && (
                <section className="section" style={{ marginTop: 40 }}>
                  <h2 className="label" style={{ marginBottom: 14 }}>
                    Facts
                  </h2>
                  <div className="table-wrap">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>About</th>
                          <th>Relation</th>
                          <th>Value</th>
                          <th>Importance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rec.extractor.facts.map((f, i) => (
                          <tr key={i}>
                            <td className="mono">{f.subject}</td>
                            <td>{f.relation.replaceAll("_", " ")}</td>
                            <td>{f.value}</td>
                            <td>{f.importance}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
