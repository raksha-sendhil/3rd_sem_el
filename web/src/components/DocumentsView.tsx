"use client";

import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ApiError,
  api,
  type DocumentSummary,
  type DocumentView,
  type RoutingDecision,
} from "@/api/client";
import { ErrorNotice, Waiting, formatMs } from "./common";

const ACTION_LABEL: Record<string, string> = {
  placeholder: "Replaced with a token",
  generalise: "Made vague",
  keep: "Kept",
};

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The original text with every changed span marked at its offsets. */
function markOriginal(text: string, decisions: RoutingDecision[]): ReactNode[] {
  const spans = decisions
    .filter((d) => d.action !== "keep" && d.start != null && d.end != null)
    .sort((a, b) => a.start! - b.start!);
  const out: ReactNode[] = [];
  let at = 0;
  for (const d of spans) {
    if (d.start! < at) continue; // overlapping span, already covered
    out.push(<Fragment key={`t${at}`}>{text.slice(at, d.start!)}</Fragment>);
    out.push(
      <mark key={`m${d.start}`} data-action={d.action} title={`${d.category} becomes ${d.replacement}`}>
        {text.slice(d.start!, d.end!)}
      </mark>,
    );
    at = d.end!;
  }
  out.push(<Fragment key="end">{text.slice(at)}</Fragment>);
  return out;
}

/** The sanitised text with the tokens and vague phrases that replaced a detail marked. */
function markSanitised(text: string, decisions: RoutingDecision[]): ReactNode[] {
  const action = new Map<string, string>();
  for (const d of decisions) {
    if (d.action !== "keep" && d.replacement) action.set(d.replacement.toLowerCase(), d.action);
  }
  const phrases = [...action.keys()].sort((a, b) => b.length - a.length).map(escape);
  const pattern = new RegExp(`(\\[[A-Z]+_\\d+\\]${phrases.length ? "|" + phrases.join("|") : ""})`, "gi");
  return text.split(pattern).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} data-action={action.get(part.toLowerCase()) ?? "placeholder"}>
        {part}
      </mark>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
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
        setSelected((s) => s ?? list.find((d) => d.n_changes > 0)?.doc_id ?? list[0]?.doc_id ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setDoc(null);
    api
      .document(selected)
      .then(setDoc)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
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
      <div className="page-head">
        <h1>Every record, stored twice</h1>
        <p>
          The original goes to Standard RAG. Protected RAG gets a version rewritten from the
          facts in it, with names and numbers replaced by tokens and other telling details made
          vague.
        </p>
      </div>

      <ErrorNotice error={error} />

      <div className="split">
        <div className="list-scroll">
          <div className="field filter">
            <input
              aria-label="Filter records"
              placeholder="Filter by id or text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <div className="list" role="listbox" aria-label="Records">
            {docs === null && !error && <div className="empty">Loading records</div>}
            {docs && shown.length === 0 && <div className="empty">No record matches.</div>}
            {shown.map((d) => (
              <button
                key={d.doc_id}
                type="button"
                role="option"
                className="list-item"
                aria-selected={d.doc_id === selected}
                onClick={() => setSelected(d.doc_id)}
              >
                <span className="title">
                  <span className="mono">{d.doc_id}</span>
                  <span className="helper">
                    {d.n_changes === 0 ? "unchanged" : `${d.n_changes} changed`}
                  </span>
                </span>
                <span className="meta clamp" style={{ WebkitLineClamp: 2 }}>
                  {d.preview}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div>
          {selected && !doc && !error && <Waiting label="Loading the record" seconds={0} />}
          {doc && rec && (
            <>
              <div className="detail-head" style={{ display: "block" }}>
                <div className="eyebrow">Record</div>
                <h2 className="mono" style={{ fontSize: 20 }}>
                  {doc.doc_id}
                </h2>
                <div className="stats">
                  <span>
                    Details changed <b>{changed.length}</b>
                  </span>
                  <span>
                    Facts extracted <b>{rec.extractor.facts.length}</b>
                  </span>
                  <span>
                    Rewrites after a failed leak check <b>{rec.meta.leak_retries}</b>
                  </span>
                  <span>
                    Time to sanitise <b>{formatMs(rec.meta.total_ms)}</b>
                  </span>
                  <span>
                    Model <b>{rec.meta.model}</b>
                  </span>
                </div>
                <div className="doc-legend">
                  <span>
                    <span className="key" data-action="placeholder" aria-hidden />
                    {ACTION_LABEL.placeholder}
                  </span>
                  <span>
                    <span className="key" data-action="generalise" aria-hidden />
                    {ACTION_LABEL.generalise}
                  </span>
                </div>
              </div>

              {(rec.residual_leaks ?? []).length > 0 && (
                <div className="notice" style={{ marginTop: 24 }}>
                  <b>Leak check</b>
                  <span>
                    {(rec.residual_leaks ?? []).length} detail(s) could not be removed by rewriting and
                    were masked instead.
                  </span>
                </div>
              )}

              <div className="pair" style={{ marginTop: 0, borderTop: 0 }}>
                <section className="pane">
                  <div className="series-label">
                    <span className="swatch" data-series="baseline" aria-hidden />
                    Original
                  </div>
                  <div className="series-sub">what Standard RAG reads</div>
                  <p className="doc-text" style={{ marginTop: 16 }}>
                    {markOriginal(doc.original, decisions)}
                  </p>
                </section>
                <section className="pane">
                  <div className="series-label">
                    <span className="swatch" data-series="protected" aria-hidden />
                    Sanitised
                  </div>
                  <div className="series-sub">what Protected RAG reads</div>
                  <p className="doc-text" style={{ marginTop: 16 }}>
                    {markSanitised(rec.rewriter.safe_text, decisions)}
                  </p>
                </section>
              </div>

              <section className="section" style={{ marginTop: 24 }}>
                <h2 className="section-title">What was changed and why</h2>
                {decisions.length === 0 ? (
                  <p className="helper">Nothing in this record was found to be identifying.</p>
                ) : (
                  <div className="table-scroll">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Detail</th>
                          <th>Kind</th>
                          <th>Action</th>
                          <th>Became</th>
                          <th>Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {decisions.map((d, i) => (
                          <tr key={i}>
                            <td>{d.span_text}</td>
                            <td className="mono">{d.category}</td>
                            <td>{ACTION_LABEL[d.action] ?? d.action}</td>
                            <td>{d.replacement ?? "unchanged"}</td>
                            <td className="helper">{d.reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              <section className="section">
                <details className="disclosure">
                  <summary>Facts the rewrite was built from ({rec.extractor.facts.length})</summary>
                  <div className="table-scroll" style={{ marginTop: 12 }}>
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
                </details>
              </section>
            </>
          )}
        </div>
      </div>
    </>
  );
}
