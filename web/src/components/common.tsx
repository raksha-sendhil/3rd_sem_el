"use client";

import { Fragment, useEffect, useState, type ReactNode } from "react";
import { api, type RagAnswer } from "@/api/client";

export const SERIES = {
  baseline: { name: "Standard RAG", color: "#e8414f" },
  protected: { name: "Protected RAG", color: "#0078bf" },
} as const;
export type SeriesId = keyof typeof SERIES;
export const SIDES: SeriesId[] = ["baseline", "protected"];

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const TOKEN = /^\[[A-Z]+_\d+\]$/;

/** ``text`` with placeholder tokens set as chips and every ``leaks`` term marked in red. */
export function Marked({ text, leaks = [] }: { text: string; leaks?: string[] }) {
  const parts = [String.raw`\[[A-Z]+_\d+\]`, ...leaks.map(escape)];
  const pattern = new RegExp(`(${parts.join("|")})`, "gi");
  return (
    <>
      {text.split(pattern).map((part, i) => {
        if (i % 2 === 0) return <Fragment key={i}>{part}</Fragment>;
        if (TOKEN.test(part))
          return (
            <span key={i} className="token">
              {part}
            </span>
          );
        return (
          <mark key={i} className="leak">
            {part}
          </mark>
        );
      })}
    </>
  );
}

/** The personal details in the given records: every span the sanitiser removed or blurred.
 * Seeing one of them in Standard RAG's output means it leaked. */
export function useLeakTerms(docIds: string[]): string[] {
  const key = [...new Set(docIds)].sort().join(",");
  const [terms, setTerms] = useState<string[]>([]);
  useEffect(() => {
    let alive = true;
    const ids = key ? key.split(",") : [];
    Promise.all(ids.map((id) => api.document(id).catch(() => null))).then((views) => {
      if (!alive) return;
      const found = new Set<string>();
      for (const v of views)
        for (const d of v?.record.rewriter.decisions ?? [])
          if (d.action !== "keep" && d.span_text.trim().length > 2) found.add(d.span_text.trim());
      setTerms([...found].sort((a, b) => b.length - a.length));
    });
    return () => {
      alive = false;
    };
  }, [key]);
  return terms;
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** Seconds since ``running`` became true. */
export function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    setSeconds(0);
    const start = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 500);
    return () => clearInterval(timer);
  }, [running]);
  return seconds;
}

export function Working({ seconds }: { seconds: number }) {
  return (
    <div className="working" role="status" aria-label="Working">
      <span className="dots" aria-hidden>
        <i />
        <i />
        <i />
      </span>
      {seconds > 0 && <span>{seconds}s</span>}
    </div>
  );
}

export function Notice({ error }: { error: string | null }) {
  return error ? (
    <div className="notice" role="alert">
      {error}
    </div>
  ) : null;
}

export function ArrowRight() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden>
      <path d="M3 9h11.2l-4-4L11.6 3.6 18 10l-6.4 6.4-1.4-1.4 4-4H3z" fill="currentColor" />
    </svg>
  );
}

/** One pipeline's answer, the records it used, and the exact text sent to the model. */
export function AnswerPanel({
  side,
  answer,
  leaks,
  children,
}: {
  side: SeriesId;
  answer: RagAnswer | null;
  leaks: string[];
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className="panel" data-series={side}>
      <div className="panel-head">
        <span className="label">{SERIES[side].name}</span>
        {answer && <span className="meta">{formatMs(answer.latency_ms)}</span>}
      </div>
      <div className="panel-body">
        {answer ? (
          <p className="answer">
            <Marked text={answer.answer} leaks={side === "baseline" ? leaks : []} />
          </p>
        ) : (
          children
        )}
      </div>
      {answer && (
        <>
          <div className="panel-foot">
            {answer.doc_ids.map((id) => (
              <span className="doc-chip" key={id}>
                {id}
              </span>
            ))}
            {side === "protected" && (
              <span
                className="verdict"
                data-verdict={answer.monitor?.verdict}
                title={answer.monitor?.reason}
              >
                {answer.monitor?.verdict ?? "allowed"}
              </span>
            )}
            <button
              type="button"
              className="link-btn payload-toggle"
              aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
            >
              Payload
            </button>
          </div>
          {open && (
            <pre className="payload">
              <Marked text={answer.payload} leaks={side === "baseline" ? leaks : []} />
            </pre>
          )}
        </>
      )}
    </section>
  );
}
