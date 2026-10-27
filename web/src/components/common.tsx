"use client";

import { Fragment, useEffect, useState, type ReactNode } from "react";
import { api, type RagAnswer } from "@/api/client";

export const SERIES = {
  baseline: { name: "Standard RAG", color: "#f2412f" },
  protected: { name: "Protected RAG", color: "#2f6bff" },
} as const;
export type SeriesId = keyof typeof SERIES;
export const SIDES: SeriesId[] = ["baseline", "protected"];

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const TOKEN = /^\[[A-Z]+_\d+\]$/;

/** A pixel window: coloured title bar, three window squares, body. */
export function Win({
  title,
  series,
  tone,
  end,
  children,
  className = "",
}: {
  title: ReactNode;
  series?: SeriesId;
  tone?: string;
  end?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`win ${className}`} data-series={series} data-tone={tone}>
      <div className="win-bar">
        <span className="win-title">{title}</span>
        <span className="win-end">
          {end}
          <span className="win-dots" aria-hidden>
            <i />
            <i />
            <i />
          </span>
        </span>
      </div>
      {children}
    </section>
  );
}

/** ``text`` with placeholder tokens as chips and every ``leaks`` term marked. */
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

/** How many distinct ``leaks`` terms occur in ``text``. */
export function countLeaks(text: string, leaks: string[]): number {
  const lower = text.toLowerCase();
  return leaks.filter((t) => lower.includes(t.toLowerCase())).length;
}

/** The personal details in the given records: every span the sanitiser removed or blurred.
 * Any of them appearing in an answer means it was exposed. */
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

export function Loader({ seconds = 0 }: { seconds?: number }) {
  return (
    <span className="loader" role="status" aria-label="Working">
      <i />
      <i />
      <i />
      <i />
      <i />
      {seconds > 0 && <small>{seconds}s</small>}
    </span>
  );
}

export function Notice({ error }: { error: string | null }) {
  return error ? (
    <div className="notice" role="alert">
      {error}
    </div>
  ) : null;
}

/** A pixel arrow drawn on a 7x7 grid. */
export function PixelArrow() {
  const cells = [
    [3, 0],
    [4, 1],
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
    [4, 3],
    [5, 3],
    [6, 3],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    [4, 4],
    [5, 4],
    [4, 5],
    [3, 6],
  ];
  return (
    <svg viewBox="0 0 7 7" shapeRendering="crispEdges" aria-hidden>
      {cells.map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="currentColor" />
      ))}
    </svg>
  );
}

/** One pipeline's answer: what it exposed, the records used, and the raw payload. */
export function AnswerWin({
  side,
  answer,
  leaks,
  busy,
  seconds,
}: {
  side: SeriesId;
  answer: RagAnswer | null;
  leaks: string[];
  busy: boolean;
  seconds: number;
}) {
  const [open, setOpen] = useState(false);
  const exposed = answer ? countLeaks(answer.answer, leaks) : null;
  return (
    <Win
      title={SERIES[side].name}
      series={side}
      end={
        exposed !== null && (
          <span className="tag" data-tone={exposed ? "yellow" : "lime"}>
            {exposed} exposed
          </span>
        )
      }
    >
      <div className="win-body">
        {answer ? (
          <p className="answer">
            <Marked text={answer.answer} leaks={leaks} />
          </p>
        ) : (
          <div className="empty-answer">
            {busy ? <Loader seconds={seconds} /> : <span className="cursor" aria-hidden />}
          </div>
        )}
      </div>
      {answer && (
        <>
          <div className="win-foot">
            {answer.doc_ids.map((id) => (
              <span className="doc-id" key={id}>
                {id}
              </span>
            ))}
            <span className="foot-end">
              {side === "protected" && (
                <span className="tag" data-tone="lime" title={answer.monitor?.reason}>
                  {answer.monitor?.verdict ?? "allowed"}
                </span>
              )}
              {formatMs(answer.latency_ms)}
              <button
                type="button"
                className="mini"
                aria-expanded={open}
                onClick={() => setOpen((o) => !o)}
              >
                Payload
              </button>
            </span>
          </div>
          {open && (
            <pre className="payload">
              <Marked text={answer.payload} leaks={leaks} />
            </pre>
          )}
        </>
      )}
    </Win>
  );
}
