"use client";

import { Fragment, useEffect, useState, type ReactNode } from "react";
import type { RagAnswer } from "@/api/client";

export const SERIES = {
  baseline: { name: "Standard RAG", sub: "reads the original records", color: "#6929c4" },
  protected: { name: "Protected RAG", sub: "reads the sanitised records", color: "#1192e8" },
} as const;
export type SeriesId = keyof typeof SERIES;

const TOKEN = /(\[[A-Z]+_\d+\])/g;

/** Text with placeholder tokens such as [PERSON_1] set apart. */
export function WithTokens({ text }: { text: string }) {
  return (
    <>
      {text.split(TOKEN).map((part, i) =>
        i % 2 === 1 ? (
          <span key={i} className="token">
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

export function SeriesLabel({ series, compact }: { series: SeriesId; compact?: boolean }) {
  return (
    <div>
      <div className="series-label">
        <span className="swatch" data-series={series} aria-hidden />
        {SERIES[series].name}
      </div>
      {!compact && <div className="series-sub">{SERIES[series].sub}</div>}
    </div>
  );
}

export function ArrowRight() {
  return (
    <svg className="arrow" viewBox="0 0 16 16" aria-hidden>
      <path d="M9.3 3.3 8.6 4l3.5 3.5H2v1h10.1L8.6 12l.7.7L14 8z" fill="currentColor" />
    </svg>
  );
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** Seconds since `running` became true; resets when it turns false. */
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

export function Waiting({ label, seconds }: { label: string; seconds: number }) {
  return (
    <div className="waiting" role="status">
      {label}
      {seconds > 0 && `, ${seconds} s`}
      <div className="progress" />
    </div>
  );
}

export function ErrorNotice({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="notice" role="alert">
      <b>Request failed</b>
      <span>{error}</span>
    </div>
  );
}

/** One pipeline's answer with its retrieval facts and the literal model input. */
export function AnswerPane({ answer }: { answer: RagAnswer }) {
  const monitor = answer.monitor;
  const rows: [string, ReactNode][] = [
    [
      "Records used",
      answer.doc_ids.length ? (
        <span className="mono">{answer.doc_ids.join("  ")}</span>
      ) : (
        "none"
      ),
    ],
    ["Time", formatMs(answer.latency_ms)],
    [
      "Session monitor",
      monitor
        ? `${monitor.verdict}${monitor.reason ? `, ${monitor.reason}` : ""}`
        : "not installed",
    ],
  ];
  return (
    <>
      <p className="answer">
        <WithTokens text={answer.answer} />
      </p>
      <dl className="facts">
        {rows.map(([k, v]) => (
          <Fragment key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </Fragment>
        ))}
      </dl>
      <details className="disclosure">
        <summary>What the model was given ({answer.payload.length} characters)</summary>
        <pre className="payload">
          <WithTokens text={answer.payload} />
        </pre>
      </details>
    </>
  );
}
