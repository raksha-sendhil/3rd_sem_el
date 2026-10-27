"use client";

import { useState, type FormEvent } from "react";
import { Poster } from "./Poster";
import { ApiError, api, type CompareResponse, type SessionTrace } from "@/api/client";
import {
  AnswerPanel,
  ArrowRight,
  Marked,
  Notice,
  SIDES,
  Working,
  useElapsed,
  useLeakTerms,
} from "./common";

const EXAMPLES = [
  "Which patient has vertigo?",
  "Who studies at IIT Madras?",
  "What is the kabaddi player's phone number?",
  "Who was diagnosed at AIIMS Delhi?",
];

export function CompareView() {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [trace, setTrace] = useState<SessionTrace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const elapsed = useElapsed(busy);
  const leaks = useLeakTerms(result?.baseline.doc_ids ?? []);
  const traceLeaks = useLeakTerms(trace?.rounds.flatMap((r) => r.baseline.doc_ids) ?? []);

  async function run(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const sid = sessionId ?? (await api.createSession()).session_id;
      setSessionId(sid);
      setResult(await api.compare(q.trim(), sid));
      setTrace(await api.session(sid));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    run(query);
  }

  return (
    <>
      <Notice error={error} />

      <form className="ask" onSubmit={submit}>
        <input
          aria-label="Question"
          placeholder="Ask the records…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn" type="submit" disabled={busy || !query.trim()}>
          Ask
          <ArrowRight />
        </button>
      </form>
      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => {
              setQuery(ex);
              run(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {!busy && !result && !trace && <Poster />}

      {(busy || result) && (
        <div className="duo section" style={{ marginTop: 40 }}>
          {SIDES.map((side) => (
            <AnswerPanel key={side} side={side} answer={result?.[side] ?? null} leaks={leaks}>
              <Working seconds={elapsed} />
            </AnswerPanel>
          ))}
        </div>
      )}

      {trace && trace.rounds.length > 1 && (
        <section className="section">
          <div className="section-head">
            <h2 className="label">Session</h2>
            <button
              type="button"
              className="link-btn"
              onClick={() => {
                setSessionId(null);
                setTrace(null);
                setResult(null);
              }}
            >
              New session
            </button>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Question</th>
                  <th>Standard RAG</th>
                  <th>Protected RAG</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {[...trace.rounds].reverse().map((r) => (
                  <tr key={r.round}>
                    <td className="num">{r.round}</td>
                    <td style={{ minWidth: 160 }}>{r.query}</td>
                    <td>
                      <span className="clamp">
                        <Marked text={r.baseline.answer} leaks={traceLeaks} />
                      </span>
                    </td>
                    <td>
                      <span className="clamp">
                        <Marked text={r.protected.answer} />
                      </span>
                    </td>
                    <td>{r.protected.monitor?.verdict ?? "allowed"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
