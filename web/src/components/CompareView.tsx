"use client";

import { useState, type FormEvent } from "react";
import { ApiError, api, type CompareResponse, type SessionTrace } from "@/api/client";
import { AnswerWin, Marked, Notice, PixelArrow, SIDES, Win, useElapsed, useLeakTerms } from "./common";

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
    <div className="stack">
      <Notice error={error} />

      <div>
        <form className="ask" onSubmit={submit}>
          <label className="ask-field">
            <span className="prompt-char" aria-hidden>
              &gt;
            </span>
            <input
              aria-label="Question"
              placeholder="ask the records"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <button className="btn" type="submit" disabled={busy || !query.trim()}>
            Ask
            <PixelArrow />
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
      </div>

      <div className="duo" style={{ marginTop: 36 }}>
        {SIDES.map((side) => (
          <AnswerWin
            key={side}
            side={side}
            answer={result?.[side] ?? null}
            leaks={leaks}
            busy={busy}
            seconds={elapsed}
          />
        ))}
      </div>

      {trace && trace.rounds.length > 1 && (
        <Win
          title="Session"
          tone="purple"
          end={
            <button
              type="button"
              className="mini"
              onClick={() => {
                setSessionId(null);
                setTrace(null);
                setResult(null);
              }}
            >
              New
            </button>
          }
        >
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
                    <td className="num px">{r.round}</td>
                    <td style={{ minWidth: 160, fontWeight: 600 }}>{r.query}</td>
                    <td>
                      <span className="clamp">
                        <Marked text={r.baseline.answer} leaks={traceLeaks} />
                      </span>
                    </td>
                    <td>
                      <span className="clamp">
                        <Marked text={r.protected.answer} leaks={traceLeaks} />
                      </span>
                    </td>
                    <td>
                      <span className="tag" data-tone="lime">
                        {r.protected.monitor?.verdict ?? "allowed"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Win>
      )}
    </div>
  );
}
