"use client";

import { useState, type FormEvent } from "react";
import { ApiError, api, type CompareResponse, type SessionTrace } from "@/api/client";
import {
  AnswerPane,
  ArrowRight,
  ErrorNotice,
  SeriesLabel,
  Waiting,
  WithTokens,
  useElapsed,
} from "./common";

const EXAMPLES = [
  "Which patient has vertigo?",
  "Who studies at IIT Madras?",
  "What is the phone number of the kabaddi player?",
  "Which patient was diagnosed at AIIMS Delhi?",
];

const STEPS = [
  [
    "Search",
    "The question is matched against every record, by keywords and by meaning, and the four closest are picked.",
  ],
  [
    "Build the prompt",
    "Standard RAG pastes in those records as written. Protected RAG pastes in the same records after Agents 1 to 3 have rewritten them.",
  ],
  [
    "Answer",
    "The same model answers from what it was given. Anything private it cannot see, it cannot repeat.",
  ],
] as const;

export function CompareView() {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [trace, setTrace] = useState<SessionTrace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const elapsed = useElapsed(busy);

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

  function newSession() {
    setSessionId(null);
    setTrace(null);
    setResult(null);
  }

  return (
    <>
      <div className="page-head">
        <h1>Ask both pipelines the same question</h1>
        <p>
          Both use the same search, model and prompt. The only difference is what they read:
          Standard RAG reads the records as they were written, Protected RAG reads the
          versions rewritten without identifying details.
        </p>
      </div>

      <ErrorNotice error={error} />

      <form className="field" onSubmit={submit}>
        <input
          aria-label="Question"
          placeholder="Ask about the records, for example: which patient has vertigo?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn" type="submit" disabled={busy || !query.trim()}>
          Compare
          <ArrowRight />
        </button>
      </form>
      <div className="examples">
        <span>Try</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="link"
            onClick={() => {
              setQuery(ex);
              run(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {!busy && !result && (
        <ol className="steps" aria-label="How a question is answered">
          {STEPS.map(([title, text], i) => (
            <li key={title}>
              <span className="step-no">{i + 1}</span>
              <b>{title}</b>
              <span>{text}</span>
            </li>
          ))}
        </ol>
      )}

      {(busy || result) && (
        <div className="pair">
          {(["baseline", "protected"] as const).map((side) => (
            <section className="pane" key={side}>
              <SeriesLabel series={side} />
              {result ? (
                <AnswerPane answer={result[side]} />
              ) : (
                <Waiting label="Waiting for the answer" seconds={elapsed} />
              )}
            </section>
          ))}
        </div>
      )}

      {trace && trace.rounds.length > 0 && (
        <section className="section">
          <div className="detail-head" style={{ borderBottom: 0, paddingBottom: 16 }}>
            <div>
              <h2 className="section-title" style={{ marginBottom: 4 }}>
                This session
              </h2>
              <p className="helper">
                Every question asked in session <span className="mono">{trace.session_id}</span>.
                An attacker gets to ask many questions in a row, so the session is what a
                privacy monitor would watch.
              </p>
            </div>
            <button type="button" className="btn btn-ghost" onClick={newSession}>
              Start a new session
            </button>
          </div>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Question</th>
                  <th>Standard RAG answer</th>
                  <th>Protected RAG answer</th>
                </tr>
              </thead>
              <tbody>
                {[...trace.rounds].reverse().map((r) => (
                  <tr key={r.round}>
                    <td className="num">{r.round}</td>
                    <td style={{ minWidth: 180 }}>{r.query}</td>
                    <td>
                      <span className="clamp">{r.baseline.answer}</span>
                    </td>
                    <td>
                      <span className="clamp">
                        <WithTokens text={r.protected.answer} />
                      </span>
                    </td>
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
