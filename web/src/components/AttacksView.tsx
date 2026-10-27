"use client";

import { useEffect, useState } from "react";
import {
  ApiError,
  api,
  type AttackDefinition,
  type AttackRun,
  type RoundLeakage,
} from "@/api/client";
import {
  ArrowRight,
  ErrorNotice,
  SERIES,
  SeriesLabel,
  Waiting,
  WithTokens,
  useElapsed,
  type SeriesId,
} from "./common";
import { LeakageChart } from "./LeakageChart";

const KIND: Record<string, string> = {
  direct: "Asks outright for the person's name and contact details.",
  linkage: "Combines a few ordinary details that together point to one person.",
  stepwise: "Asks a series of small questions, each one adding a detail.",
};

const SIDES: SeriesId[] = ["baseline", "protected"];
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const stillMatch = (n: number) => (n === 1 ? "1 person still matches" : `${n} people still match`);

function leakage(run: AttackRun, side: SeriesId, i: number): RoundLeakage | null {
  const r = run.rounds[i];
  return (side === "baseline" ? r.baseline_leakage : r.protected_leakage) ?? null;
}

export function AttacksView() {
  const [attacks, setAttacks] = useState<AttackDefinition[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [run, setRun] = useState<AttackRun | null>(null);
  const [shown, setShown] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asTable, setAsTable] = useState(false);
  const elapsed = useElapsed(busy);

  useEffect(() => {
    api
      .attacks()
      .then((list) => {
        setAttacks(list);
        setSelected((s) => s ?? list[0]?.attack_id ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  // play the rounds back one per second, so the audience can follow them
  useEffect(() => {
    if (!run || shown >= run.rounds.length) return;
    const timer = setTimeout(() => setShown((n) => n + 1), shown === 0 ? 150 : 1000);
    return () => clearTimeout(timer);
  }, [run, shown]);

  const attack = attacks?.find((a) => a.attack_id === selected) ?? null;

  function select(id: string) {
    if (busy) return;
    setSelected(id);
    setRun(null);
    setShown(0);
    setError(null);
  }

  async function start() {
    if (!attack) return;
    setBusy(true);
    setError(null);
    setRun(null);
    setShown(0);
    try {
      setRun(await api.runAttack(attack.attack_id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const visible = run ? run.rounds.slice(0, shown) : [];
  const series = (pick: (l: RoundLeakage) => number) =>
    Object.fromEntries(
      SIDES.map((s) => [s, visible.map((_, i) => pick(leakage(run!, s, i)!))]),
    ) as Record<SeriesId, number[]>;
  const hasLeakage = run !== null && run.rounds.every((r) => r.baseline_leakage);
  const maxDisclosed = hasLeakage
    ? Math.max(3, ...run.rounds.flatMap((_, i) => SIDES.map((s) => leakage(run, s, i)!.disclosed.length)))
    : 0;
  const maxPool = hasLeakage
    ? Math.max(1, ...run.rounds.flatMap((_, i) => SIDES.map((s) => leakage(run, s, i)!.candidate_pool)))
    : 0;
  const done = run !== null && shown >= run.rounds.length;

  return (
    <>
      <div className="page-head">
        <h1>Try to identify a person</h1>
        <p>
          Each attack is a scripted line of questioning aimed at one person in the records. It
          runs against both pipelines. An attack succeeds when the answers narrow everyone
          down to that one person.
        </p>
      </div>

      <ErrorNotice error={error} />

      <div className="split">
        <div className="list" role="listbox" aria-label="Attacks">
          {attacks === null && !error && <div className="empty">Loading attacks</div>}
          {attacks?.map((a) => (
            <button
              key={a.attack_id}
              type="button"
              role="option"
              className="list-item"
              aria-selected={a.attack_id === selected}
              onClick={() => select(a.attack_id)}
            >
              <span className="title">
                {a.name}
                <span className="helper">{plural(a.queries.length, "question")}</span>
              </span>
              <span className="meta" style={{ display: "block" }}>
                {KIND[a.kind] ?? a.kind}
              </span>
            </button>
          ))}
        </div>

        {attack && (
          <div>
            <div className="detail-head">
              <div>
                <div className="eyebrow">
                  {attack.kind} attack, target person{" "}
                  <span className="mono">{attack.target_person_id}</span>
                </div>
                <h2>{attack.name}</h2>
                <ol className="questions">
                  {attack.queries.map((q) => (
                    <li key={q}>{q}</li>
                  ))}
                </ol>
              </div>
              <button type="button" className="btn" onClick={start} disabled={busy}>
                {run ? "Run again" : "Run attack"}
                <ArrowRight />
              </button>
            </div>

            {busy && (
              <Waiting
                label={`Asking ${plural(attack.queries.length, "question")} of both pipelines`}
                seconds={elapsed}
              />
            )}

            {run && (
              <>
                <div className="tiles">
                  {SIDES.map((s) => {
                    const at =
                      s === "baseline" ? run.baseline_identified_at : run.protected_identified_at;
                    const revealed = at !== null && at !== undefined && at <= shown;
                    const lastLeak = shown > 0 ? leakage(run, s, shown - 1) : null;
                    return (
                      <div className="tile" key={s}>
                        <SeriesLabel series={s} />
                        <div className="tile-value">
                          {revealed ? `Identified at question ${at}` : done ? "Not identified" : "..."}
                        </div>
                        <div className="tile-note">
                          {lastLeak
                            ? `${stillMatch(lastLeak.candidate_pool)} after question ${shown}`
                            : " "}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {hasLeakage && (
                  <section className="section" style={{ marginTop: 40 }}>
                    <div className="detail-head" style={{ borderBottom: 0, paddingBottom: 0 }}>
                      <div className="legend" aria-label="Legend">
                        {SIDES.map((s) => (
                          <span key={s}>
                            <span className="swatch" data-series={s} aria-hidden />
                            {SERIES[s].name}
                          </span>
                        ))}
                      </div>
                      <div className="switcher" role="group" aria-label="Show as">
                        <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
                          Chart
                        </button>
                        <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
                          Table
                        </button>
                      </div>
                    </div>
                    {asTable ? (
                      <div className="table-scroll" style={{ marginTop: 16 }}>
                        <table className="table">
                          <thead>
                            <tr>
                              <th rowSpan={2}>Question</th>
                              {SIDES.map((s) => (
                                <th colSpan={2} key={s} className="group">
                                  {SERIES[s].name}
                                </th>
                              ))}
                            </tr>
                            <tr>
                              {SIDES.flatMap((s) => [
                                <th className="num" key={s + "d"}>
                                  Details revealed
                                </th>,
                                <th className="num" key={s + "p"}>
                                  People matching
                                </th>,
                              ])}
                            </tr>
                          </thead>
                          <tbody>
                            {visible.map((r, i) => (
                              <tr key={r.round}>
                                <td>Q{r.round}</td>
                                {SIDES.flatMap((s) => {
                                  const l = leakage(run, s, i)!;
                                  return [
                                    <td className="num" key={s + "d"}>
                                      {l.disclosed.length}
                                    </td>,
                                    <td className="num" key={s + "p"}>
                                      {l.candidate_pool}
                                    </td>,
                                  ];
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="charts">
                        <LeakageChart
                          title="Details revealed about the target"
                          sub="Name, age, town, job and so on, counted after each question"
                          rounds={run.rounds.length}
                          values={series((l) => l.disclosed.length)}
                          yMax={maxDisclosed}
                          unit={(v) => plural(v, "detail")}
                        />
                        <LeakageChart
                          title="People who still match"
                          sub="Identified once this reaches 1"
                          rounds={run.rounds.length}
                          values={series((l) => l.candidate_pool)}
                          yMax={maxPool}
                          unit={(v) => (v === 1 ? "1 person" : `${v} people`)}
                        />
                      </div>
                    )}
                  </section>
                )}

                <section className="section">
                  <h2 className="section-title">Questions and answers</h2>
                  {visible.map((r, i) => (
                    <div className="round round-enter" key={r.round}>
                      <div className="round-no">Q{r.round}</div>
                      <div className="round-query">{r.query}</div>
                      <div className="round-pad" aria-hidden />
                      {SIDES.map((s) => {
                        const l = leakage(run, s, i);
                        return (
                          <div key={s}>
                            <SeriesLabel series={s} compact />
                            <p className="answer" style={{ marginTop: 8 }}>
                              <WithTokens text={r[s].answer} />
                            </p>
                            {l && (
                              <div className="round-leak">
                                {l.disclosed.length
                                  ? `Revealed: ${l.disclosed.join(", ")}. `
                                  : "Revealed nothing about the target. "}
                                {stillMatch(l.candidate_pool)}.
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </section>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
}
