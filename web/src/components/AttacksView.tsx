"use client";

import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  streamAttack,
  type AttackDefinition,
  type AttackKind,
  type AttackRound,
  type AttackRun,
  type RoundLeakage,
} from "@/api/client";
import {
  Loader,
  Marked,
  Notice,
  PixelArrow,
  SERIES,
  SIDES,
  Win,
  useElapsed,
  useLeakTerms,
  type SeriesId,
} from "./common";
import { LeakageChart } from "./LeakageChart";

const ORDER: AttackKind[] = ["context_dump", "stepwise", "interception"];
const TONE: Record<AttackKind, string> = {
  context_dump: "pink",
  stepwise: "teal",
  interception: "orange",
};
const COUNT: Record<AttackKind, (n: number) => string> = {
  context_dump: (n) => (n === 1 ? "1 prompt" : `${n} prompts`),
  stepwise: (n) => `${n} prompts`,
  interception: (n) => `${n} requests, no prompts`,
};

const pct = (v: number) => `${Math.round(v * 100)}%`;
const people = (n: number) => (n === 1 ? "1 person matches" : `${n} people match`);

function leakOf(r: AttackRound, side: SeriesId): RoundLeakage | null {
  return (side === "baseline" ? r.baseline_leakage : r.protected_leakage) ?? null;
}

export function AttacksView() {
  const [attacks, setAttacks] = useState<AttackDefinition[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [rounds, setRounds] = useState<AttackRound[]>([]);
  const [run, setRun] = useState<AttackRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asTable, setAsTable] = useState(false);
  const stop = useRef<(() => void) | null>(null);
  const elapsed = useElapsed(busy);
  const leaks = useLeakTerms(rounds.flatMap((r) => r.baseline.doc_ids));

  useEffect(() => {
    api
      .attacks()
      .then((list) => {
        const sorted = [...list].sort((a, b) => ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind));
        setAttacks(sorted);
        setSelected((s) => s ?? sorted[0]?.attack_id ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    return () => stop.current?.();
  }, []);

  const attack = attacks?.find((a) => a.attack_id === selected) ?? null;
  const intercept = attack?.kind === "interception";

  function reset() {
    stop.current?.();
    setRounds([]);
    setRun(null);
    setBusy(false);
    setError(null);
  }

  function start() {
    if (!attack) return;
    reset();
    setBusy(true);
    stop.current = streamAttack(attack.attack_id, {
      round: (r) => setRounds((rs) => [...rs, r]),
      done: (result) => {
        setRun(result);
        setBusy(false);
      },
      fail: (message) => {
        setError(message);
        setBusy(false);
      },
    });
  }

  const total = attack?.queries.length ?? 0;
  const scored = rounds.length > 0 && rounds.every((r) => r.baseline_leakage);
  const series = (pick: (l: RoundLeakage) => number) =>
    Object.fromEntries(SIDES.map((s) => [s, rounds.map((r) => pick(leakOf(r, s)!))])) as Record<
      SeriesId,
      number[]
    >;
  const maxPool = Math.max(
    1,
    ...rounds.flatMap((r) => SIDES.map((s) => leakOf(r, s)?.candidate_pool ?? 0)),
  );
  const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find((st) => maxPool / st <= 5) ?? 1000;
  const poolTop = Math.ceil(maxPool / step) * step;
  const poolTicks = Array.from({ length: poolTop / step + 1 }, (_, i) => i * step);

  return (
    <>
      <Notice error={error} />
      <div className="arena">
        <ol className="steps">
          {attacks?.map((a) => {
            const current = a.attack_id === selected;
            return (
              <li className="step" key={a.attack_id} data-kind={a.kind} aria-current={current}>
                <span className="step-node" aria-hidden />
                <button
                  type="button"
                  className="step-pick"
                  onClick={() => {
                    if (busy || current) return;
                    reset();
                    setSelected(a.attack_id);
                  }}
                >
                  <span className="step-name">{a.name}</span>
                  <span className="step-count" style={{ display: "block" }}>
                    {COUNT[a.kind](a.queries.length)}
                  </span>
                </button>
                {current && (
                  <button type="button" className="btn btn-lime" onClick={start} disabled={busy}>
                    {rounds.length ? "Run again" : "Run attack"}
                    <PixelArrow />
                  </button>
                )}
              </li>
            );
          })}
        </ol>

        {attack && (
          <div className="stack">
            <Win title={attack.name} tone={TONE[attack.kind]}>
              <div className="win-body">
                <ol className="prompts">
                  {attack.queries.map((q, i) => (
                    <li key={q}>
                      <b>{String(i + 1).padStart(2, "0")}</b>
                      <span>{q}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </Win>

            {(busy || rounds.length > 0) && (
              <div className="scores">
                {SIDES.map((s) => {
                  const latest = rounds.length ? leakOf(rounds[rounds.length - 1], s) : null;
                  const at = rounds.find((r) => leakOf(r, s)?.identified)?.round;
                  const filled = Math.round((latest?.recovery_rate ?? 0) * 10);
                  return (
                    <Win key={s} title={SERIES[s].name} series={s}>
                      <div className="win-body">
                        <div className="score-num">
                          {latest ? Math.round(latest.recovery_rate * 100) : 0}
                          <small>%</small>
                        </div>
                        <div className="meter" aria-hidden>
                          {Array.from({ length: 10 }, (_, i) => (
                            <i key={i} data-on={i < filled} />
                          ))}
                        </div>
                        <div className="score-row">
                          {at ? (
                            <span className="tag" data-tone="hot">
                              Identified · R{at}
                            </span>
                          ) : run ? (
                            <span className="tag" data-tone="lime">
                              Not identified
                            </span>
                          ) : (
                            <Loader seconds={elapsed} />
                          )}
                          <span>{latest ? people(latest.candidate_pool) : ""}</span>
                        </div>
                      </div>
                    </Win>
                  );
                })}
              </div>
            )}

            {scored && (
              <>
                <div className="bar-row">
                  <div className="legend" aria-label="Legend">
                    {SIDES.map((s) => (
                      <span key={s}>
                        <span className="sq" data-series={s} aria-hidden />
                        {SERIES[s].name}
                      </span>
                    ))}
                  </div>
                  <span style={{ display: "inline-flex", gap: 8 }}>
                    <button
                      type="button"
                      className="mini"
                      aria-pressed={!asTable}
                      onClick={() => setAsTable(false)}
                    >
                      Chart
                    </button>
                    <button
                      type="button"
                      className="mini"
                      aria-pressed={asTable}
                      onClick={() => setAsTable(true)}
                    >
                      Table
                    </button>
                  </span>
                </div>
                {asTable ? (
                  <Win title="Session trace" tone="purple">
                    <div className="table-wrap">
                      <table className="table">
                        <thead>
                          <tr>
                            <th className="num">Round</th>
                            <th>{intercept ? "Request" : "Prompt"}</th>
                            {SIDES.map((s) => (
                              <th key={s}>{SERIES[s].name}</th>
                            ))}
                            <th>Verdict</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rounds.map((r) => (
                            <tr key={r.round}>
                              <td className="num px">{r.round}</td>
                              <td style={{ minWidth: 200 }}>{r.query}</td>
                              {SIDES.map((s) => {
                                const l = leakOf(r, s)!;
                                return (
                                  <td key={s} style={{ minWidth: 170 }}>
                                    <b>{pct(l.recovery_rate)}</b> · {people(l.candidate_pool)}
                                    <div className="sub">{l.disclosed.join(", ") || "nothing"}</div>
                                  </td>
                                );
                              })}
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
                ) : (
                  <div className="charts">
                    <Win title="Details recovered" tone="yellow">
                      <div className="win-body">
                        <LeakageChart
                          label="Share of the target's details recovered, per round"
                          rounds={total}
                          values={series((l) => l.recovery_rate)}
                          yMax={1}
                          ticks={[0, 0.25, 0.5, 0.75, 1]}
                          format={pct}
                        />
                      </div>
                    </Win>
                    <Win title="People still matching" tone="lime">
                      <div className="win-body">
                        <LeakageChart
                          label="People still matching what was revealed, per round"
                          rounds={total}
                          values={series((l) => l.candidate_pool)}
                          yMax={poolTop}
                          ticks={poolTicks}
                          format={(v) => String(v)}
                        />
                      </div>
                    </Win>
                  </div>
                )}
              </>
            )}

            {rounds.map((r) => (
              <Win
                key={r.round}
                className="enter"
                title={`Round ${String(r.round).padStart(2, "0")}`}
                tone={TONE[attack.kind]}
              >
                <div className="win-body">
                  <span className="lbl" style={{ color: "var(--ink-2)" }}>
                    {intercept ? "A user asks, the attacker listens" : "Attacker asks"}
                  </span>
                  <p className="query-line">{r.query}</p>
                </div>
                <div className="round-sides">
                  {SIDES.map((s) => {
                    const l = leakOf(r, s);
                    return (
                      <div className="half" key={s}>
                        <div className="half-name">
                          <span className="sq" data-series={s} aria-hidden />
                          {SERIES[s].name}
                          {intercept && " → model"}
                        </div>
                        <div className={intercept ? "wire" : "answer"}>
                          <Marked text={intercept ? r[s].payload : r[s].answer} leaks={leaks} />
                        </div>
                        {l && (
                          <div className="found">
                            {l.disclosed.length ? (
                              l.disclosed.map((d) => (
                                <span className="tag" data-tone="ink" key={d}>
                                  {d.replace(/_\d+$/, "")}
                                </span>
                              ))
                            ) : (
                              <span className="tag" data-tone="lime">
                                nothing
                              </span>
                            )}
                            <span className="pool">{people(l.candidate_pool)}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </Win>
            ))}
            {busy && rounds.length < total && <Loader seconds={elapsed} />}
          </div>
        )}
      </div>
    </>
  );
}
