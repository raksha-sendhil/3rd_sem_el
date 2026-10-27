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
  ArrowRight,
  Marked,
  Notice,
  SERIES,
  SIDES,
  Working,
  useElapsed,
  useLeakTerms,
  type SeriesId,
} from "./common";
import { LeakageChart } from "./LeakageChart";

const ORDER: AttackKind[] = ["context_dump", "stepwise", "interception"];
const COUNT: Record<AttackKind, (n: number) => string> = {
  context_dump: (n) => (n === 1 ? "1 prompt" : `${n} prompts`),
  stepwise: (n) => `${n} prompts`,
  interception: (n) => `${n} intercepted requests, no prompts`,
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
  const maxPool = Math.max(1, ...rounds.flatMap((r) => SIDES.map((s) => leakOf(r, s)?.candidate_pool ?? 0)));
  const poolStep = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find((st) => maxPool / st <= 5) ?? 1000;
  const poolTop = Math.ceil(maxPool / poolStep) * poolStep;
  const poolTicks = Array.from({ length: poolTop / poolStep + 1 }, (_, i) => i * poolStep);

  return (
    <>
      <Notice error={error} />

      <div className="attack-cards">
        {attacks?.map((a, i) => (
          <button
            key={a.attack_id}
            type="button"
            className="attack-card"
            data-kind={a.kind}
            aria-pressed={a.attack_id === selected}
            onClick={() => {
              if (busy) return;
              reset();
              setSelected(a.attack_id);
            }}
          >
            <span className="no">{String(i + 1).padStart(2, "0")}</span>
            <span className="name">{a.name}</span>
            <span className="count">{COUNT[a.kind](a.queries.length)}</span>
          </button>
        ))}
      </div>

      {attack && (
        <div className="attack-run">
          <div>
            <div className="label-sm" style={{ marginBottom: 6 }}>
              {intercept ? "Requests observed" : "Prompts"}
            </div>
            <ol className="prompts">
              {attack.queries.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ol>
          </div>
          <button type="button" className="btn" onClick={start} disabled={busy}>
            {rounds.length ? "Again" : "Run"}
            <ArrowRight />
          </button>
        </div>
      )}

      {(busy || rounds.length > 0) && (
        <div className="scoreboard">
          {SIDES.map((s) => {
            const latest = rounds.length ? leakOf(rounds[rounds.length - 1], s) : null;
            const at = rounds.find((r) => leakOf(r, s)?.identified)?.round;
            return (
              <div className="score" data-series={s} key={s}>
                <span className="big">
                  {latest ? Math.round(latest.recovery_rate * 100) : 0}
                  <small>%</small>
                </span>
                <span className="who">
                  <span className="swatch" data-series={s} aria-hidden />
                  <b>{SERIES[s].name}</b>
                </span>
                <span className="status">
                  {at ? (
                    <span className="stamp">Identified · round {at}</span>
                  ) : run ? (
                    "Not identified"
                  ) : (
                    <Working seconds={elapsed} />
                  )}
                </span>
                <span className="sub">
                  {latest ? `recovered · ${people(latest.candidate_pool)}` : "recovered"}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {scored && (
        <section className="section">
          <div className="section-head">
            <div className="legend" aria-label="Legend">
              {SIDES.map((s) => (
                <span key={s}>
                  <span className="swatch" data-series={s} aria-hidden />
                  {SERIES[s].name}
                </span>
              ))}
            </div>
            <div className="toggle" role="group" aria-label="Show as">
              <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
                Chart
              </button>
              <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
                Table
              </button>
            </div>
          </div>
          {asTable ? (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th className="num">Round</th>
                    <th>{intercept ? "Request" : "Prompt"}</th>
                    {SIDES.map((s) => (
                      <th key={s}>
                        <span className="side swatch" data-series={s} aria-hidden />
                        {SERIES[s].name}
                      </th>
                    ))}
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {rounds.map((r) => (
                    <tr key={r.round}>
                      <td className="num">{r.round}</td>
                      <td style={{ minWidth: 200 }}>{r.query}</td>
                      {SIDES.map((s) => {
                        const l = leakOf(r, s)!;
                        return (
                          <td key={s} style={{ minWidth: 170 }}>
                            <b>{pct(l.recovery_rate)}</b> · {people(l.candidate_pool)}
                            <div className="mono" style={{ color: "var(--ink-2)" }}>
                              {l.disclosed.join(", ") || "nothing"}
                            </div>
                          </td>
                        );
                      })}
                      <td>{r.protected.monitor?.verdict ?? "allowed"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="charts">
              <LeakageChart
                title="Target's details recovered"
                rounds={total}
                values={series((l) => l.recovery_rate)}
                yMax={1}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                format={pct}
              />
              <LeakageChart
                title="People who still match"
                rounds={total}
                values={series((l) => l.candidate_pool)}
                yMax={poolTop}
                ticks={poolTicks}
                format={(v) => String(v)}
              />
            </div>
          )}
        </section>
      )}

      {rounds.length > 0 && (
        <section className="section">
          {rounds.map((r) => (
            <div className="round enter" key={r.round}>
              <div className="round-no">{String(r.round).padStart(2, "0")}</div>
              <div className="round-query">
                <span className="label-sm">{intercept ? "Someone asks" : "Attacker asks"}</span>
                {r.query}
              </div>
              {SIDES.map((s) => {
                const l = leakOf(r, s);
                const said = intercept ? r[s].payload : r[s].answer;
                return (
                  <section className="panel" data-series={s} key={s}>
                    <div className="panel-head">
                      <span className="label">
                        {SERIES[s].name}
                        {intercept && " → model"}
                      </span>
                    </div>
                    <div className="panel-body">
                      <div className={intercept ? "wire" : "answer"}>
                        <Marked text={said} leaks={s === "baseline" ? leaks : []} />
                      </div>
                    </div>
                    {l && (
                      <div className="found">
                        {l.disclosed.length ? (
                          l.disclosed.map((d) => (
                            <span className="tag" key={d}>
                              {d.replace(/_\d+$/, "")}
                            </span>
                          ))
                        ) : (
                          <span>nothing about the target</span>
                        )}
                        <span className="pool">{people(l.candidate_pool)}</span>
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          ))}
          {busy && rounds.length < total && <Working seconds={elapsed} />}
        </section>
      )}
    </>
  );
}
