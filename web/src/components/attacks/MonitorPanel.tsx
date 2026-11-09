"use client";

import { Eye } from "lucide-react";
import { useState } from "react";
import type { MonitorLedger, MonitorVerdict } from "@/api/client";
import { Verdict } from "../Verdict";

export type MonitorRound = { query: string; monitor: MonitorVerdict };

const INK = { primary: "#17162B", secondary: "#4A4760", muted: "#6B6880", grid: "#EFECE4" };

/** Candidate pool per question as columns, with the k line: below it the session could single
 * a person out. A column's colour is the verdict; its value sits on the cap. */
function PoolChart({ rounds, k }: { rounds: MonitorRound[]; k: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 360;
  const H = 180;
  const pad = { top: 18, right: 12, bottom: 26, left: 34 };
  const pools = rounds.map((r) => r.monitor.candidate_pool);
  const top = Math.max(k * 2, ...pools.map((p) => p ?? 0));
  const y = (v: number) => pad.top + (H - pad.top - pad.bottom) * (1 - v / top);
  const band = (W - pad.left - pad.right) / Math.max(rounds.length, 1);
  const barW = Math.min(24, band * 0.5);
  const COLOR = { allowed: "#0FB5A0", generalised: "#E08E00", blocked: "#FF4D3D" };
  const ticks = [0, Math.round(top / 2), top];

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Candidate pool per question">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)} stroke={INK.grid} strokeWidth={1} />
            <text x={pad.left - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={INK.muted}>
              {t}
            </text>
          </g>
        ))}
        {rounds.map((r, i) => {
          const p = r.monitor.candidate_pool;
          const cx = pad.left + band * i + band / 2;
          if (p == null)
            return (
              <text key={i} x={cx} y={y(0) - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>
                none
              </text>
            );
          const h = Math.max(y(0) - y(p), 2);
          const x0 = cx - barW / 2;
          const rr = Math.min(4, h);
          const path = `M${x0},${y(0)} V${y(p) + rr} Q${x0},${y(p)} ${x0 + rr},${y(p)} H${x0 + barW - rr} Q${x0 + barW},${y(p)} ${x0 + barW},${y(p) + rr} V${y(0)} Z`;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={cx - band / 2} y={pad.top} width={band} height={H - pad.top - pad.bottom} fill="transparent" />
              <path d={path} fill={COLOR[r.monitor.verdict]} opacity={hover === null || hover === i ? 1 : 0.45} />
              <text x={cx} y={y(p) - 5} textAnchor="middle" fontSize={10} fontWeight={600} fill={INK.primary}>
                {p}
              </text>
              <text x={cx} y={H - 9} textAnchor="middle" fontSize={10} fill={INK.muted}>
                Q{i + 1}
              </text>
            </g>
          );
        })}
        <line x1={pad.left} x2={W - pad.right} y1={y(k)} y2={y(k)} stroke={INK.primary} strokeWidth={1} />
        <text x={W - pad.right} y={y(k) - 4} textAnchor="end" fontSize={10} fontWeight={600} fill={INK.primary}>
          k = {k}
        </text>
      </svg>
      {hover !== null && (
        <div className="pointer-events-none absolute left-1/2 top-0 w-[90%] -translate-x-1/2 rounded-xl border border-[#E6E1D6] bg-white px-3 py-2 text-xs text-[#2A2840] shadow-lg">
          <b>Q{hover + 1}</b> · {rounds[hover].monitor.reason}
        </div>
      )}
    </div>
  );
}

/** The protected chat seen through the session monitor: per question, its verdict, the kinds
 * of detail the session now knows about the person, and how many records still match. */
export function MonitorPanel({ rounds, ledger }: { rounds: MonitorRound[]; ledger: MonitorLedger | null }) {
  const k = ledger?.k ?? 5;
  const n = ledger?.max_attributes ?? 3;
  const off = ledger?.monitor === "off";
  return (
    <section className="card overflow-hidden" data-testid="monitor-panel">
      <div className="flex flex-wrap items-center gap-3 border-b border-[#E6E1D6] bg-[#DDF7F2] px-5 py-4">
        <span className="grid h-10 w-10 place-items-center rounded-2xl bg-[#0FB5A0] text-white">
          <Eye className="h-5 w-5" strokeWidth={2.4} />
        </span>
        <div>
          <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#08776A]">
            Protected RAG · session privacy monitor
          </div>
          <div className="font-display text-base font-bold text-[#17162B]">
            {off ? "Switched off on this API" : "What this session has learned, question by question"}
          </div>
        </div>
        <div className="ml-auto font-mono text-[11px] text-[#4A4760]">
          at least {k} matching records · at most {n} kinds of detail per person
        </div>
      </div>
      {rounds.length === 0 ? (
        <p className="px-5 py-6 text-sm text-[#6B6880]">
          Ask the protected side something, or pick <b>Stepwise inference</b> above: each answer is checked against
          what the session already knows before it is shown.
        </p>
      ) : (
        <div className="grid gap-5 px-5 py-5 lg:grid-cols-[1.4fr_1fr]">
          <table className="w-full text-left text-sm" data-testid="monitor-rounds">
            <thead>
              <tr className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#6B6880]">
                <th className="pb-2 pr-2">#</th>
                <th className="pb-2 pr-2">Question</th>
                <th className="pb-2 pr-2">Known about the person</th>
                <th className="pb-2">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {rounds.map((r, i) => (
                <tr key={i} className="border-t border-[#EFECE4] align-top">
                  <td className="py-2 pr-2 font-mono text-xs text-[#6B6880]">Q{i + 1}</td>
                  <td className="py-2 pr-2 text-[#2A2840]">{r.query}</td>
                  <td className="py-2 pr-2">
                    <div className="flex flex-wrap gap-1">
                      {(r.monitor.disclosed ?? []).length === 0 ? (
                        <span className="font-mono text-[11px] text-[#A9A5B8]">nothing</span>
                      ) : (
                        (r.monitor.disclosed ?? []).map((d) => (
                          <span key={d} className="chip bg-[#F1EFE9] text-[#4A4760]">
                            {d}
                          </span>
                        ))
                      )}
                    </div>
                  </td>
                  <td className="py-2">
                    <Verdict monitor={r.monitor} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div>
            <div className="mb-1 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880]">
              Records still matching the person
            </div>
            <PoolChart rounds={rounds} k={k} />
            <div className="mt-1 flex flex-wrap gap-3 font-mono text-[10px] text-[#4A4760]">
              {[
                ["#0FB5A0", "allowed"],
                ["#E08E00", "generalised"],
                ["#FF4D3D", "blocked"],
              ].map(([c, l]) => (
                <span key={l} className="flex items-center gap-1">
                  <span className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />
                  {l}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
