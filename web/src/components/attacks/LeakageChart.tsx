"use client";

import { useState } from "react";
import type { AttackRun } from "@/api/client";

// identity colours, checked for colour-blind separation (coral / purple / teal)
const SERIES = [
  { key: "baseline_leakage", label: "Standard RAG", color: "#FF4D3D", at: "baseline_identified_at", dash: undefined },
  // dashed: where it equals the monitored line, both stay visible
  { key: "unmonitored_leakage", label: "Rewriting alone", color: "#6D4AFF", at: "unmonitored_identified_at", dash: "5 4" },
  { key: "protected_leakage", label: "Rewriting + session monitor", color: "#0FB5A0", at: "protected_identified_at", dash: undefined },
] as const;


const INK = { primary: "#17162B", muted: "#6B6880", grid: "#EFECE4" };

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Secret Recovery Rate (Sun et al., 2026) after each round of a scored attack: the share of
 * the target's private fields the attacker has recovered so far, one line per set-up. */
export function LeakageChart({ run }: { run: AttackRun }) {
  const [hover, setHover] = useState<number | null>(null);
  const rounds = run.rounds;
  const W = 560;
  const H = 230;
  const pad = { top: 14, right: 16, bottom: 28, left: 40 };
  const x = (i: number) =>
    pad.left + (rounds.length < 2 ? 0.5 : i / (rounds.length - 1)) * (W - pad.left - pad.right);
  const y = (v: number) => pad.top + (H - pad.top - pad.bottom) * (1 - v);
  const value = (i: number, key: (typeof SERIES)[number]["key"]) => rounds[i][key]?.recovery_rate ?? 0;
  // "rewriting alone" only where the API scores it (the legal pipeline reports two set-ups)
  const shown = SERIES.filter((s) => s.key !== "unmonitored_leakage" || rounds.some((r) => r.unmonitored_leakage));
  // the dashed line goes on top, so it shows where it runs along another
  const drawOrder = [...shown.filter((s) => !s.dash), ...shown.filter((s) => s.dash)];

  return (
    <div data-testid="leakage-chart" className="grid gap-6 lg:grid-cols-[minmax(0,680px)_1fr]">
      <div className="min-w-0">
      <div className="mb-2 flex flex-wrap gap-4 text-xs text-[#4A4760]">
        {shown.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <svg width="20" height="4" aria-hidden>
              <line x1="1" x2="19" y1="2" y2="2" stroke={s.color} strokeWidth={3} strokeLinecap="round" strokeDasharray={s.dash ? "4 3" : undefined} />
            </svg>
            {s.label}
          </span>
        ))}
      </div>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Secret Recovery Rate per round">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)} stroke={INK.grid} strokeWidth={1} />
              <text x={pad.left - 8} y={y(t) + 4} textAnchor="end" fontSize={10} fill={INK.muted}>
                {pct(t)}
              </text>
            </g>
          ))}
          {rounds.map((r, i) => (
            <text key={r.round} x={x(i)} y={H - 8} textAnchor="middle" fontSize={10} fill={INK.muted}>
              Q{r.round}
            </text>
          ))}
          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={pad.top} y2={y(0)} stroke="#CFC8B8" strokeWidth={1} />
          )}
          {drawOrder.map((s) => (
            <g key={s.key}>
              <polyline
                points={rounds.map((_, i) => `${x(i)},${y(value(i, s.key))}`).join(" ")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                strokeDasharray={s.dash}
              />
              {rounds.map((_, i) => (
                <circle key={i} cx={x(i)} cy={y(value(i, s.key))} r={4} fill={s.color} stroke="#fff" strokeWidth={2} />
              ))}
            </g>
          ))}
          {rounds.map((_, i) => (
            <rect
              key={i}
              x={x(i) - (W - pad.left - pad.right) / Math.max(rounds.length - 1, 1) / 2}
              y={pad.top}
              width={(W - pad.left - pad.right) / Math.max(rounds.length - 1, 1)}
              height={H - pad.top - pad.bottom}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
        {hover !== null && (
          <div
            className="pointer-events-none absolute top-2 rounded-xl border border-[#E6E1D6] bg-white px-3 py-2 text-xs text-[#2A2840] shadow-lg"
            style={{ left: `${Math.min(70, (x(hover) / W) * 100)}%` }}
          >
            <div className="mb-1 font-bold">Q{rounds[hover].round}</div>
            {shown.map((s) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                {s.label}: <b>{pct(value(hover, s.key))}</b>
                <span className="text-[#6B6880]">
                  · {rounds[hover][s.key]?.candidate_pool ?? "-"} match
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
      </div>
      <div className="min-w-0">
      <table className="w-full text-left text-xs" data-testid="leakage-table">
        <thead>
          <tr className="font-mono text-[10px] uppercase tracking-[0.12em] text-[#6B6880]">
            <th className="pb-2 pr-2">Round</th>
            {shown.map((s) => (
              <th key={s.key} className="pb-2 pr-2">
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rounds.map((r, i) => (
            <tr key={r.round} className="border-t border-[#EFECE4]">
              <td className="py-1.5 pr-2 font-mono text-[#6B6880]">Q{r.round}</td>
              {shown.map((s) => (
                <td key={s.key} className="py-1.5 pr-2 tabular-nums text-[#17162B]">
                  {pct(value(i, s.key))}
                  <span className="text-[#6B6880]"> · {r[s.key]?.candidate_pool ?? "-"} match</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <ul className="mt-4 space-y-0.5 text-xs text-[#4A4760]">
        {shown.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
            {s.label}:{" "}
            {run[s.at] != null ? (
              <b className="text-[#17162B]">target identified at Q{run[s.at]}</b>
            ) : (
              <span>target never identified</span>
            )}
          </li>
        ))}
      </ul>
      </div>
    </div>
  );
}
