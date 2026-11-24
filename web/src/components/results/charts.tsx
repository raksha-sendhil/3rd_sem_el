"use client";

import type { Pair } from "@/lib/results";

// the dashboard's two sides: coral for standard RAG, teal for ours
export const BASE = { label: "Standard RAG", color: "#FF4D3D", soft: "#FFE8E4" };
export const OURS = { label: "Protected RAG", color: "#0FB5A0", soft: "#DDF6F1" };
const INK = { primary: "#17162B", muted: "#6B6880", grid: "#EFECE4", track: "#F3F0E9" };

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

export function SideLegend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs font-medium text-[#4A4760]">
      {[BASE, OURS].map((s) => (
        <span key={s.label} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** One metric as two horizontal bars, standard RAG above ours, on a 0..max scale. */
export function PairBars({
  pair,
  max = 1,
  format = (v: number) => pct(v),
}: {
  pair: Pair;
  max?: number;
  format?: (v: number) => string;
}) {
  return (
    <div className="space-y-1.5">
      {[
        { side: BASE, v: pair.base },
        { side: OURS, v: pair.ours },
      ].map(({ side, v }) => (
        <div key={side.label} className="flex items-center gap-3">
          <div className="h-3 flex-1 overflow-hidden rounded-full" style={{ background: INK.track }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${Math.max(0, Math.min(1, v / max)) * 100}%`, background: side.color, minWidth: v > 0 ? 4 : 0 }}
            />
          </div>
          <span className="w-16 text-right font-mono text-xs font-bold" style={{ color: side.color }}>
            {format(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Grouped bars: each label with standard RAG and ours side by side. */
export function GroupedBars({
  items,
  max = 1,
  format = (v: number) => pct(v),
}: {
  items: { label: string; pair: Pair }[];
  max?: number;
  format?: (v: number) => string;
}) {
  return (
    <div className="space-y-5">
      {items.map((it) => (
        <div key={it.label}>
          <div className="mb-2 text-sm font-semibold text-[#17162B]">{it.label}</div>
          <PairBars pair={it.pair} max={max} format={format} />
        </div>
      ))}
    </div>
  );
}

/** Membership inference AUC on its meaningful range: 0.5 is guessing, 1.0 always right. */
export function AucMeter({ auc }: { auc: Pair }) {
  const W = 640;
  const H = 120;
  const pad = { left: 16, right: 16 };
  const x = (v: number) => pad.left + ((Math.max(0.5, Math.min(1, v)) - 0.5) / 0.5) * (W - pad.left - pad.right);
  const rows = [
    { side: BASE, v: auc.base, y: 30 },
    { side: OURS, v: auc.ours, y: 72 },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Membership inference AUC">
      {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={14} y2={96} stroke={INK.grid} strokeWidth={1} />
          <text x={x(t)} y={113} textAnchor="middle" fontSize={11} fill={INK.muted}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={x(0.5)} x2={x(0.5)} y1={10} y2={98} stroke={INK.primary} strokeWidth={2} strokeDasharray="4 4" />
      <text x={x(0.5) + 6} y={16} fontSize={11} fontWeight={700} fill={INK.primary}>
        guessing
      </text>
      {rows.map(({ side, v, y }) => (
        <g key={side.label}>
          <rect x={x(0.5)} y={y - 9} width={x(v) - x(0.5)} height={18} rx={9} fill={side.color} opacity={0.18} />
          <circle cx={x(v)} cy={y} r={10} fill={side.color} />
          <text x={x(v) + (v > 0.93 ? -16 : 16)} y={y + 5} textAnchor={v > 0.93 ? "end" : "start"} fontSize={15} fontWeight={800} fill={side.color}>
            {v.toFixed(2)}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Secret Recovery Rate after each round of an 8-round attack conversation. */
export function SrrLines({ base, ours }: { base: number[]; ours: number[] }) {
  const W = 640;
  const H = 240;
  const pad = { top: 14, right: 18, bottom: 30, left: 44 };
  const n = Math.max(base.length, ours.length);
  const top = Math.max(0.25, Math.ceil(Math.max(...base, ...ours) * 4) / 4);
  const x = (i: number) => pad.left + (n < 2 ? 0.5 : i / (n - 1)) * (W - pad.left - pad.right);
  const y = (v: number) => pad.top + (H - pad.top - pad.bottom) * (1 - v / top);
  const path = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  const ticks = Array.from({ length: 5 }, (_, i) => (top * i) / 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Secret Recovery Rate per round">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)} stroke={INK.grid} strokeWidth={1} />
          <text x={pad.left - 8} y={y(t) + 4} textAnchor="end" fontSize={10} fill={INK.muted}>
            {pct(t)}
          </text>
        </g>
      ))}
      {Array.from({ length: n }, (_, i) => (
        <text key={i} x={x(i)} y={H - 10} textAnchor="middle" fontSize={10} fill={INK.muted}>
          {i + 1}
        </text>
      ))}
      {[
        { side: BASE, vals: base },
        { side: OURS, vals: ours },
      ].map(({ side, vals }) => (
        <g key={side.label}>
          <path d={path(vals)} fill="none" stroke={side.color} strokeWidth={3} strokeLinejoin="round" strokeLinecap="round" />
          {vals.map((v, i) => (
            <circle key={i} cx={x(i)} cy={y(v)} r={3.5} fill="#fff" stroke={side.color} strokeWidth={2} />
          ))}
        </g>
      ))}
    </svg>
  );
}
