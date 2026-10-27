"use client";

import { Crosshair, Lock } from "lucide-react";
import type { AttackRound } from "@/api/client";
import { category } from "@/lib/agents";
import { scan, type Terms } from "@/lib/marks";
import { leakOf, pctOf, type Side } from "./meta";

// The target's attributes as evaluation/attacks.py scores them, and the detector categories
// that hold their values in the records.
const FIELDS = [
  { key: "name", label: "Name", cats: ["NAME"] },
  { key: "age", label: "Age", cats: ["AGE"] },
  { key: "town", label: "Town", cats: ["LOCATION", "ADDRESS"] },
  { key: "occupation", label: "Job", cats: ["OCCUPATION"] },
  { key: "phone", label: "Phone", cats: ["PHONE"] },
  { key: "condition", label: "Condition", cats: ["MEDICAL_CONDITION", "SYMPTOM"] },
];

const Ring = ({ pct, color, label, testid }: { pct: number; color: string; label: string; testid: string }) => {
  const c = 2 * Math.PI * 34;
  return (
    <div className="flex flex-col items-center" data-testid={testid}>
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle cx="44" cy="44" r="34" fill="none" stroke="#EFECE4" strokeWidth="9" />
        <circle
          cx="44"
          cy="44"
          r="34"
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct / 100)}
          transform="rotate(-90 44 44)"
          style={{ transition: "stroke-dashoffset 600ms ease" }}
        />
        <text x="44" y="50" textAnchor="middle" fontSize="18" fontWeight="800" fill="#17162B" className="font-display">
          {pct}%
        </text>
      </svg>
      <span className="mt-1 font-mono text-[10px] font-bold uppercase tracking-[0.14em]" style={{ color }}>
        {label}
      </span>
    </div>
  );
};

const Cell = ({ found, value, side }: { found: boolean; value: string; side: Side }) =>
  found ? (
    <span className={side === "std" ? "mark-leak" : "mark-vague"}>{value}</span>
  ) : (
    <span className="inline-flex items-center gap-1 font-mono text-[11px] text-[#A9A5B8]">
      <Lock className="h-3 w-3" /> hidden
    </span>
  );

export const TargetCard = ({
  targetId,
  rows,
  terms,
  intercept,
}: {
  targetId: string;
  rows: AttackRound[];
  terms: Terms;
  intercept: boolean;
}) => {
  const last = rows[rows.length - 1];
  const latest = { std: last ? leakOf(last, "std") : null, prot: last ? leakOf(last, "prot") : null };
  const said = (side: Side) =>
    rows
      .map((r) => {
        const a = side === "std" ? r.baseline : r.protected;
        return intercept ? a.payload : a.answer;
      })
      .join("\n");
  const texts = { std: said("std"), prot: said("prot") };

  /** The value the attacker now holds for ``cats``, as it appeared in what they saw. */
  const valueOf = (side: Side, cats: string[]) => {
    const types = cats.map(category);
    return (
      scan(texts[side], terms, "std").find((s) => s.kind === "leak" && types.includes(s.type ?? ""))?.text ??
      "revealed"
    );
  };
  const found = (side: Side, key: string) =>
    (latest[side]?.disclosed ?? []).some((d) => d.replace(/_\d+$/, "") === key);

  return (
    <aside className="card p-5" data-testid="target-card">
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-[#FFF3D1] text-[#8A5A00]">
          <Crosshair className="h-5 w-5" />
        </span>
        <div>
          <h3 className="font-display text-base font-bold text-[#17162B]">The target</h3>
          <p className="font-mono text-[11px] text-[#6B6880]">person {targetId}</p>
        </div>
      </div>
      <div className="mt-5 flex justify-around rounded-2xl bg-[#FBFAF7] py-4">
        <Ring pct={pctOf(latest.std)} color="#FF4D3D" label="Standard" testid="ring-standard" />
        <Ring pct={pctOf(latest.prot)} color="#0FB5A0" label="Protected" testid="ring-protected" />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-center">
        <div className="rounded-2xl bg-[#FFE8E4] p-3" data-testid="match-standard">
          <div className="font-display text-2xl font-extrabold text-[#B42318]">
            {latest.std?.candidate_pool.toLocaleString() ?? "—"}
          </div>
          <div className="font-mono text-[10px] text-[#B42318]">still match · std</div>
        </div>
        <div className="rounded-2xl bg-[#DDF7F2] p-3" data-testid="match-protected">
          <div className="font-display text-2xl font-extrabold text-[#08776A]">
            {latest.prot?.candidate_pool.toLocaleString() ?? "—"}
          </div>
          <div className="font-mono text-[10px] text-[#08776A]">still match · prot</div>
        </div>
      </div>
      <ul className="mt-5 divide-y divide-dashed divide-[#E6E1D6]">
        {FIELDS.map((f) => (
          <li
            key={f.key}
            className="grid grid-cols-[80px_1fr] gap-2 py-2.5 text-sm"
            data-testid={`target-detail-${f.key}`}
          >
            <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-[#6B6880]">
              {f.label}
            </span>
            <div className="flex flex-col items-start gap-1">
              <Cell found={found("std", f.key)} value={valueOf("std", f.cats)} side="std" />
              <Cell found={found("prot", f.key)} value={valueOf("prot", f.cats)} side="prot" />
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
};
