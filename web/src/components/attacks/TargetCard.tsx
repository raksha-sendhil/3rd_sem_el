"use client";

import { Crosshair } from "lucide-react";
import type { AttackRound } from "@/api/client";
import { leakOf, pctOf } from "./meta";

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

/** How much of the target each pipeline has given away so far. */
export const TargetCard = ({ targetId, rows }: { targetId: string; rows: AttackRound[] }) => {
  const last = rows[rows.length - 1];
  const std = last ? leakOf(last, "std") : null;
  const prot = last ? leakOf(last, "prot") : null;
  return (
    <aside className="card flex flex-col p-5 lg:col-span-2 xl:col-span-1" data-testid="target-card">
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
        <Ring pct={pctOf(std)} color="#FF4D3D" label="Standard" testid="ring-standard" />
        <Ring pct={pctOf(prot)} color="#0FB5A0" label="Protected" testid="ring-protected" />
      </div>
      <div className="mt-4 grid flex-1 grid-cols-2 gap-3 text-center">
        <div className="grid place-content-center rounded-2xl bg-[#FFE8E4] p-3" data-testid="match-standard">
          <div className="font-display text-2xl font-extrabold text-[#B42318]">
            {std?.candidate_pool.toLocaleString() ?? "—"}
          </div>
          <div className="font-mono text-[10px] text-[#B42318]">still match · std</div>
        </div>
        <div className="grid place-content-center rounded-2xl bg-[#DDF7F2] p-3" data-testid="match-protected">
          <div className="font-display text-2xl font-extrabold text-[#08776A]">
            {prot?.candidate_pool.toLocaleString() ?? "—"}
          </div>
          <div className="font-mono text-[10px] text-[#08776A]">still match · prot</div>
        </div>
      </div>
    </aside>
  );
};
