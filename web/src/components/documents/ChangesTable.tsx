"use client";

import { ArrowRight } from "lucide-react";
import { AGENTS, KIND } from "@/lib/agents";
import type { Change } from "./changes";

const H = ["#", "Original", "", "Replacement", "Type", "Rewritten by", "Technique", "Detect conf."];

export const ChangesTable = ({
  docId,
  changes,
  active,
  onHover,
}: {
  docId: string;
  changes: Change[];
  active: number | null;
  onHover: (i: number | null) => void;
}) => (
  <section className="card overflow-hidden" data-testid="transformation-table">
    <div className="px-5 pb-3 pt-5">
      <h3 className="font-display text-base font-bold text-[#17162B] md:text-lg">Table of changes</h3>
      <p className="font-mono text-[11px] text-[#6B6880]">
        {changes.length} edits applied to {docId}
      </p>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-sm">
        <thead>
          <tr className="border-y border-[#E6E1D6] bg-[#FBFAF7] text-left">
            {H.map((h, i) => (
              <th
                key={i}
                className="px-4 py-2.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880]"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {changes.map((c, i) => {
            const k = KIND[c.kind];
            return (
              <tr
                key={c.id}
                data-testid={`change-row-${c.id}`}
                onMouseEnter={() => onHover(c.id)}
                onMouseLeave={() => onHover(null)}
                className={`border-b border-[#EFEBE2] transition-colors ${
                  active === c.id ? "bg-[#FFF8E6]" : "hover:bg-[#FBFAF7]"
                }`}
              >
                <td className="px-4 py-3 font-mono text-xs text-[#6B6880]">{String(i + 1).padStart(2, "0")}</td>
                <td className="px-4 py-3">
                  <span className="mark-leak">{c.original}</span>
                </td>
                <td className="px-1 py-3 text-[#A9A5B8]">
                  <ArrowRight className="h-4 w-4" />
                </td>
                <td className="px-4 py-3">
                  <span className={c.kind === "token" ? "mark-token" : "mark-vague"}>{c.replacement}</span>
                </td>
                <td className="px-4 py-3 text-xs font-medium text-[#4A4760]">{c.type}</td>
                <td className="px-4 py-3">
                  <span className="chip" style={{ background: k.bg, color: k.color }}>
                    {AGENTS.rewrite.step} · {AGENTS.rewrite.name}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-[#4A4760]">{k.label}</td>
                <td className="px-4 py-3">
                  {c.conf === null ? (
                    <span className="font-mono text-xs text-[#A9A5B8]">—</span>
                  ) : (
                    <div className="flex items-center gap-2 font-mono text-xs text-[#4A4760]">
                      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-[#EFECE4]">
                        <div className="h-full rounded-full bg-[#2D8CFF]" style={{ width: `${c.conf * 100}%` }} />
                      </div>
                      {c.conf.toFixed(2)}
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  </section>
);
