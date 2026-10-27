"use client";

import type { DocumentView } from "@/api/client";
import { AGENTS } from "@/lib/agents";

const H = ["#", "About", "Relation", "Value", "Importance"];

const LEVEL = {
  high: { bg: "#FFE8E4", color: "#B42318" },
  medium: { bg: "#FFF3D1", color: "#8A5A00" },
  low: { bg: "#EFECE4", color: "#6B6880" },
};

/** Agent 2's facts for one record. */
export const FactsTable = ({ facts }: { facts: DocumentView["record"]["extractor"]["facts"] }) => (
  <section className="card overflow-hidden" data-testid="facts-table">
    <div className="px-5 pb-3 pt-5">
      <h3 className="font-display text-base font-bold text-[#17162B] md:text-lg">Facts</h3>
      <p className="font-mono text-[11px] text-[#6B6880]">
        {facts.length} extracted by {AGENTS.extract.step} · {AGENTS.extract.name}
      </p>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-sm">
        <thead>
          <tr className="border-y border-[#E6E1D6] bg-[#FBFAF7] text-left">
            {H.map((h) => (
              <th
                key={h}
                className="px-4 py-2.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880]"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {facts.map((f, i) => {
            const level = LEVEL[f.importance];
            return (
              <tr key={i} className="border-b border-[#EFEBE2] transition-colors last:border-0 hover:bg-[#FBFAF7]">
                <td className="px-4 py-3 font-mono text-xs text-[#6B6880]">{String(i + 1).padStart(2, "0")}</td>
                <td className="px-4 py-3 font-mono text-xs text-[#4A4760]">{f.subject}</td>
                <td className="px-4 py-3 text-xs font-medium text-[#4A4760]">{f.relation.replaceAll("_", " ")}</td>
                <td className="px-4 py-3 font-medium text-[#17162B]">{f.value}</td>
                <td className="px-4 py-3">
                  <span className="chip" style={{ background: level.bg, color: level.color }}>
                    {f.importance}
                  </span>
                </td>
              </tr>
            );
          })}
          {!facts.length && (
            <tr>
              <td colSpan={H.length} className="px-4 py-10 text-center text-sm text-[#A9A5B8]">
                No facts
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  </section>
);
