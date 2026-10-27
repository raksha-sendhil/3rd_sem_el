"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { AttackRound } from "@/api/client";
import { Highlighted, type Terms } from "@/lib/marks";
import { Verdict, detail } from "../Verdict";
import { leakOf, pctOf, type Meta } from "./meta";

const H = [
  "#",
  "Attacker",
  "Standard RAG reply",
  "Protected RAG reply",
  "Recovered",
  "Still matching",
  "Verdict",
];

const Disclosed = ({ keys, color }: { keys: string[] | undefined; color: string }) => (
  <div className="mt-1 max-w-[150px] whitespace-normal text-[10px] leading-4" style={{ color }}>
    {keys?.length ? keys.map(detail).join(", ") : "nothing"}
  </div>
);

const kb = (s: string) => `${(new Blob([s]).size / 1024).toFixed(1)} KB`;
const pool = (n: number | undefined) => (n === undefined ? "—" : n.toLocaleString());

export const RoundsTable = ({
  attackId,
  meta,
  intercept,
  rows,
  terms,
}: {
  attackId: string;
  meta: Meta;
  intercept: boolean;
  rows: AttackRound[];
  terms: Terms;
}) => (
  <section className="card overflow-hidden" data-testid="attack-rounds-table">
    <div className="px-5 pb-3 pt-5">
      <h3 className="font-display text-base font-bold text-[#17162B] md:text-lg">Live transcript</h3>
      <p className="font-mono text-[11px] text-[#6B6880]">
        each {meta.unit.toLowerCase()} appears as the attacker receives it
      </p>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] text-sm">
        <thead>
          <tr className="border-y border-[#E6E1D6] bg-[#FBFAF7] text-left">
            {H.map((h) => (
              <th
                key={h}
                className="whitespace-nowrap px-4 py-2.5 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880]"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <AnimatePresence initial={false}>
            {rows.map((r, i) => {
              const std = leakOf(r, "std");
              const prot = leakOf(r, "prot");
              return (
                <motion.tr
                  key={`${attackId}-${i}`}
                  data-testid={`attack-round-row-${i + 1}`}
                  initial={{ opacity: 0, backgroundColor: meta.bg }}
                  animate={{ opacity: 1, backgroundColor: "#FFFFFF" }}
                  transition={{ duration: 0.9 }}
                  className="border-b border-[#EFEBE2] align-top"
                >
                  <td className="px-4 py-3 font-mono text-xs font-bold" style={{ color: meta.color }}>
                    {String(i + 1).padStart(2, "0")}
                  </td>
                  <td className="max-w-[220px] px-4 py-3 font-medium text-[#17162B]">
                    {intercept ? (
                      <>
                        Captured request · {kb(r.baseline.payload)}
                        <div className="mt-1 text-xs font-normal text-[#6B6880]">A user asks: {r.query}</div>
                      </>
                    ) : (
                      r.query
                    )}
                  </td>
                  <td className="max-w-[260px] px-4 py-3 leading-7 text-[#2A2840]">
                    <div className={intercept ? "clamp-6" : undefined}>
                      <Highlighted
                        text={intercept ? r.baseline.payload : r.baseline.answer}
                        terms={terms}
                        side="std"
                      />
                    </div>
                  </td>
                  <td className="max-w-[260px] px-4 py-3 leading-7 text-[#2A2840]">
                    <div className={intercept ? "clamp-6" : undefined}>
                      <Highlighted
                        text={intercept ? r.protected.payload : r.protected.answer}
                        terms={terms}
                        side="prot"
                      />
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs">
                    <span className="font-bold text-[#D92D20]">{pctOf(std)}%</span> /{" "}
                    <span className="font-bold text-[#08776A]">{pctOf(prot)}%</span>
                    <Disclosed keys={std?.disclosed} color="#B42318" />
                    <Disclosed keys={prot?.disclosed} color="#08776A" />
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs">
                    <span className="font-bold text-[#D92D20]">{pool(std?.candidate_pool)}</span> /{" "}
                    <span className="font-bold text-[#08776A]">{pool(prot?.candidate_pool)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <Verdict monitor={r.protected.monitor} />
                  </td>
                </motion.tr>
              );
            })}
          </AnimatePresence>
          {!rows.length && (
            <tr>
              <td
                colSpan={H.length}
                className="px-4 py-12 text-center text-sm text-[#A9A5B8]"
                data-testid="attack-rounds-empty"
              >
                Press “Launch attack” to start streaming {meta.unit.toLowerCase()}s.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  </section>
);
