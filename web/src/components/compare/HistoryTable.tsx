"use client";

import { RotateCcw } from "lucide-react";

export type HistoryRow = {
  id: string;
  time: string;
  question: string;
  leaks: number;
  tokens: number;
  vague: number;
  std: string;
  prot: string;
};

const Pill = ({ n, bg, color }: { n: number; bg: string; color: string }) => (
  <span
    className="inline-flex min-w-[2rem] justify-center rounded-full px-2 py-0.5 font-mono text-xs font-bold"
    style={{ background: bg, color }}
  >
    {n}
  </span>
);

const HEAD = ["Time", "Question", "Leaked (std)", "Tokens (prot)", "Generalised", "Latency std / prot", ""];

export const HistoryTable = ({
  rows,
  onRerun,
  busy,
}: {
  rows: HistoryRow[];
  onRerun: (q: string) => void;
  busy: boolean;
}) => (
  <section className="card overflow-hidden" data-testid="session-history-table">
    <div className="flex items-end justify-between px-5 pb-3 pt-5">
      <div>
        <h2 className="font-display text-base font-bold text-[#17162B] md:text-lg">Session history</h2>
        <p className="font-mono text-[11px] text-[#6B6880]">
          {rows.length} {rows.length === 1 ? "run" : "runs"} this session
        </p>
      </div>
    </div>
    <div className="relative w-full overflow-x-auto">
      <table className="w-full caption-bottom text-sm">
        <thead>
          <tr className="border-b border-[#E6E1D6] bg-[#FBFAF7]">
            {HEAD.map((h) => (
              <th
                key={h}
                className="h-10 whitespace-nowrap px-2 text-left align-middle font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880] first:pl-[10px]"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              data-testid={`history-row-${r.id}`}
              className="border-b border-[#EFEBE2] transition-colors last:border-0 hover:bg-[#FBFAF7]"
            >
              <td className="p-2 pl-[10px] align-middle font-mono text-xs text-[#6B6880]">{r.time}</td>
              <td className="min-w-[260px] max-w-[420px] p-2 align-middle text-sm font-medium text-[#17162B]">
                {r.question}
              </td>
              <td className="p-2 align-middle">
                <Pill n={r.leaks} bg="#FFE8E4" color="#B42318" />
              </td>
              <td className="p-2 align-middle">
                <Pill n={r.tokens} bg="#EEE9FF" color="#4B2FD6" />
              </td>
              <td className="p-2 align-middle">
                <Pill n={r.vague} bg="#FFF3D1" color="#8A5A00" />
              </td>
              <td className="whitespace-nowrap p-2 align-middle font-mono text-xs text-[#4A4760]">
                {r.std} / {r.prot}
              </td>
              <td className="p-2 align-middle">
                <button
                  data-testid={`history-rerun-${r.id}`}
                  onClick={() => onRerun(r.question)}
                  disabled={busy}
                  className="grid h-8 w-8 place-items-center rounded-full text-[#6B6880] transition-[background-color,color] hover:bg-[#EEE9FF] hover:text-[#4B2FD6] disabled:opacity-40"
                  aria-label="Run again"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </td>
            </tr>
          ))}
          {!rows.length && (
            <tr>
              <td colSpan={HEAD.length} className="px-4 py-10 text-center text-sm text-[#A9A5B8]">
                Runs you make appear here.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  </section>
);
