"use client";

import { Database, ShieldAlert, ShieldCheck, Terminal, Timer } from "lucide-react";
import type { CompareResponse } from "@/api/client";
import { seconds } from "@/lib/agents";
import { Highlighted, stats, type Terms } from "@/lib/marks";
import { Verdict } from "../Verdict";

export type Run = { id: string; question: string; time: string; data: CompareResponse; terms: Terms };
export type Variant = "standard" | "protected";

export const V = {
  standard: {
    name: "Standard RAG",
    icon: ShieldAlert,
    color: "#FF4D3D",
    bg: "#FFE8E4",
    ink: "#B42318",
    key: "baseline",
    side: "std",
  },
  protected: {
    name: "Protected RAG",
    icon: ShieldCheck,
    color: "#0FB5A0",
    bg: "#DDF7F2",
    ink: "#08776A",
    key: "protected",
    side: "prot",
  },
} as const;

const Meter = ({ value, max, color }: { value: number; max: number; color: string }) => (
  <div className="flex gap-1">
    {Array.from({ length: max }).map((_, i) => (
      <span
        key={i}
        className="h-2 w-3 rounded-sm transition-colors duration-500"
        style={{ background: i < value ? color : "#ECE8DF" }}
      />
    ))}
  </div>
);

const Skeleton = () => (
  <div className="space-y-3 py-2">
    {[92, 78, 85, 40].map((w, i) => (
      <div key={i} className="shimmer h-4 rounded-full" style={{ width: `${w}%` }} />
    ))}
  </div>
);

export const AnswerPanel = ({
  variant,
  run,
  busy,
  onInspect,
}: {
  variant: Variant;
  run: Run | null;
  busy: boolean;
  onInspect: () => void;
}) => {
  const v = V[variant];
  const answer = run?.data[v.key] ?? null;
  const s = answer && run ? stats(answer.answer, run.terms, v.side) : null;
  const testid = `${variant}-rag-panel`;
  return (
    <section data-testid={testid} className="card flex flex-col overflow-hidden">
      <div className="flex min-h-[72px] items-center justify-between gap-3 px-5 py-4" style={{ background: v.bg }}>
        <div className="flex items-center gap-3">
          <span
            className="grid h-10 w-10 place-items-center rounded-xl text-white shadow-sm"
            style={{ background: v.color }}
          >
            <v.icon className="h-5 w-5" strokeWidth={2.4} />
          </span>
          <h2 className="font-display text-base font-bold text-[#17162B] md:text-lg">{v.name}</h2>
        </div>
        {s && !busy && answer && (
          <div className="flex flex-wrap items-center justify-end gap-2">
            {variant === "protected" && <Verdict monitor={answer.monitor} />}
            <span data-testid={`${variant}-risk-badge`} className="chip bg-white" style={{ color: v.ink }}>
              {variant === "standard"
                ? `${s.leaks} details leaked`
                : `${s.leaks} leaked · ${s.tokens} ${s.tokens === 1 ? "token" : "tokens"}`}
            </span>
          </div>
        )}
      </div>
      <div
        className="min-h-[180px] flex-1 px-5 py-5 text-[15px] leading-[1.9] text-[#2A2840] sm:text-base"
        data-testid={`${variant}-answer`}
      >
        {busy ? (
          <Skeleton />
        ) : answer && run ? (
          <Highlighted key={run.id} animate text={answer.answer} terms={run.terms} side={v.side} />
        ) : (
          <p className="text-[#A9A5B8]">Ask a question to see this pipeline&apos;s answer.</p>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-dashed border-[#E6E1D6] px-5 py-3">
        <div className="flex flex-wrap items-center gap-4 font-mono text-[11px] text-[#6B6880]">
          <span className="flex items-center gap-1">
            <Timer className="h-3.5 w-3.5" />
            {answer && !busy ? seconds(answer.latency_ms) : "—"}
          </span>
          <span className="flex items-center gap-1" title={answer?.doc_ids.join(", ")}>
            <Database className="h-3.5 w-3.5" />
            {answer && !busy && answer.doc_ids.length
              ? answer.doc_ids[0] + (answer.doc_ids.length > 1 ? ` +${answer.doc_ids.length - 1}` : "")
              : "—"}
          </span>
          <span className="flex items-center gap-2">
            exposure <Meter value={busy ? 0 : (s?.leaks ?? 0)} max={9} color={v.color} />
          </span>
        </div>
        <button
          data-testid={`${variant}-inspect-payload-button`}
          disabled={!answer || busy}
          onClick={onInspect}
          className="flex items-center gap-1.5 rounded-full border border-[#E6E1D6] bg-white px-3 py-1.5 text-xs font-semibold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-40"
        >
          <Terminal className="h-3.5 w-3.5" /> What the model saw
        </button>
      </div>
    </section>
  );
};
