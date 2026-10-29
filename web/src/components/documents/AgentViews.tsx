"use client";

import { Loader2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { FactTuple, RewriterOutput, SensitiveSpan } from "@/api/client";
import { AGENTS, KIND } from "@/lib/agents";
import { Tip } from "@/lib/marks";
import { detectedPieces, rewrittenPieces, subjectLabel } from "./pieces";

const LEVEL = { high: "#FF4D3D", medium: "#F5A300", low: "#CFC8B8" };

const Shimmer = () => (
  <div className="space-y-3 py-1">
    {[92, 78, 85, 64, 40].map((w, i) => (
      <div key={i} className="shimmer h-4 rounded-full" style={{ width: `${w}%` }} />
    ))}
  </div>
);

const AgentCard = ({
  agent,
  count,
  running,
  children,
}: {
  agent: keyof typeof AGENTS;
  count: number | null; // null while this agent has not run yet
  running: boolean;
  children: ReactNode;
}) => {
  const a = AGENTS[agent];
  return (
    <section className="card flex min-w-0 flex-col overflow-hidden" data-testid={`agent-${agent}`}>
      <div className="flex items-center gap-4 border-b border-[#E6E1D6] px-5 py-4" style={{ background: a.bg }}>
        <div
          className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl font-display text-xl font-extrabold text-white"
          style={{ background: a.color }}
        >
          {running ? <Loader2 className="h-5 w-5 animate-spin" /> : (count ?? "–")}
        </div>
        <div>
          <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em]" style={{ color: a.color }}>
            {a.step}
          </div>
          <div className="font-display text-base font-bold text-[#17162B]">{a.name}</div>
        </div>
      </div>
      <div className="flex-1 px-5 py-5">{children}</div>
    </section>
  );
};

/** One document through the three agents: what Agent 1 found, the facts Agent 2 pulled out, and
 * Agent 3's rewrite. Parts not yet run are null; ``running`` is the agent working right now. */
export function AgentViews({
  original,
  spans,
  facts,
  rewriter,
  residual = [],
  running = null,
}: {
  original: string;
  spans: SensitiveSpan[] | null;
  facts: FactTuple[] | null;
  rewriter: RewriterOutput | null;
  residual?: SensitiveSpan[];
  running?: 1 | 2 | 3 | null;
}) {
  const [active, setActive] = useState<string | null>(null);
  const hover = (key: string) => ({
    onMouseEnter: () => setActive(key),
    onMouseLeave: () => setActive(null),
  });
  const decisions = rewriter?.decisions ?? null;
  const combos = (spans ?? []).filter((s) => s.start == null);
  const subjects = [...new Set((facts ?? []).map((f) => f.subject))];
  const changed = decisions?.filter((d) => d.action !== "keep").length ?? null;

  return (
    <div className="grid items-start gap-6 xl:grid-cols-3">
      <AgentCard agent="detect" count={spans?.length ?? null} running={running === 1}>
        {spans ? (
          <>
            <p className="whitespace-pre-wrap font-mono text-[13px] leading-[2] text-[#2A2840]">
              {detectedPieces(original, spans, decisions).map((p, i) =>
                typeof p === "string" ? (
                  <span key={i}>{p}</span>
                ) : (
                  <Tip
                    key={i}
                    content={
                      <span>
                        <b className="text-[#2D8CFF]">{p.type}</b> · {p.how}
                        {p.kept && " · kept"}
                      </span>
                    }
                  >
                    <span
                      {...hover(p.key)}
                      className={`${p.kept ? "mark-kept" : "mark-leak"} ${active === p.key ? "mark-active" : ""}`}
                    >
                      {p.text}
                    </span>
                  </Tip>
                ),
              )}
            </p>
            {combos.length > 0 && (
              <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-dashed border-[#E6E1D6] pt-4">
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#6B6880]">
                  Combinations
                </span>
                {combos.map((c, i) => (
                  <span key={i} className="mark-leak text-xs">
                    {c.text}
                  </span>
                ))}
              </div>
            )}
          </>
        ) : (
          <Shimmer />
        )}
      </AgentCard>

      <AgentCard agent="extract" count={facts?.length ?? null} running={running === 2}>
        {facts ? (
          <div className="space-y-5">
            {subjects.map((subject) => (
              <div key={subject}>
                <div className="mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#6D4AFF]">
                  {subjectLabel(subject)}
                </div>
                <ul className="space-y-2">
                  {facts
                    .filter((f) => f.subject === subject)
                    .map((f, i) => (
                      <li key={i} className="flex items-baseline gap-2.5 text-sm leading-relaxed">
                        <span
                          className="mt-1.5 h-2 w-2 shrink-0 translate-y-[-1px] rounded-full"
                          style={{ background: LEVEL[f.importance] }}
                          title={`${f.importance} importance`}
                        />
                        <span>
                          <span className="font-mono text-xs text-[#6B6880]">{f.relation.replaceAll("_", " ")}</span>{" "}
                          <span className="font-medium text-[#17162B]">{f.value}</span>
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
            ))}
            {!facts.length && <p className="text-sm text-[#A9A5B8]">No facts</p>}
          </div>
        ) : (
          <Shimmer />
        )}
      </AgentCard>

      <AgentCard agent="rewrite" count={changed} running={running === 3}>
        {rewriter ? (
          <>
            {residual.length > 0 && (
              <div className="mb-4 rounded-2xl border border-[#FFB4AB] bg-[#FFE8E4] px-4 py-2.5 text-sm font-medium text-[#B42318]">
                {residual.length === 1
                  ? "1 detail survived the rewrite and was masked"
                  : `${residual.length} details survived the rewrite and were masked`}
              </div>
            )}
            <p className="whitespace-pre-wrap font-mono text-[13px] leading-[2] text-[#2A2840]">
              {rewrittenPieces(rewriter.safe_text, rewriter.decisions).map((p, i) =>
                typeof p === "string" ? (
                  <span key={i}>{p}</span>
                ) : (
                  <Tip
                    key={i}
                    content={
                      <span>
                        <b style={{ color: KIND[p.kind].color }}>{KIND[p.kind].label}</b> · {p.type}
                      </span>
                    }
                  >
                    <span
                      {...hover(p.key)}
                      className={`${p.kind === "token" ? "mark-token" : "mark-vague"} ${active === p.key ? "mark-active" : ""}`}
                    >
                      {p.text}
                    </span>
                  </Tip>
                ),
              )}
            </p>
          </>
        ) : (
          <Shimmer />
        )}
      </AgentCard>
    </div>
  );
}
