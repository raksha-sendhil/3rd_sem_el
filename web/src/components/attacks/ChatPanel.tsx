"use client";

import { motion } from "framer-motion";
import { ArrowUp, MessageSquare, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { RagAnswer } from "@/api/client";
import { seconds } from "@/lib/agents";
import { Highlighted, scan, stats, type Terms } from "@/lib/marks";
import { Verdict } from "../Verdict";
import { SIDES, type Variant } from "./sides";

export type Msg =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "assistant"; answer: RagAnswer; terms: Terms }
  | { id: number; role: "error"; text: string };

/** Types a reply out once, when it first appears, then marks its details. */
function Reveal({
  text,
  terms,
  side,
  onGrow,
}: {
  text: string;
  terms: Terms;
  side: "std" | "prot";
  onGrow: () => void;
}) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (shown >= text.length) return;
    const t = setTimeout(() => setShown((n) => Math.min(text.length, n + 3)), 16);
    onGrow();
    return () => clearTimeout(t);
  }, [shown, text, onGrow]);
  if (shown >= text.length) return <Highlighted text={text} terms={terms} side={side} />;
  return (
    <span>
      {text.slice(0, shown)}
      <span className="ml-0.5 inline-block h-4 w-[3px] translate-y-0.5 animate-pulse rounded-full bg-[#17162B]" />
    </span>
  );
}

const Typing = () => (
  <span className="flex gap-1 py-1.5" aria-label="Thinking">
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        className="h-2 w-2 rounded-full bg-[#A9A5B8]"
        animate={{ y: [0, -4, 0] }}
        transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
      />
    ))}
  </span>
);

export const ChatPanel = ({
  variant,
  messages,
  draft,
  busy,
  onDraft,
  onSend,
  onClear,
}: {
  variant: Variant;
  messages: Msg[];
  draft: string;
  busy: boolean;
  onDraft: (text: string) => void;
  onSend: () => void;
  onClear: () => void;
}) => {
  const v = SIDES[variant];
  const scroller = useRef<HTMLDivElement>(null);

  const toBottom = useRef(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }).current;
  useEffect(toBottom, [messages, busy, toBottom]);

  // distinct details this pipeline has given away in the whole conversation
  const leaked = new Set<string>();
  for (const m of messages)
    if (m.role === "assistant")
      for (const s of scan(m.answer.answer, m.terms, v.side))
        if (s.kind === "leak") leaked.add(s.text.toLowerCase());

  return (
    <section className="card flex h-[640px] flex-col overflow-hidden" data-testid={`${variant}-chat`}>
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
        <div className="flex items-center gap-2">
          {messages.some((m) => m.role === "assistant") && (
            <span className="chip bg-white" style={{ color: v.ink }} data-testid={`${variant}-leaked`}>
              {leaked.size} {leaked.size === 1 ? "detail" : "details"} leaked
            </span>
          )}
          <button
            onClick={onClear}
            disabled={busy || !messages.length}
            aria-label="New chat"
            title="New chat"
            className="grid h-8 w-8 place-items-center rounded-full bg-white/70 text-[#6B6880] transition-[background-color,color] hover:bg-white hover:text-[#17162B] disabled:opacity-40"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div ref={scroller} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
        {!messages.length && !busy && (
          <div className="grid h-full place-items-center">
            <span className="grid h-14 w-14 place-items-center rounded-2xl" style={{ background: v.bg, color: v.color }}>
              <MessageSquare className="h-6 w-6" />
            </span>
          </div>
        )}
        {messages.map((m) => {
          if (m.role === "user")
            return (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-end"
              >
                <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-[#17162B] px-4 py-2.5 text-[15px] leading-relaxed text-white">
                  {m.text}
                </div>
              </motion.div>
            );
          if (m.role === "error")
            return (
              <div
                key={m.id}
                className="rounded-2xl border border-[#FFB4AB] bg-[#FFE8E4] px-4 py-2.5 text-sm font-medium text-[#B42318]"
              >
                {m.text}
              </div>
            );
          const leaks = stats(m.answer.answer, m.terms, v.side).leaks;
          return (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3"
            >
              <span
                className="grid h-8 w-8 shrink-0 place-items-center rounded-xl text-white"
                style={{ background: v.color }}
              >
                <v.icon className="h-4 w-4" strokeWidth={2.4} />
              </span>
              <div className="min-w-0 max-w-[85%]">
                <div className="rounded-2xl rounded-tl-md border border-[#E6E1D6] bg-white px-4 py-3 text-[15px] leading-[1.8] text-[#2A2840]">
                  <Reveal text={m.answer.answer} terms={m.terms} side={v.side} onGrow={toBottom} />
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2 font-mono text-[11px] text-[#6B6880]">
                  <span className="chip" style={{ background: v.bg, color: v.ink }}>
                    {leaks} {leaks === 1 ? "detail" : "details"} leaked
                  </span>
                  {variant === "protected" && <Verdict monitor={m.answer.monitor} />}
                  <span>{seconds(m.answer.latency_ms)}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
        {busy && (
          <div className="flex gap-3">
            <span
              className="grid h-8 w-8 shrink-0 place-items-center rounded-xl text-white"
              style={{ background: v.color }}
            >
              <v.icon className="h-4 w-4" strokeWidth={2.4} />
            </span>
            <div className="rounded-2xl rounded-tl-md border border-[#E6E1D6] bg-white px-4 py-2.5">
              <Typing />
            </div>
          </div>
        )}
      </div>

      <form
        className="flex items-end gap-2 border-t border-dashed border-[#E6E1D6] p-4"
        onSubmit={(e) => {
          e.preventDefault();
          onSend();
        }}
      >
        <textarea
          value={draft}
          onChange={(e) => onDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          rows={2}
          placeholder="Message"
          aria-label={`Message ${v.name}`}
          data-testid={`${variant}-input`}
          className="min-h-[56px] flex-1 resize-none rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 text-[15px] font-medium text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:text-[#A9A5B8] focus:shadow-[0_0_0_4px_var(--ring)]"
          style={{ ["--ring" as string]: v.bg, borderColor: draft ? v.color : undefined }}
        />
        <button
          type="submit"
          disabled={busy || !draft.trim()}
          aria-label="Send"
          data-testid={`${variant}-send`}
          className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl text-white transition-[transform,box-shadow] hover:-translate-y-0.5 active:translate-y-1 disabled:opacity-40 disabled:hover:translate-y-0"
          style={{ background: v.color, boxShadow: `0 5px 0 ${v.shade}` }}
        >
          <ArrowUp className="h-5 w-5" strokeWidth={2.6} />
        </button>
      </form>
    </section>
  );
};
