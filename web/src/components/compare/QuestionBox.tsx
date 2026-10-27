"use client";

import { ArrowUpRight, Loader2, Sparkles } from "lucide-react";
import { useState } from "react";

export const EXAMPLES = [
  "Which patient has vertigo, and how do we reach her?",
  "What is the kabaddi player's name and age?",
  "Who studies at IIT Madras?",
];

export const QuestionBox = ({ onRun, busy }: { onRun: (q: string) => void; busy: boolean }) => {
  const [q, setQ] = useState(EXAMPLES[0]);
  const submit = (text: string) => text.trim() && onRun(text.trim());
  return (
    <section data-testid="question-input-container" className="card relative overflow-hidden p-5 sm:p-7">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-[#EEE9FF]" />
      <div className="absolute -right-2 top-16 h-16 w-16 rounded-full bg-[#FFF3D1]" />
      <div className="relative">
        <label
          htmlFor="q"
          className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#6B6880]"
        >
          Ask both pipelines
        </label>
        <div className="mt-3 flex flex-col gap-3 lg:flex-row">
          <textarea
            id="q"
            data-testid="question-input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(q);
            }}
            rows={2}
            className="min-h-[72px] flex-1 resize-none rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-display text-lg font-semibold text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:text-[#A9A5B8] focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
            placeholder="Ask something about the private records…"
          />
          <button
            data-testid="run-compare-button"
            disabled={busy}
            onClick={() => submit(q)}
            className="btn-primary flex items-center justify-center gap-2 rounded-2xl px-7 py-4 text-base lg:self-stretch"
          >
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <ArrowUpRight className="h-5 w-5" />}
            {busy ? "Running…" : "Run both"}
          </button>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1 text-xs font-semibold text-[#6B6880]">
            <Sparkles className="h-3.5 w-3.5 text-[#F5A300]" /> Try
          </span>
          {EXAMPLES.map((e) => (
            <button
              key={e}
              onClick={() => {
                setQ(e);
                submit(e);
              }}
              disabled={busy}
              className="rounded-full border border-[#E6E1D6] bg-white px-3 py-1.5 text-left text-xs font-medium text-[#4A4760] transition-[border-color,color,transform] hover:-translate-y-0.5 hover:border-[#6D4AFF] hover:text-[#4B2FD6] disabled:opacity-50"
            >
              {e}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};
