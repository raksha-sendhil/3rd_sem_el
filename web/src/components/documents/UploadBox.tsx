"use client";

import { Loader2, Play, Upload } from "lucide-react";
import { useRef, useState } from "react";

// the API's document size limit (settings.max_doc_chars)
export const MAX_CHARS = 2000;

export const UploadBox = ({ onRun, busy }: { onRun: (text: string) => void; busy: boolean }) => {
  const [text, setText] = useState("");
  const file = useRef<HTMLInputElement>(null);
  const tooLong = text.length > MAX_CHARS;
  return (
    <section data-testid="upload-box" className="card relative overflow-hidden p-5 sm:p-7">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-[#EEE9FF]" />
      <div className="absolute -right-2 top-16 h-16 w-16 rounded-full bg-[#FFF3D1]" />
      <div className="relative">
        <label htmlFor="doc" className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#6B6880]">
          Your document
        </label>
        <textarea
          id="doc"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
          placeholder="Paste a message or note, or choose a .txt file"
          className="mt-3 min-h-[160px] w-full resize-y rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-mono text-[13px] leading-[1.9] text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:font-sans placeholder:text-[#A9A5B8] focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
        />
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <input
            ref={file}
            type="file"
            accept=".txt,.md,text/plain"
            className="hidden"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (f) setText(await f.text());
              e.target.value = "";
            }}
          />
          <button
            onClick={() => file.current?.click()}
            disabled={busy}
            className="flex items-center gap-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-4 py-2.5 text-sm font-bold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-50"
          >
            <Upload className="h-4 w-4" /> Choose file
          </button>
          <span className={`font-mono text-[11px] ${tooLong ? "font-bold text-[#B42318]" : "text-[#6B6880]"}`}>
            {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </span>
          <button
            data-testid="run-agents-button"
            onClick={() => onRun(text.trim())}
            disabled={busy || !text.trim() || tooLong}
            className="btn-primary ml-auto flex items-center gap-2 rounded-2xl px-6 py-3 text-sm"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {busy ? "Running…" : "Run the three agents"}
          </button>
        </div>
      </div>
    </section>
  );
};
