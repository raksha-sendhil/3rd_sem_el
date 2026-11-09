"use client";

import { FileUp, Loader2, Play, TriangleAlert, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { useDomain } from "@/components/Providers";

// the API's document size limit (settings.max_doc_chars)
export const MAX_CHARS = 2000;

const TEXT_FILE = /\.(txt|md)$/i;

/** The input: paste a note and run the agents on it, or choose a file (PDF, photo, scan, health
 * record), which the page sends to the API and shows in the "read the file" step below. */
export const UploadBox = ({
  onRun,
  onFile,
  onClear,
  busy,
  reading,
  fileName,
  error,
}: {
  onRun: (text: string) => void;
  onFile: (file: File) => void;
  onClear: () => void;
  busy: boolean;
  reading: string | null; // name of the file being read
  fileName: string | null; // the file shown below, if any
  error: string | null;
}) => {
  const { domain } = useDomain();
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const tooLong = text.length > MAX_CHARS;
  const working = busy || reading !== null;

  const choose = async (f: File) => {
    if (TEXT_FILE.test(f.name) || f.type === "text/plain") {
      onClear();
      setText(await f.text());
      return;
    }
    setText("");
    onFile(f);
  };

  const fileInput = (
    <input
      ref={input}
      type="file"
      accept={domain.accept}
      className="hidden"
      data-testid="file-input"
      onChange={(e) => {
        const f = e.target.files?.[0];
        if (f) void choose(f);
        e.target.value = "";
      }}
    />
  );
  const chooseButton = (
    <button
      onClick={() => input.current?.click()}
      disabled={working}
      className="flex items-center gap-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-4 py-2.5 text-sm font-bold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-50"
    >
      {reading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
      {reading ? `Reading ${reading}…` : fileName ? "Choose another file" : "Choose file"}
    </button>
  );

  return (
    <section data-testid="upload-box" className="card relative overflow-hidden p-5 sm:p-7">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-[#EEE9FF]" />
      <div className="absolute -right-2 top-16 h-16 w-16 rounded-full bg-[#FFF3D1]" />
      <div className="relative">
        <label htmlFor="doc" className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#6B6880]">
          Your document · {domain.label}
        </label>

        {fileName || reading ? (
          <div className="mt-3 flex flex-wrap items-center gap-3 rounded-2xl border-2 border-dashed border-[#E6E1D6] bg-[#FBFAF7] px-4 py-4">
            <FileUp className="h-5 w-5 text-[#6D4AFF]" />
            <span className="text-sm text-[#17162B]">
              {reading ? (
                <>
                  Reading <b>{reading}</b>: typed pages take a moment, scanned pages and photos about half a minute each
                </>
              ) : (
                <>
                  <b>{fileName}</b> was read: see the file and its text below
                </>
              )}
            </span>
            {fileInput}
            <div className="ml-auto flex gap-2">
              {chooseButton}
              <button
                onClick={onClear}
                disabled={working}
                className="flex items-center gap-1.5 rounded-2xl border-2 border-[#E6E1D6] bg-white px-3 py-2.5 text-sm font-bold text-[#4A4760] hover:border-[#17162B] disabled:opacity-50"
              >
                <X className="h-4 w-4" /> Paste text instead
              </button>
            </div>
          </div>
        ) : (
          <>
            <textarea
              id="doc"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              placeholder={domain.placeholder}
              className="mt-3 min-h-[160px] w-full resize-y rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-mono text-[13px] leading-[1.9] text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:font-sans placeholder:text-[#A9A5B8] focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
            />
            <div className="mt-4 flex flex-wrap items-center gap-3">
              {fileInput}
              {chooseButton}
              <span className={`font-mono text-[11px] ${tooLong ? "font-bold text-[#B42318]" : "text-[#6B6880]"}`}>
                {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              </span>
              <button
                data-testid="run-agents-button"
                onClick={() => onRun(text.trim())}
                disabled={working || !text.trim() || tooLong}
                className="btn-primary ml-auto flex items-center gap-2 rounded-2xl px-6 py-3 text-sm"
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {busy ? "Running…" : "Run the three agents"}
              </button>
            </div>
          </>
        )}

        {error && (
          <p className="mt-3 flex items-start gap-1.5 font-mono text-[11px] text-[#B42318]">
            <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" /> {error}
          </p>
        )}
      </div>
    </section>
  );
};
