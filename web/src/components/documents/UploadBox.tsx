"use client";

import { FileScan, FileText, Loader2, Play, TriangleAlert, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { ApiError, api, type IngestResult } from "@/api/client";

// the API's document size limit (settings.max_doc_chars)
export const MAX_CHARS = 2000;

const TEXT_FILE = /\.(txt|md)$/i;

/** The file's bytes as base64 (what POST /v1/ingest takes). */
function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

const METHOD: Record<string, string> = {
  text: "text layer",
  vision: "read by vision model",
  plain: "text file",
  failed: "could not be read",
};

export const UploadBox = ({ onRun, busy }: { onRun: (text: string) => void; busy: boolean }) => {
  // one entry per part: a long PDF is cut into parts the agents accept; each part is editable
  const [parts, setParts] = useState<string[]>([""]);
  const [part, setPart] = useState(0);
  const [read, setRead] = useState<IngestResult | null>(null);
  const [reading, setReading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const text = parts[part] ?? "";
  const removed = read?.removed_lines ?? [];
  const warnings = read?.warnings ?? [];
  const tooLong = text.length > MAX_CHARS;
  const working = busy || reading !== null;

  const setText = (value: string) => setParts((ps) => ps.map((p, i) => (i === part ? value : p)));

  const choose = async (f: File) => {
    setError(null);
    if (TEXT_FILE.test(f.name) || f.type === "text/plain") {
      setRead(null);
      setParts([await f.text()]);
      setPart(0);
      return;
    }
    setReading(f.name);
    try {
      const result = await api.ingest(f.name, await toBase64(f));
      setRead(result);
      setParts(result.parts.length ? result.parts : [""]);
      setPart(0);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setReading(null);
    }
  };

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
          placeholder="Paste a message or note, or choose a file: text, PDF (typed or scanned) or a photo"
          className="mt-3 min-h-[160px] w-full resize-y rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-mono text-[13px] leading-[1.9] text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:font-sans placeholder:text-[#A9A5B8] focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
        />

        {read && (
          <div data-testid="ingest-summary" className="mt-3 space-y-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-4 py-3">
            <p className="text-sm text-[#17162B]">
              <span className="font-bold">{read.filename}</span>
              <span className="text-[#6B6880]">
                {" "}
                · {read.pages.length} {read.pages.length === 1 ? "page" : "pages"} · check the text before running the
                agents
              </span>
            </p>
            <div className="flex flex-wrap gap-2">
              {read.pages.map((p) => (
                <span
                  key={p.index}
                  title={p.note ?? undefined}
                  className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[10px] ${
                    p.method === "vision"
                      ? "bg-[#EEE9FF] text-[#4B2FD6]"
                      : p.method === "failed"
                        ? "bg-[#FFE8E4] text-[#B42318]"
                        : "bg-[#E3F7F3] text-[#0B7F70]"
                  }`}
                >
                  {p.method === "vision" ? <FileScan className="h-3 w-3" /> : <FileText className="h-3 w-3" />}
                  page {p.index + 1}: {METHOD[p.method] ?? p.method}
                  {p.method === "vision" ? ` · ${p.seconds.toFixed(0)} s` : ""}
                </span>
              ))}
            </div>
            {removed.length > 0 && (
              <p className="font-mono text-[11px] text-[#6B6880]">
                {removed.length} repeated header/footer {removed.length === 1 ? "line" : "lines"}{" "}
                removed after their first page
              </p>
            )}
            {warnings.map((w) => (
              <p key={w} className="flex items-start gap-1.5 font-mono text-[11px] text-[#9A6700]">
                <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" /> {w}
              </p>
            ))}
            {parts.length > 1 && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="font-mono text-[11px] text-[#6B6880]">Part</span>
                {parts.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setPart(i)}
                    disabled={working}
                    className={`rounded-xl border-2 px-3 py-1 font-mono text-xs font-bold ${
                      i === part ? "border-[#6D4AFF] text-[#6D4AFF]" : "border-[#E6E1D6] text-[#17162B]"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {error && (
          <p className="mt-3 flex items-start gap-1.5 font-mono text-[11px] text-[#B42318]">
            <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" /> {error}
          </p>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <input
            ref={file}
            type="file"
            accept=".txt,.md,text/plain,.pdf,application/pdf,image/*,.heic,.tif,.tiff"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void choose(f);
              e.target.value = "";
            }}
          />
          <button
            onClick={() => file.current?.click()}
            disabled={working}
            className="flex items-center gap-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-4 py-2.5 text-sm font-bold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-50"
          >
            {reading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            {reading ? `Reading ${reading}…` : "Choose file"}
          </button>
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
      </div>
    </section>
  );
};
