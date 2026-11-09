"use client";

import { FileScan, FileText, HeartPulse, Loader2, Play, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import type { IngestResult, PageResult } from "@/api/client";
import { MAX_CHARS } from "./UploadBox";

/** A file the user chose and what the API read from it. */
export type Upload = {
  file: File;
  url: string; // object URL of the file, for the preview
  raw: string | null; // the file as text, for health records (shown as they are)
  result: IngestResult;
};

const METHOD: Record<PageResult["method"], { label: string; cls: string }> = {
  text: { label: "text layer, read directly", cls: "bg-[#E3F7F3] text-[#0B7F70]" },
  vision: { label: "read by the vision model", cls: "bg-[#EEE9FF] text-[#4B2FD6]" },
  plain: { label: "text file", cls: "bg-[#E3F7F3] text-[#0B7F70]" },
  electronic: { label: "health record, written out", cls: "bg-[#FFF3D1] text-[#8A5A00]" },
  failed: { label: "could not be read", cls: "bg-[#FFE8E4] text-[#B42318]" },
};

/** "scanned PDF, 3 pages", "HL7 health record", ... */
function describe(r: IngestResult): string {
  const n = r.pages.length;
  const pages = `${n} ${n === 1 ? "page" : "pages"}`;
  if (r.kind === "ehr") return `${r.source_format.toUpperCase()} health record`;
  if (r.kind === "image") return n > 1 ? `image, ${pages}` : "photo or scan";
  if (r.kind === "text") return "text file";
  const vision = r.pages.filter((p) => p.method === "vision").length;
  const kind = vision === 0 ? "typed PDF" : vision === n ? "scanned PDF" : "mixed PDF (typed and scanned pages)";
  return `${kind}, ${pages}`;
}

const Label = ({ children }: { children: React.ReactNode }) => (
  <div className="mb-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#6B6880]">{children}</div>
);

/** Step 0: the file next to what was read from it, page by page; then the text the agents get
 * (editable, part by part for a long file) and the button that runs them. */
export function ReadView({ upload, busy, onRun }: { upload: Upload; busy: boolean; onRun: (text: string) => void }) {
  const { file, url, raw, result } = upload;
  const [parts, setParts] = useState<string[]>(result.parts.length ? result.parts : [""]);
  const [part, setPart] = useState(0);
  useEffect(() => {
    setParts(result.parts.length ? result.parts : [""]);
    setPart(0);
  }, [result]);
  const text = parts[part] ?? "";
  const tooLong = text.length > MAX_CHARS;
  const removed = result.removed_lines ?? [];
  const warnings = result.warnings ?? [];

  return (
    <section className="card overflow-hidden" data-testid="read-view">
      <div className="flex flex-wrap items-center gap-4 border-b border-[#E6E1D6] bg-[#F1EFE9] px-5 py-4">
        <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-[#17162B] font-display text-xl font-extrabold text-white">
          0
        </div>
        <div className="min-w-0">
          <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#17162B]">
            Step 0 · read the file
          </div>
          <div className="truncate font-display text-base font-bold text-[#17162B]">
            {file.name} <span className="font-sans text-sm font-medium text-[#6B6880]">· {describe(result)}</span>
          </div>
        </div>
        <div className="ml-auto font-mono text-[11px] text-[#6B6880]">
          {result.char_count.toLocaleString()} characters · {result.seconds.toFixed(1)} s
        </div>
      </div>

      <div className="grid gap-5 px-5 py-5 lg:grid-cols-2">
        <div className="min-w-0">
          <Label>The file</Label>
          {result.kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={url}
              alt={file.name}
              className="max-h-[560px] w-full rounded-xl border border-[#E6E1D6] bg-[#FBFAF7] object-contain"
            />
          ) : result.kind === "pdf" ? (
            <iframe src={url} title={file.name} className="h-[560px] w-full rounded-xl border border-[#E6E1D6] bg-white" />
          ) : (
            <pre className="max-h-[560px] overflow-auto whitespace-pre-wrap rounded-xl border border-[#E6E1D6] bg-[#FBFAF7] p-4 font-mono text-[12px] leading-relaxed text-[#17162B]">
              {raw ?? "(no preview)"}
            </pre>
          )}
        </div>

        <div className="min-w-0">
          <Label>What was read</Label>
          <div className="max-h-[560px] space-y-3 overflow-auto pr-1" data-testid="read-pages">
            {result.pages.map((p) => {
              const m = METHOD[p.method] ?? METHOD.failed;
              const Icon = p.method === "vision" ? FileScan : p.method === "electronic" ? HeartPulse : FileText;
              return (
                <div key={p.index} className="rounded-xl border border-[#E6E1D6] bg-white">
                  <div className="flex flex-wrap items-center gap-2 border-b border-[#EFECE4] px-3 py-2">
                    <span className="font-mono text-[11px] font-bold text-[#17162B]">
                      {result.pages.length > 1 ? `Page ${p.index + 1}` : "Content"}
                    </span>
                    <span className={`flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[10px] ${m.cls}`}>
                      <Icon className="h-3 w-3" /> {m.label}
                    </span>
                    {p.seconds >= 0.5 && <span className="font-mono text-[10px] text-[#6B6880]">{p.seconds.toFixed(0)} s</span>}
                    {p.note && p.method !== "electronic" && (
                      <span className="font-mono text-[10px] text-[#6B6880]">({p.note})</span>
                    )}
                  </div>
                  <pre className="whitespace-pre-wrap px-3 py-2 font-mono text-[12px] leading-relaxed text-[#17162B]">
                    {p.text || "(nothing read)"}
                  </pre>
                </div>
              );
            })}
          </div>
          {removed.length > 0 && (
            <details className="mt-3 font-mono text-[11px] text-[#6B6880]">
              <summary className="cursor-pointer">
                {removed.length} repeated header/footer {removed.length === 1 ? "line" : "lines"} removed after their
                first page
              </summary>
              <ul className="mt-1 space-y-0.5 pl-4">
                {removed.map((r, i) => (
                  <li key={i}>
                    page {r.page + 1}: {r.line}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {warnings.map((w) => (
            <p key={w} className="mt-2 flex items-start gap-1.5 font-mono text-[11px] text-[#9A6700]">
              <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" /> {w}
            </p>
          ))}
        </div>
      </div>

      <div className="border-t border-[#E6E1D6] px-5 py-5">
        <div className="flex flex-wrap items-center gap-3">
          <Label>Text the agents will get · check it and correct any misreadings</Label>
          {parts.length > 1 && (
            <div className="mb-2 ml-auto flex items-center gap-2">
              <span className="font-mono text-[11px] text-[#6B6880]">Part</span>
              {parts.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setPart(i)}
                  disabled={busy}
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
        <textarea
          value={text}
          onChange={(e) => setParts((ps) => ps.map((x, i) => (i === part ? e.target.value : x)))}
          rows={8}
          data-testid="read-text"
          className="w-full resize-y rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-mono text-[13px] leading-[1.9] text-[#17162B] outline-none focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <span className={`font-mono text-[11px] ${tooLong ? "font-bold text-[#B42318]" : "text-[#6B6880]"}`}>
            {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            {parts.length > 1 ? ` · part ${part + 1} of ${parts.length}` : ""}
          </span>
          <button
            data-testid="run-agents-on-file"
            onClick={() => onRun(text.trim())}
            disabled={busy || !text.trim() || tooLong}
            className="btn-primary ml-auto flex items-center gap-2 rounded-2xl px-6 py-3 text-sm"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {busy ? "Running…" : parts.length > 1 ? `Run the three agents on part ${part + 1}` : "Run the three agents"}
          </button>
        </div>
      </div>
    </section>
  );
}
