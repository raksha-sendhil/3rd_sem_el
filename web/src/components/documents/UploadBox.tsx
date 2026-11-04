"use client";

import { Loader2, Play, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "@/api/client";

// the API's document size limit (settings.max_doc_chars)
export const MAX_CHARS = 2000;

export const UploadBox = ({ onRun, busy }: { onRun: (text: string) => void; busy: boolean }) => {
  const [text, setText] = useState("");
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const tooLong = text.length > MAX_CHARS;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setParseError(null);
    setParsing(true);

    try {
      const name = f.name.toLowerCase();
      const isBinary = name.endsWith(".pdf") || /\.(png|jpe?g|webp|tiff|bmp)$/.test(name);

      if (isBinary) {
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            const raw = reader.result as string;
            const base64 = raw.includes(",") ? raw.split(",")[1] : raw;
            const res = await api.ingest(base64, undefined, f.name);
            setText(res.text);
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Failed to extract text from file";
            setParseError(msg);
          } finally {
            setParsing(false);
          }
        };
        reader.onerror = () => {
          setParseError("Could not read local file");
          setParsing(false);
        };
        reader.readAsDataURL(f);
      } else {
        const content = await f.text();
        if (name.endsWith(".hl7") || name.endsWith(".xml") || name.endsWith(".fhir")) {
          try {
            const res = await api.ingest(undefined, content, f.name);
            setText(res.text);
          } catch {
            setText(content);
          }
        } else {
          setText(content);
        }
        setParsing(false);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Error reading file";
      setParseError(msg);
      setParsing(false);
    }

    e.target.value = "";
  };

  return (
    <section data-testid="upload-box" className="card relative overflow-hidden p-5 sm:p-7">
      <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-[#EEE9FF]" />
      <div className="absolute -right-2 top-16 h-16 w-16 rounded-full bg-[#FFF3D1]" />
      <div className="relative">
        <label htmlFor="doc" className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#6B6880]">
          Your document (Text, PDF, Handwritten/Printed Image, or EHR Record)
        </label>
        <textarea
          id="doc"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
          placeholder="Paste clinical text, or upload a .txt, .pdf, image (handwritten note or prescription), or EHR record (.hl7, .fhir, .xml)..."
          className="mt-3 min-h-[160px] w-full resize-y rounded-2xl border-2 border-[#E6E1D6] bg-[#FBFAF7] px-4 py-3 font-mono text-[13px] leading-[1.9] text-[#17162B] outline-none transition-[border-color,box-shadow] placeholder:font-sans placeholder:text-[#A9A5B8] focus:border-[#6D4AFF] focus:shadow-[0_0_0_4px_#EEE9FF]"
        />

        {parseError && (
          <div className="mt-2 text-xs font-medium text-red-600">
            ⚠️ {parseError}
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <input
            ref={file}
            type="file"
            accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp,.tiff,.bmp,.hl7,.fhir,.xml,text/plain,application/pdf,image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            onClick={() => file.current?.click()}
            disabled={busy || parsing}
            className="flex items-center gap-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-4 py-2.5 text-sm font-bold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-50"
          >
            {parsing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            {parsing ? "Parsing File…" : "Upload PDF / Image / EHR"}
          </button>
          <span className={`font-mono text-[11px] ${tooLong ? "font-bold text-[#B42318]" : "text-[#6B6880]"}`}>
            {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </span>
          <button
            data-testid="run-agents-button"
            onClick={() => onRun(text.trim())}
            disabled={busy || parsing || !text.trim() || tooLong}
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
