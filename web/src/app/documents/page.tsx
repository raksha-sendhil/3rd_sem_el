"use client";

import { FileText, Mail, MessageSquare, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type DocumentSummary,
  type DocumentView,
  type FactTuple,
  type RewriterOutput,
  type SensitiveSpan,
} from "@/api/client";
import { AgentViews } from "@/components/documents/AgentViews";
import { ReadView, type Upload as FileUpload } from "@/components/documents/ReadView";
import { UploadBox } from "@/components/documents/UploadBox";
import { ErrorNote, Legend, PageHead } from "@/components/PageHead";
import { useDomain } from "@/components/Providers";

const KINDS = {
  patient: { title: "Patient message", icon: MessageSquare, accent: "#FF4D3D" },
  email: { title: "Staff email", icon: Mail, accent: "#2D8CFF" },
  hr: { title: "HR note", icon: FileText, accent: "#0FB5A0" },
};
const OWN = "own"; // the selector entry for an uploaded document

// the legal corpus: fictional client files, named by how each starts
const LEGAL_KINDS = {
  intake: { title: "Intake note", icon: FileText, accent: "#6D4AFF" },
  case: { title: "Case note", icon: FileText, accent: "#0FB5A0" },
  email: { title: "Advice email", icon: Mail, accent: "#2D8CFF" },
  client: { title: "Client file", icon: MessageSquare, accent: "#FF4D3D" },
};

/** What a record is, from how its text starts. */
function kindOf(d: DocumentSummary, legal = false) {
  const first = d.preview.trim().split("\n")[0];
  if (legal) {
    if (/^intake note/i.test(first)) return LEGAL_KINDS.intake;
    if (/^case note/i.test(first)) return LEGAL_KINDS.case;
    if (/^(from|to|subject|dear)\b/i.test(first)) return LEGAL_KINDS.email;
    return LEGAL_KINDS.client;
  }
  if (!/^(from|subject):/i.test(first)) return KINDS.patient;
  return /\b(hr|leave)\b/i.test(d.preview) ? KINDS.hr : KINDS.email;
}

const RECORD_FILE = /\.(hl7|fhir|xml|json)$/i; // health records: shown as they are next to what was read

/** The file's bytes as base64 (what POST /v1/ingest takes). */
function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** An uploaded document going through the agents one at a time. */
type Live = {
  text: string;
  spans: SensitiveSpan[] | null;
  facts: FactTuple[] | null;
  rewriter: RewriterOutput | null;
  running: 1 | 2 | 3 | null;
};

const Option = ({
  on,
  accent,
  icon: Icon,
  title,
  sub,
  onClick,
  testid,
}: {
  on: boolean;
  accent: string;
  icon: typeof FileText;
  title: string;
  sub: string;
  onClick: () => void;
  testid: string;
}) => (
  <button
    data-testid={testid}
    onClick={onClick}
    className={`flex shrink-0 items-center gap-3 rounded-2xl border-2 bg-white px-4 py-3 text-left transition-[transform,border-color] hover:-translate-y-0.5 ${
      on ? "" : "border-[#E6E1D6]"
    }`}
    style={on ? { borderColor: accent } : undefined}
  >
    <span className="grid h-9 w-9 place-items-center rounded-xl text-white" style={{ background: accent }}>
      <Icon className="h-4 w-4" />
    </span>
    <span>
      <span className="block font-display text-sm font-bold text-[#17162B]">{title}</span>
      <span className="block font-mono text-[10px] text-[#6B6880]">{sub}</span>
    </span>
  </button>
);

export default function Documents() {
  const { domain } = useDomain();
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [docId, setDocId] = useState<string | null>(null);
  const [view, setView] = useState<DocumentView | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [error, setError] = useState<string | null>(null);
  // a chosen file: sent to /v1/ingest, then shown as step 0 before the agents
  const [upload, setUpload] = useState<FileUpload | null>(null);
  const [reading, setReading] = useState<string | null>(null);
  const [readError, setReadError] = useState<string | null>(null);
  const previewUrl = useRef<string | null>(null);

  const clearUpload = () => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = null;
    setUpload(null);
    setReadError(null);
    setLive(null);
  };

  const readFile = async (file: File) => {
    clearUpload();
    setReading(file.name);
    try {
      const raw = RECORD_FILE.test(file.name) ? await file.text() : null;
      const result = await api.ingest(file.name, await toBase64(file));
      const url = URL.createObjectURL(file);
      previewUrl.current = url;
      setUpload({ file, url, raw, result });
    } catch (e) {
      setReadError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setReading(null);
    }
  };

  useEffect(() => () => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
  }, []);

  useEffect(() => {
    api
      .documents()
      .then((list) => {
        setDocs(list);
        setDocId((cur) => cur ?? list.find((d) => d.n_changes > 0)?.doc_id ?? list[0]?.doc_id ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!docId || docId === OWN) return;
    let alive = true;
    setView(null);
    api
      .document(docId)
      .then((v) => alive && setView(v))
      .catch((e) => alive && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [docId]);

  // Agents 1, 2 and 3 as separate calls, so each panel fills in as its agent finishes
  const run = async (text: string) => {
    setError(null);
    setLive({ text, spans: null, facts: null, rewriter: null, running: 1 });
    try {
      const detected = await api.detect(text);
      const spans = detected.output.spans;
      setLive((l) => l && { ...l, spans, running: 2 });
      const { facts } = await api.extract(detected.doc_id, text, spans);
      setLive((l) => l && { ...l, facts, running: 3 });
      const rewriter = await api.rewrite(detected.doc_id, spans, facts);
      setLive((l) => l && { ...l, rewriter, running: null });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setLive((l) => l && { ...l, running: null });
    }
  };

  const own = docId === OWN;
  const running = live?.running != null;

  return (
    <div className="space-y-8">
      <div>
        <PageHead
          index="03"
          label="Documents"
          color="#0FB5A0"
          title={
            <>
              Before and after
              <br />
              <span className="text-[#0FB5A0]">the three agents.</span>
            </>
          }
        >
          <Legend />
        </PageHead>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-2" data-testid="document-selector">
        <Option
          testid="document-option-own"
          on={own}
          accent="#6D4AFF"
          icon={Upload}
          title="Your own"
          sub="upload or paste"
          onClick={() => !running && setDocId(OWN)}
        />
        {docs.map((d) => {
          const k = kindOf(d, domain.id === "legal");
          return (
            <Option
              key={d.doc_id}
              testid={`document-option-${d.doc_id}`}
              on={d.doc_id === docId}
              accent={k.accent}
              icon={k.icon}
              title={k.title}
              sub={`${d.doc_id} · ${d.n_changes} ${d.n_changes === 1 ? "change" : "changes"}`}
              onClick={() => !running && setDocId(d.doc_id)}
            />
          );
        })}
      </div>

      <ErrorNote error={error} />

      {own ? (
        <>
          <UploadBox
            onRun={(text) => {
              clearUpload();
              void run(text);
            }}
            onFile={(file) => void readFile(file)}
            onClear={clearUpload}
            busy={running}
            reading={reading}
            fileName={upload?.file.name ?? null}
            error={readError}
          />
          {upload && <ReadView upload={upload} busy={running} onRun={run} />}
          {live && (
            <AgentViews
              original={live.text}
              spans={live.spans}
              facts={live.facts}
              rewriter={live.rewriter}
              running={live.running}
            />
          )}
        </>
      ) : (
        view && (
          <AgentViews
            key={view.doc_id}
            original={view.original}
            spans={view.record.detector.spans}
            facts={view.record.extractor.facts}
            rewriter={view.record.rewriter}
            residual={view.record.residual_leaks ?? []}
          />
        )
      )}
    </div>
  );
}
