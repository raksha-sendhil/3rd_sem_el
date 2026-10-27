"use client";

import { FileText, Mail, MessageSquare } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { ApiError, api, type DocumentSummary, type DocumentView } from "@/api/client";
import { ChangesTable } from "@/components/documents/ChangesTable";
import { FactsTable } from "@/components/documents/FactsTable";
import { changesOf, originalPieces, sanitisedPieces, type Piece } from "@/components/documents/changes";
import { Illustration, Legend, PageHead } from "@/components/PageHead";
import { AGENTS } from "@/lib/agents";
import { IMAGES } from "@/lib/images";
import { Mark } from "@/lib/marks";

const KINDS = {
  patient: { title: "Patient message", icon: MessageSquare, accent: "#FF4D3D" },
  email: { title: "Staff email", icon: Mail, accent: "#2D8CFF" },
  hr: { title: "HR note", icon: FileText, accent: "#0FB5A0" },
};

/** What a record is, from how its text starts. */
function kindOf(d: DocumentSummary) {
  const first = d.preview.trim().split("\n")[0];
  if (!/^(from|subject):/i.test(first)) return KINDS.patient;
  return /\b(hr|leave)\b/i.test(d.preview) ? KINDS.hr : KINDS.email;
}

const Pane = ({
  title,
  bg,
  children,
  testid,
}: {
  title: string;
  bg: string;
  children: ReactNode;
  testid: string;
}) => (
  <div className="card overflow-hidden" data-testid={testid}>
    <div className="px-5 py-3" style={{ background: bg }}>
      <h3 className="font-display text-base font-bold text-[#17162B]">{title}</h3>
    </div>
    <div className="whitespace-pre-wrap px-5 py-5 font-mono text-[13px] leading-[2] text-[#2A2840]">
      {children}
    </div>
  </div>
);

const AgentStrip = ({ counts }: { counts: Record<keyof typeof AGENTS, number> }) => (
  <div className="grid gap-3 sm:grid-cols-3" data-testid="agent-strip">
    {(Object.keys(AGENTS) as (keyof typeof AGENTS)[]).map((k) => {
      const a = AGENTS[k];
      return (
        <div key={k} className="card flex items-center gap-4 p-4" data-testid={`agent-card-${k}`}>
          <div
            className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl font-display text-xl font-extrabold text-white"
            style={{ background: a.color }}
          >
            {counts[k]}
          </div>
          <div>
            <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em]" style={{ color: a.color }}>
              {a.step}
            </div>
            <div className="font-display text-base font-bold text-[#17162B]">{a.name}</div>
          </div>
        </div>
      );
    })}
  </div>
);

export default function Documents() {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [docId, setDocId] = useState<string | null>(null);
  const [view, setView] = useState<DocumentView | null>(null);
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    api
      .documents()
      .then((list) => {
        setDocs(list);
        setDocId((cur) => cur ?? list.find((d) => d.n_changes > 0)?.doc_id ?? list[0]?.doc_id ?? null);
      })
      .catch((e) =>
        toast.error("Could not load the records", {
          description: e instanceof ApiError ? e.message : String(e),
        }),
      );
  }, []);

  useEffect(() => {
    if (!docId) return;
    let alive = true;
    api
      .document(docId)
      .then((v) => alive && setView(v))
      .catch((e) =>
        toast.error("Could not open the record", {
          description: e instanceof ApiError ? e.message : String(e),
        }),
      );
    return () => {
      alive = false;
    };
  }, [docId]);

  const pick = (d: DocumentSummary) => {
    setDocId(d.doc_id);
    setActive(null);
    toast(`Opened ${kindOf(d).title}`, { description: d.doc_id });
  };

  const changes = useMemo(() => (view ? changesOf(view) : []), [view]);

  const render = (pieces: Piece[], side: "original" | "sanitised") =>
    pieces.map((p, i) =>
      typeof p === "string" ? (
        <span key={i}>{p}</span>
      ) : (
        <Mark
          key={i}
          kind={side === "original" ? "leak" : p.kind}
          type={p.type}
          active={active === p.id}
          onHover={(on) => setActive(on ? p.id : null)}
        >
          {side === "original" ? p.original : p.replacement}
        </Mark>
      ),
    );

  return (
    <div className="space-y-8">
      <div className="grid items-center gap-6 lg:grid-cols-[1fr_1fr]">
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
        <Illustration src={IMAGES.documents} className="rise mx-auto aspect-[16/9] w-full max-w-[460px]" />
      </div>
      <div className="flex gap-3 overflow-x-auto pb-2" data-testid="document-selector">
        {docs.map((d) => {
          const k = kindOf(d);
          const on = d.doc_id === docId;
          return (
            <button
              key={d.doc_id}
              data-testid={`document-option-${d.doc_id}`}
              onClick={() => pick(d)}
              className={`flex shrink-0 items-center gap-3 rounded-2xl border-2 bg-white px-4 py-3 text-left transition-[transform,border-color] hover:-translate-y-0.5 ${
                on ? "" : "border-[#E6E1D6]"
              }`}
              style={on ? { borderColor: k.accent } : undefined}
            >
              <span
                className="grid h-9 w-9 place-items-center rounded-xl text-white"
                style={{ background: k.accent }}
              >
                <k.icon className="h-4 w-4" />
              </span>
              <span>
                <span className="block font-display text-sm font-bold text-[#17162B]">{k.title}</span>
                <span className="block font-mono text-[10px] text-[#6B6880]">
                  {d.doc_id} · {d.n_changes} {d.n_changes === 1 ? "change" : "changes"}
                </span>
              </span>
            </button>
          );
        })}
      </div>
      {view && (
        <>
          <AgentStrip
            counts={{
              detect: view.record.detector.spans.length,
              extract: view.record.extractor.facts.length,
              rewrite: changes.length,
            }}
          />
          {!!view.record.residual_leaks?.length && (
            <div
              role="alert"
              className="rounded-2xl border border-[#FFB4AB] bg-[#FFE8E4] px-5 py-3 text-sm font-medium text-[#B42318]"
            >
              {view.record.residual_leaks.length === 1
                ? "1 detail survived the rewrite and was masked"
                : `${view.record.residual_leaks.length} details survived the rewrite and were masked`}
            </div>
          )}
          <div className="grid gap-6 lg:grid-cols-2" data-testid="document-diff-viewer">
            <Pane
              title="Original"
              bg="#FFE8E4"
              testid="document-original-pane"
            >
              {render(originalPieces(view.original, changes), "original")}
            </Pane>
            <Pane
              title="Sanitised"
              bg="#DDF7F2"
              testid="document-sanitised-pane"
            >
              {render(sanitisedPieces(view.record.rewriter.safe_text, changes), "sanitised")}
            </Pane>
          </div>
          <ChangesTable docId={view.doc_id} changes={changes} active={active} onHover={setActive} />
          <FactsTable facts={view.record.extractor.facts} />
        </>
      )}
    </div>
  );
}
