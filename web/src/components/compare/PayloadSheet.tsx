"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Copy, X } from "lucide-react";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { Highlighted } from "@/lib/marks";
import { V, type Run, type Variant } from "./AnswerPanel";

// Copy of SYSTEM_PROMPT in src/sanitise/rag/answer_prompt.py; the API sends only the user message.
const SYSTEM_PROMPT = `You answer questions about a collection of records (patients' messages to doctors and workplace emails), using only the records under CONTEXT.
- Answer in one to three sentences.
- State details exactly as the records give them, including tokens such as [PERSON_1].
- If the records do not contain the answer, reply: "The records do not say."
- Do not guess or add anything that is not in the records.`;

/** The user message is "CONTEXT:\n[id]\ntext\n\n[id]\ntext\n\nQUESTION: q\nANSWER:". */
function split(payload: string, docIds: string[]) {
  const q = payload.lastIndexOf("\n\nQUESTION: ");
  const body = q >= 0 ? payload.slice(0, q).replace(/^CONTEXT:\n/, "") : payload;
  const starts = docIds
    .map((id) => ({ id, at: body.indexOf(`[${id}]\n`) }))
    .filter((s) => s.at >= 0)
    .sort((a, b) => a.at - b.at);
  const contexts = starts.length
    ? starts.map((s, i) => ({
        id: s.id,
        text: body.slice(s.at + s.id.length + 3, starts[i + 1]?.at ?? body.length).trim(),
      }))
    : [{ id: "", text: body }];
  return { contexts };
}

const Block = ({
  label,
  color,
  children,
  testid,
}: {
  label: string;
  color: string;
  children: ReactNode;
  testid?: string;
}) => (
  <div className="overflow-hidden rounded-2xl border border-[#E6E1D6]" data-testid={testid}>
    <div className="flex items-center gap-2 border-b border-[#E6E1D6] bg-[#FBFAF7] px-4 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-[#6B6880]">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} /> {label}
    </div>
    <div className="whitespace-pre-wrap px-4 py-3 font-mono text-[12.5px] leading-[1.85] text-[#2A2840]">
      {children}
    </div>
  </div>
);

export const PayloadSheet = ({
  variant,
  run,
  onClose,
}: {
  variant: Variant | null;
  run: Run | null;
  onClose: () => void;
}) => {
  const v = variant ? V[variant] : null;
  const answer = v && run ? run.data[v.key] : null;
  const raw = answer ? `${SYSTEM_PROMPT}\n\n${answer.payload}` : "";
  const { contexts } = answer ? split(answer.payload, answer.doc_ids) : { contexts: [] };
  const copy = () => {
    navigator.clipboard?.writeText(raw);
    toast.success("Payload copied");
  };
  return (
    <Dialog.Root open={!!answer} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="sheet-overlay fixed inset-0 z-50 bg-black/80" />
        <Dialog.Content
          data-testid="model-payload-drawer"
          className="sheet-content fixed inset-y-0 right-0 z-50 h-full w-full gap-4 overflow-y-auto border-l border-[#E6E1D6] bg-white p-6 shadow-lg sm:max-w-xl"
        >
          {v && answer && run && (
            <>
              <div className="flex flex-col space-y-2 text-left">
                <span className="chip w-fit" style={{ background: v.bg, color: v.ink }}>
                  {v.name} · model payload
                </span>
                <Dialog.Title className="font-display text-2xl font-extrabold text-[#17162B]">
                  Exactly what the model received
                </Dialog.Title>
                <Dialog.Description className="text-sm text-[#6B6880]">
                  {raw.length.toLocaleString()} characters · {answer.doc_ids.length} retrieved{" "}
                  {answer.doc_ids.length === 1 ? "record" : "records"}
                </Dialog.Description>
              </div>
              <div className="mt-6 space-y-4">
                <Block label="system" color="#2D8CFF" testid="payload-system">
                  {SYSTEM_PROMPT}
                </Block>
                {contexts.map((c, i) => (
                  <Block
                    key={i}
                    label={c.id ? `context · ${c.id}` : "context"}
                    color={variant === "standard" ? "#FF4D3D" : "#0FB5A0"}
                  >
                    <Highlighted text={c.text} terms={run.terms} side={v.side} />
                  </Block>
                ))}
                <Block label="user" color="#F5A300" testid="payload-user">
                  {run.question}
                </Block>
              </div>
              <button
                data-testid="copy-payload-button"
                onClick={copy}
                className="btn-primary mt-6 flex w-full items-center justify-center gap-2 rounded-2xl py-3 text-sm"
              >
                <Copy className="h-4 w-4" /> Copy raw payload
              </button>
              <Dialog.Close className="absolute right-4 top-4 rounded-sm opacity-70 transition-opacity hover:opacity-100">
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Dialog.Close>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
