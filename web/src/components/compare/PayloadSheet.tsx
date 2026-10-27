"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { Highlighted } from "@/lib/marks";
import { V, type Run, type Variant } from "./AnswerPanel";

/** The user message is "CONTEXT:\n[id]\ntext\n\n[id]\ntext\n\nQUESTION: q\nANSWER:". */
function split(payload: string, docIds: string[]) {
  const q = payload.lastIndexOf("\n\nQUESTION: ");
  const body = q >= 0 ? payload.slice(0, q).replace(/^CONTEXT:\n/, "") : payload;
  const starts = docIds
    .map((id) => ({ id, at: body.indexOf(`[${id}]\n`) }))
    .filter((s) => s.at >= 0)
    .sort((a, b) => a.at - b.at);
  return starts.length
    ? starts.map((s, i) => ({
        id: s.id,
        text: body.slice(s.at + s.id.length + 3, starts[i + 1]?.at ?? body.length).trim(),
      }))
    : [{ id: "", text: body }];
}

const Block = ({ label, color, children }: { label: string; color: string; children: ReactNode }) => (
  <div className="overflow-hidden rounded-2xl border border-[#E6E1D6]">
    <div className="flex items-center gap-2 border-b border-[#E6E1D6] bg-[#FBFAF7] px-4 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-[#6B6880]">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} /> {label}
    </div>
    <div className="whitespace-pre-wrap px-4 py-3 font-mono text-[12.5px] leading-[1.85] text-[#2A2840]">
      {children}
    </div>
  </div>
);

/** The records and question one pipeline sent to the model. */
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
  return (
    <Dialog.Root open={!!answer} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="sheet-overlay fixed inset-0 z-50 bg-black/80" />
        <Dialog.Content
          data-testid="model-payload-drawer"
          className="sheet-content fixed inset-y-0 right-0 z-50 h-full w-full overflow-y-auto border-l border-[#E6E1D6] bg-white p-6 shadow-lg sm:max-w-xl"
        >
          {v && answer && run && (
            <>
              <div className="flex flex-col space-y-2 text-left">
                <span className="chip w-fit" style={{ background: v.bg, color: v.ink }}>
                  {v.name}
                </span>
                <Dialog.Title className="font-display text-2xl font-extrabold text-[#17162B]">
                  What the model saw
                </Dialog.Title>
                <Dialog.Description className="sr-only">
                  The records and question sent to the model
                </Dialog.Description>
              </div>
              <div className="mt-6 space-y-4">
                {split(answer.payload, answer.doc_ids).map((c, i) => (
                  <Block
                    key={i}
                    label={c.id ? `context · ${c.id}` : "context"}
                    color={variant === "standard" ? "#FF4D3D" : "#0FB5A0"}
                  >
                    <Highlighted text={c.text} terms={run.terms} side={v.side} />
                  </Block>
                ))}
                <Block label="question" color="#F5A300">
                  {run.question}
                </Block>
              </div>
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
