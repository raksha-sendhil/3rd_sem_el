"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Radio, X } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, type AttackDefinition, type CompareResponse } from "@/api/client";
import { ErrorNote } from "@/components/PageHead";
import { Highlighted, loadTerms, type Terms } from "@/lib/marks";
import { SIDES, VARIANTS } from "./sides";

// The request OllamaClient sends for an answer (src/sanitise/llm/ollama.py, _generate), with
// the system prompt of src/sanitise/rag/answer_prompt.py and the answer settings of
// src/sanitise/rag/answer.py and config.py. The API returns only the user message, so these
// stay in step with the backend by hand.
const OLLAMA = "localhost:11434";
const SYSTEM_PROMPT = `You answer questions about a collection of records (patients' messages to doctors and workplace emails), using only the records under CONTEXT.
- Answer in one to three sentences.
- State details exactly as the records give them, including tokens such as [PERSON_1].
- If the records do not contain the answer, reply: "The records do not say."
- Do not guess or add anything that is not in the records.
`;
const OPTIONS = { temperature: 0.4, num_ctx: 4096, seed: 42, top_p: 0.9, num_predict: 512 };

/** The request split where the user message starts: only that part carries the records. */
function request(model: string, userMessage: string): [string, string] {
  const body = {
    model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userMessage },
    ],
    stream: false,
    options: OPTIONS,
  };
  const text =
    `POST /api/chat HTTP/1.1\nHost: ${OLLAMA}\nContent-Type: application/json\n\n` +
    JSON.stringify(body, null, 2);
  const at = text.indexOf('"role": "user"');
  return [text.slice(0, at), text.slice(at)];
}

type Captured = { data: CompareResponse; terms: Terms };

/** What a passive observer on the wire sees: each pipeline's request to the model. */
export const PayloadDialog = ({
  attack,
  model,
  onClose,
}: {
  attack: AttackDefinition | null;
  model: string;
  onClose: () => void;
}) => {
  const [index, setIndex] = useState(0);
  const [captured, setCaptured] = useState<Record<string, Captured>>({});
  const [error, setError] = useState<string | null>(null);
  const query = attack?.queries[index];
  const shot = query ? captured[query] : undefined;

  useEffect(() => setIndex(0), [attack]);

  useEffect(() => {
    if (!query || captured[query]) return;
    let alive = true;
    setError(null);
    (async () => {
      const session = (await api.createSession()).session_id;
      const data = await api.compare(query, session);
      const terms = await loadTerms([...data.baseline.doc_ids, ...data.protected.doc_ids]);
      if (alive) setCaptured((c) => ({ ...c, [query]: { data, terms } }));
    })().catch((e) => alive && setError(e instanceof ApiError ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [query, captured]);

  return (
    <Dialog.Root open={!!attack} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="sheet-overlay fixed inset-0 z-50 bg-black/80" />
        <Dialog.Content
          data-testid="payload-dialog"
          className="fixed inset-x-4 top-1/2 z-50 mx-auto max-h-[90vh] max-w-6xl -translate-y-1/2 overflow-y-auto rounded-3xl border border-[#E6E1D6] bg-white p-6 shadow-lg"
        >
          {attack && (
            <>
              <div className="flex items-center gap-3 pr-8">
                <span className="grid h-10 w-10 place-items-center rounded-xl bg-[#E3F0FF] text-[#2D8CFF]">
                  <Radio className="h-5 w-5" />
                </span>
                <Dialog.Title className="font-display text-2xl font-extrabold text-[#17162B]">
                  {attack.name}
                </Dialog.Title>
                <Dialog.Description className="sr-only">
                  The requests each pipeline sends to the model
                </Dialog.Description>
              </div>

              <div className="mt-5 flex flex-wrap gap-2">
                {attack.queries.map((q, i) => (
                  <button
                    key={q}
                    onClick={() => setIndex(i)}
                    className={`rounded-full border px-3 py-1.5 text-left text-xs font-medium transition-[border-color,color,transform] hover:-translate-y-0.5 ${
                      i === index
                        ? "border-[#2D8CFF] bg-[#E3F0FF] text-[#17162B]"
                        : "border-[#E6E1D6] bg-white text-[#4A4760] hover:border-[#2D8CFF]"
                    }`}
                  >
                    <span className="mr-1.5 font-mono font-bold text-[#2D8CFF]">0{i + 1}</span>
                    {q}
                  </button>
                ))}
              </div>

              <div className="mt-4">
                <ErrorNote error={error} />
              </div>

              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                {VARIANTS.map((variant) => {
                  const v = SIDES[variant];
                  const answer = shot?.data[v.pipeline];
                  return (
                    <div key={variant} className="overflow-hidden rounded-2xl border border-[#E6E1D6]">
                      <div className="flex items-center justify-between gap-2 border-b border-[#E6E1D6] px-4 py-2.5" style={{ background: v.bg }}>
                        <span className="flex items-center gap-2 font-display text-sm font-bold text-[#17162B]">
                          <v.icon className="h-4 w-4" style={{ color: v.color }} strokeWidth={2.4} />
                          {v.name}
                        </span>
                        <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]" style={{ color: v.ink }}>
                          to the model
                        </span>
                      </div>
                      <pre className="max-h-[56vh] overflow-y-auto whitespace-pre-wrap break-words bg-[#FBFAF7] px-4 py-3 font-mono text-[12px] leading-[1.8] text-[#2A2840]">
                        {answer && shot ? (
                          (() => {
                            const [head, user] = request(model, answer.payload);
                            return (
                              <>
                                {head}
                                <Highlighted text={user} terms={shot.terms} side={v.side} />
                              </>
                            );
                          })()
                        ) : (
                          !error && (
                            <span className="block space-y-2 py-1">
                              {[70, 92, 84, 60, 88].map((w, i) => (
                                <span key={i} className="shimmer block h-3 rounded-full" style={{ width: `${w}%` }} />
                              ))}
                            </span>
                          )
                        )}
                      </pre>
                    </div>
                  );
                })}
              </div>

              <Dialog.Close className="absolute right-5 top-5 rounded-sm opacity-70 transition-opacity hover:opacity-100">
                <X className="h-5 w-5" />
                <span className="sr-only">Close</span>
              </Dialog.Close>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
