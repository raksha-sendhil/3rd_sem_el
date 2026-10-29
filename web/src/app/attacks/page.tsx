"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api, type AttackDefinition } from "@/api/client";
import { AttackPicker } from "@/components/attacks/AttackPicker";
import { ChatPanel, type Msg } from "@/components/attacks/ChatPanel";
import { ORDER } from "@/components/attacks/meta";
import { PayloadDialog } from "@/components/attacks/PayloadDialog";
import { SIDES, VARIANTS, type Variant } from "@/components/attacks/sides";
import { ErrorNote, PageHead } from "@/components/PageHead";
import { loadTerms } from "@/lib/marks";

type Chat = {
  session: string | null;
  messages: Msg[];
  draft: string;
  queue: string[]; // the attack's next prompts, filled in one at a time after each reply
  busy: boolean;
};

const fresh = (prompts: string[] = []): Chat => ({
  session: null,
  messages: [],
  draft: prompts[0] ?? "",
  queue: prompts.slice(1),
  busy: false,
});

export default function Attacks() {
  const [attacks, setAttacks] = useState<AttackDefinition[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("llama3:8b");
  const [picked, setPicked] = useState<string | null>(null);
  const [intercept, setIntercept] = useState<AttackDefinition | null>(null);
  const [chats, setChats] = useState<Record<Variant, Chat>>({
    standard: fresh(),
    protected: fresh(),
  });
  const ids = useRef(0);

  useEffect(() => {
    api
      .attacks()
      .then((list) => setAttacks([...list].sort((a, b) => ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind))))
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    api
      .health()
      .then((h) => setModel(h.model))
      .catch(() => {});
  }, []);

  const update = (variant: Variant, change: (c: Chat) => Chat) =>
    setChats((all) => ({ ...all, [variant]: change(all[variant]) }));

  const busy = VARIANTS.some((v) => chats[v].busy);

  const pick = (id: string) => {
    const attack = attacks.find((a) => a.attack_id === id);
    if (!attack) return;
    if (attack.kind === "interception") return setIntercept(attack);
    if (busy) return;
    setPicked(id);
    setChats({ standard: fresh(attack.queries), protected: fresh(attack.queries) });
  };

  const send = async (variant: Variant) => {
    const chat = chats[variant];
    const text = chat.draft.trim();
    if (!text || chat.busy) return;
    update(variant, (c) => ({
      ...c,
      draft: "",
      busy: true,
      messages: [...c.messages, { id: ++ids.current, role: "user", text }],
    }));
    try {
      const session = chat.session ?? (await api.createSession()).session_id;
      const answer = await api.query(text, SIDES[variant].pipeline, session);
      const terms = await loadTerms(answer.doc_ids);
      update(variant, (c) => {
        const next = !c.draft && c.queue.length ? c.queue : null; // keep anything typed meanwhile
        return {
          ...c,
          session,
          busy: false,
          messages: [...c.messages, { id: ++ids.current, role: "assistant", answer, terms }],
          draft: next ? next[0] : c.draft,
          queue: next ? next.slice(1) : c.queue,
        };
      });
    } catch (e) {
      const message = e instanceof ApiError ? e.message : String(e);
      update(variant, (c) => ({
        ...c,
        busy: false,
        messages: [...c.messages, { id: ++ids.current, role: "error", text: message }],
      }));
    }
  };

  return (
    <div className="space-y-8">
      <PageHead
        index="02"
        label="Attacks"
        color="#FF4D3D"
        title={
          <>
            Try to break it.
            <br />
            <span className="text-[#6D4AFF]">Watch who gives in.</span>
          </>
        }
      />
      <AttackPicker attacks={attacks} value={picked} onChange={pick} />
      <ErrorNote error={error} />
      <div className="grid gap-6 lg:grid-cols-2">
        {VARIANTS.map((variant) => (
          <ChatPanel
            key={variant}
            variant={variant}
            messages={chats[variant].messages}
            draft={chats[variant].draft}
            busy={chats[variant].busy}
            onDraft={(draft) => update(variant, (c) => ({ ...c, draft }))}
            onSend={() => send(variant)}
            onClear={() => update(variant, () => fresh())}
          />
        ))}
      </div>
      <PayloadDialog attack={intercept} model={model} onClose={() => setIntercept(null)} />
    </div>
  );
}
