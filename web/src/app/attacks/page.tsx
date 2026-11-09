"use client";

import { Activity, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, type AttackDefinition, type AttackRun, type MonitorLedger } from "@/api/client";
import { AttackPicker } from "@/components/attacks/AttackPicker";
import { ChatPanel, type Msg } from "@/components/attacks/ChatPanel";
import { LeakageChart } from "@/components/attacks/LeakageChart";
import { MonitorPanel, type MonitorRound } from "@/components/attacks/MonitorPanel";
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
  // the session monitor's ledger for the protected chat, refreshed after each protected answer
  const [ledger, setLedger] = useState<MonitorLedger | null>(null);
  // the picked attack played end to end and scored three ways (standard / rewriting / monitor)
  const [scored, setScored] = useState<AttackRun | null>(null);
  const [scoring, setScoring] = useState(false);

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
  const pickedAttack = attacks.find((a) => a.attack_id === picked) ?? null;
  const protectedRounds: MonitorRound[] = chats.protected.messages.flatMap((m) =>
    m.role === "assistant" && m.answer.monitor ? [{ query: m.answer.query, monitor: m.answer.monitor }] : [],
  );

  const pick = (id: string) => {
    const attack = attacks.find((a) => a.attack_id === id);
    if (!attack) return;
    if (attack.kind === "interception") return setIntercept(attack);
    if (busy) return;
    setPicked(id);
    setChats({ standard: fresh(attack.queries), protected: fresh(attack.queries) });
    setLedger(null);
    setScored(null);
  };

  const score = async () => {
    if (!picked || scoring) return;
    setScoring(true);
    setError(null);
    try {
      setScored(await api.runAttack(picked));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setScoring(false);
    }
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
      if (variant === "protected") api.monitorLedger(session).then(setLedger).catch(() => {});
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
            onClear={() => {
              update(variant, () => fresh());
              if (variant === "protected") setLedger(null); // a new chat is a new session
            }}
          />
        ))}
      </div>
      <MonitorPanel rounds={protectedRounds} ledger={ledger} />
      {pickedAttack && pickedAttack.kind !== "interception" && (
        <section className="card overflow-hidden" data-testid="score-attack">
          <div className="flex flex-wrap items-center gap-3 border-b border-[#E6E1D6] px-5 py-4">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-[#17162B] text-white">
              <Activity className="h-5 w-5" strokeWidth={2.4} />
            </span>
            <div>
              <div className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#6B6880]">
                Secret Recovery Rate · {pickedAttack.name}
              </div>
              <div className="font-display text-base font-bold text-[#17162B]">
                How much of the target the attacker has recovered after each question
              </div>
            </div>
            <button
              onClick={score}
              disabled={scoring || busy}
              className="btn-primary ml-auto flex items-center gap-2 rounded-2xl px-5 py-2.5 text-sm"
            >
              {scoring && <Loader2 className="h-4 w-4 animate-spin" />}
              {scoring ? "Playing every question…" : scored ? "Score again" : "Score this attack"}
            </button>
          </div>
          <div className="px-5 py-5">
            {scored ? (
              <LeakageChart run={scored} />
            ) : (
              <p className="text-sm text-[#6B6880]">
                Plays all {pickedAttack.queries.length} questions in a fresh session and scores every answer against
                the target&apos;s real details: standard RAG, rewriting alone, and rewriting with the session
                monitor.
              </p>
            )}
          </div>
        </section>
      )}
      <PayloadDialog attack={intercept} model={model} onClose={() => setIntercept(null)} />
    </div>
  );
}
