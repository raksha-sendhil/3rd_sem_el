"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ApiError, api } from "@/api/client";
import { AnswerPanel, type Run, type Variant } from "@/components/compare/AnswerPanel";
import { HistoryTable, type HistoryRow } from "@/components/compare/HistoryTable";
import { PayloadSheet } from "@/components/compare/PayloadSheet";
import { PipelineGraphic, type Sample } from "@/components/compare/PipelineGraphic";
import { QuestionBox } from "@/components/compare/QuestionBox";
import { Legend, PageHead } from "@/components/PageHead";
import { seconds } from "@/lib/agents";
import { loadTerms, stats } from "@/lib/marks";

const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

/** The record count, and one changed record to show in the pipeline graphic. */
function usePipelineSample() {
  const [records, setRecords] = useState<number | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  useEffect(() => {
    let alive = true;
    api
      .documents()
      .then(async (list) => {
        if (!alive) return;
        setRecords(list.length);
        const pick = list.find((d) => d.n_changes >= 4) ?? list.find((d) => d.n_changes > 0);
        if (!pick) return;
        const changed = (await api.document(pick.doc_id)).record.rewriter.decisions.filter(
          (d) => d.action !== "keep",
        );
        if (!alive) return;
        setSample({
          leaks: [...new Set(changed.map((d) => d.span_text.trim()))],
          token: changed.find((d) => d.action === "placeholder")?.replacement ?? undefined,
          vague: changed.find((d) => d.action === "generalise")?.replacement ?? undefined,
        });
      })
      .catch(() => alive && setRecords(null));
    return () => {
      alive = false;
    };
  }, []);
  return { records, sample };
}

export default function Compare() {
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [drawer, setDrawer] = useState<Variant | null>(null);
  const session = useRef<string | null>(null);
  const n = useRef(0);
  const { records, sample } = usePipelineSample();

  const ask = async (question: string) => {
    if (busy) return;
    setBusy(true);
    toast("Sent to both pipelines", { description: question });
    try {
      session.current ??= (await api.createSession()).session_id;
      const data = await api.compare(question, session.current);
      const terms = await loadTerms([...data.baseline.doc_ids, ...data.protected.doc_ids]);
      const id = `r${++n.current}`;
      const time = now();
      const std = stats(data.baseline.answer, terms, "std");
      const prot = stats(data.protected.answer, terms, "prot");
      setRun({ id, question, time, data, terms });
      setHistory((h) => [
        {
          id,
          time,
          question,
          leaks: std.leaks,
          tokens: prot.tokens,
          vague: prot.vague,
          std: seconds(data.baseline.latency_ms),
          prot: seconds(data.protected.latency_ms),
        },
        ...h,
      ]);
    } catch (e) {
      toast.error("The run failed", { description: e instanceof ApiError ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="grid items-center gap-8 lg:grid-cols-[1fr_1.25fr]">
        <PageHead
          index="01"
          label="Compare"
          color="#6D4AFF"
          title={
            <>
              Same question.
              <br />
              Two pipelines.
              <br />
              <span className="text-[#FF4D3D]">One leaks.</span>
            </>
          }
        >
          Standard RAG hands the model your raw records. Protected RAG only ever sees versions
          rewritten by three agents — names become tokens, telling details go vague.
          <Legend />
        </PageHead>
        <div className="card rise p-4 sm:p-6" style={{ animationDelay: "120ms" }}>
          <PipelineGraphic records={records} sample={sample} />
        </div>
      </div>
      <QuestionBox onRun={ask} busy={busy} />
      <div className="grid gap-6 lg:grid-cols-2">
        <AnswerPanel variant="standard" run={run} busy={busy} onInspect={() => setDrawer("standard")} />
        <AnswerPanel variant="protected" run={run} busy={busy} onInspect={() => setDrawer("protected")} />
      </div>
      <HistoryTable rows={history} onRerun={ask} busy={busy} />
      <PayloadSheet variant={drawer} run={run} onClose={() => setDrawer(null)} />
    </div>
  );
}
