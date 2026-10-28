"use client";

import { ArrowUpRight, FileText } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/api/client";
import { PipelineGraphic, type Sample } from "@/components/compare/PipelineGraphic";
import { Legend, PageHead } from "@/components/PageHead";

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

export default function Overview() {
  const { records, sample } = usePipelineSample();
  return (
    <div className="grid min-h-[calc(100vh-180px)] items-center gap-10 lg:grid-cols-[1fr_1.25fr]">
      <PageHead
        index="01"
        label="Overview"
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
        <Legend />
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/attacks"
            className="btn-primary flex items-center gap-2 rounded-2xl px-6 py-3 text-sm"
          >
            <ArrowUpRight className="h-4 w-4" /> Try the attacks
          </Link>
          <Link
            href="/documents"
            className="flex items-center gap-2 rounded-2xl border-2 border-[#E6E1D6] bg-white px-6 py-3 text-sm font-bold text-[#17162B] transition-[border-color,transform] hover:-translate-y-0.5 hover:border-[#17162B]"
          >
            <FileText className="h-4 w-4" /> See the documents
          </Link>
        </div>
      </PageHead>
      <div className="card rise p-4 sm:p-6" style={{ animationDelay: "120ms" }}>
        <PipelineGraphic records={records} sample={sample} />
      </div>
    </div>
  );
}
