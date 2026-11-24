// Evaluation results shown on the Results page: no privacy (standard RAG over the original
// documents) against ours (Agents 1-3's rewrites with the session monitor).
// Health and legal: run 1 at the 5h scale (results/eval/health/run1_5h and the run's report).
// Finance: the bank source at the night scale, scored on the documents that were built.
// Rates are fractions (0.3 = 30%); latencies are median seconds.

import type { DomainId } from "./domains";

export type Pair = { base: number; ours: number };

export type DomainResults = {
  membership: { auc: Pair; tpr: Pair; members: number; nonmembers: number };
  targetInfo: { values: Pair; questionsLeaking: Pair; questions: number; oursNote: string };
  sessions: {
    sls: Pair;
    srr: { base: number[]; ours: number[] };
    count: number;
  };
  reid: { singledOut: Pair; belowK: Pair; medianPool: Pair; people: number };
  copying: { repeat: Pair; rouge: Pair; prompts: number };
  latency: { single: Pair; turn: Pair };
  utility?: { retention: number; retentionAnswered: number; answered: Pair; questions: number };
  scans: {
    documents: number;
    leaked: { label: string; base: number; ours: number }[];
    ocr: { label: string; value: string }[];
    oursNote?: string;
  };
  leftover?: { label: string; left: number; total: number; kind: "direct" | "indirect" }[];
  leftoverSource?: string;
};

export const RESULTS: Record<DomainId, DomainResults> = {
  medical: {
    membership: { auc: { base: 0.782, ours: 0.615 }, tpr: { base: 0.52, ours: 0.23 }, members: 100, nonmembers: 100 },
    targetInfo: {
      values: { base: 126, ours: 0 },
      questionsLeaking: { base: 0.95, ours: 0 },
      questions: 100,
      oursNote: "No name, phone or exact age came back. Coarse details (age band, district) still do.",
    },
    sessions: {
      sls: { base: 0.817, ours: 0 },
      srr: {
        base: [0.053, 0.125, 0.27, 0.308, 0.345, 0.374, 0.396, 0.5],
        ours: [0, 0, 0, 0, 0, 0, 0, 0],
      },
      count: 60,
    },
    reid: {
      singledOut: { base: 0.883, ours: 0.033 },
      belowK: { base: 0.95, ours: 0.3 },
      medianPool: { base: 1, ours: 9 },
      people: 120,
    },
    copying: { repeat: { base: 0.32, ours: 0.3 }, rouge: { base: 0.3, ours: 0 }, prompts: 100 },
    latency: { single: { base: 5.2, ours: 4.94 }, turn: { base: 1.2, ours: 0.88 } },
    scans: {
      documents: 25,
      leaked: [
        { label: "Typed text", base: 33, ours: 0 },
        { label: "Scan + OCR", base: 31, ours: 0 },
      ],
      ocr: [
        { label: "Character error, clinical scans (vision model)", value: "3.6%" },
        { label: "Character error, electronic PDFs (text layer)", value: "0.0%" },
        { label: "Median reading time per scanned document", value: "22 s" },
      ],
    },
    leftover: [
      { label: "Direct identifiers", left: 0, total: 146, kind: "direct" },
      { label: "Indirect details", left: 17, total: 115, kind: "indirect" },
    ],
    leftoverSource: "40 unseen annotated documents (Nemotron-PII)",
  },
  legal: {
    membership: { auc: { base: 0.993, ours: 0.563 }, tpr: { base: 0.97, ours: 0.11 }, members: 100, nonmembers: 100 },
    targetInfo: {
      values: { base: 176, ours: 0 },
      questionsLeaking: { base: 0.92, ours: 0 },
      questions: 100,
      oursNote: "No name, PAN, case number or phone came back. Coarse details still narrow some clients.",
    },
    sessions: {
      sls: { base: 0.883, ours: 0 },
      srr: {
        base: [0.09, 0.163, 0.175, 0.275, 0.315, 0.405, 0.418, 0.512],
        ours: [0, 0, 0, 0, 0, 0, 0, 0],
      },
      count: 60,
    },
    reid: {
      singledOut: { base: 0.9, ours: 0.017 },
      belowK: { base: 0.933, ours: 0.083 },
      medianPool: { base: 1, ours: 40 },
      people: 120,
    },
    copying: { repeat: { base: 0.4, ours: 0.44 }, rouge: { base: 0.4, ours: 0.06 }, prompts: 100 },
    latency: { single: { base: 3.7, ours: 2.33 }, turn: { base: 2.36, ours: 1.22 } },
    scans: {
      documents: 25,
      leaked: [
        { label: "Typed text", base: 35, ours: 2 },
        { label: "Scan + OCR", base: 34, ours: 2 },
      ],
      oursNote: "Ours let through a date and a reference number.",
      ocr: [
        { label: "Character error, contract scans (vision model)", value: "2.8%" },
        { label: "Character error, contracts as PDF (text layer)", value: "0.3%" },
        { label: "Median reading time per document", value: "0.8 s" },
      ],
    },
  },
  finance: {
    membership: { auc: { base: 0.804, ours: 0.643 }, tpr: { base: 0.412, ours: 0.059 }, members: 17, nonmembers: 40 },
    targetInfo: {
      values: { base: 36, ours: 2 },
      questionsLeaking: { base: 0.75, ours: 0.2 },
      questions: 20,
      oursNote: "The 2 are the product (\"current account\"), kept on purpose. No name, account or phone came back.",
    },
    sessions: {
      sls: { base: 0.417, ours: 0 },
      srr: {
        base: [0.008, 0.025, 0.125, 0.225, 0.283, 0.292, 0.308, 0.308],
        ours: [0, 0, 0, 0, 0, 0, 0, 0],
      },
      count: 24,
    },
    reid: {
      singledOut: { base: 0.667, ours: 0 },
      belowK: { base: 0.75, ours: 0 },
      medianPool: { base: 1, ours: 75 },
      people: 150,
    },
    copying: { repeat: { base: 0.4, ours: 0.125 }, rouge: { base: 0.375, ours: 0 }, prompts: 40 },
    latency: { single: { base: 3.52, ours: 3.17 }, turn: { base: 1.94, ours: 1.0 } },
    utility: { retention: 0.557, retentionAnswered: 0.905, answered: { base: 0.786, ours: 0.5 }, questions: 14 },
    scans: {
      documents: 10,
      leaked: [{ label: "Bank statements, scan + OCR", base: 10, ours: 0 }],
      ocr: [
        { label: "Labelled fields read correctly", value: "100% of 60" },
        { label: "Median reading time per page", value: "1.8 s" },
      ],
    },
    leftover: [
      { label: "Names", left: 0, total: 58, kind: "direct" },
      { label: "Account numbers", left: 0, total: 58, kind: "direct" },
      { label: "Phones", left: 0, total: 58, kind: "direct" },
      { label: "Towns", left: 0, total: 58, kind: "direct" },
      { label: "District / state", left: 0, total: 116, kind: "indirect" },
    ],
    leftoverSource: "58 rewritten bank statements",
  },
};
