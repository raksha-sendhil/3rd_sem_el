import type { DocumentView } from "@/api/client";
import { category, type Kind } from "@/lib/agents";

/** One detail Agent 3 replaced, with Agent 1's confidence in it. */
export type Change = {
  id: number;
  original: string;
  replacement: string;
  type: string;
  kind: Kind;
  conf: number | null;
  start: number | null;
  end: number | null;
};

export function changesOf(view: DocumentView): Change[] {
  const spans = view.record.detector.spans;
  return view.record.rewriter.decisions
    .filter((d) => d.action !== "keep")
    .map((d, id) => {
      const span =
        spans.find((s) => d.start != null && s.start === d.start && s.end === d.end) ??
        spans.find((s) => s.text === d.span_text);
      return {
        id,
        original: d.span_text,
        replacement: d.replacement ?? "",
        type: category(d.category),
        kind: d.action === "placeholder" ? "token" : "vague",
        conf: span?.confidence ?? null,
        start: d.start ?? null,
        end: d.end ?? null,
      };
    });
}

export type Piece = string | Change;

/** The original text, cut at each change's offsets. */
export function originalPieces(text: string, changes: Change[]): Piece[] {
  const placed = changes
    .filter((c) => c.start !== null && c.end !== null)
    .sort((a, b) => a.start! - b.start!);
  const out: Piece[] = [];
  let at = 0;
  for (const c of placed) {
    if (c.start! < at) continue; // overlaps a change already placed
    out.push(text.slice(at, c.start!), c);
    at = c.end!;
  }
  out.push(text.slice(at));
  return out.filter((p) => p !== "");
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The sanitised text, cut at each token and generalised phrase. */
export function sanitisedPieces(text: string, changes: Change[]): Piece[] {
  const byText = new Map<string, Change>();
  for (const c of changes) {
    const key = c.replacement.toLowerCase();
    if (key && !byText.has(key)) byText.set(key, c);
  }
  if (!byText.size) return [text];
  const pattern = [...byText.keys()].sort((a, b) => b.length - a.length).map(escape).join("|");
  return text
    .split(new RegExp(`(${pattern})`, "gi"))
    .map((p, i): Piece => {
      if (i % 2 === 0) return p;
      const c = byText.get(p.toLowerCase())!;
      return { ...c, replacement: p };
    })
    .filter((p) => p !== "");
}
