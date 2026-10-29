import type { RoutingDecision, SensitiveSpan } from "@/api/client";
import { category, type Kind } from "@/lib/agents";

/** A span's position, which links it to its Agent 3 decision and its twin in the rewrite. */
export const keyOf = (s: { start?: number | null; end?: number | null }) => `${s.start}:${s.end}`;

export type Detected = {
  key: string;
  text: string;
  type: string;
  how: string; // "rules 0.85" or "model 0.70"
  kept: boolean | null; // null until Agent 3 has decided
};

/** The original text, cut at each detected span (Agent 1). */
export function detectedPieces(
  text: string,
  spans: SensitiveSpan[],
  decisions: RoutingDecision[] | null,
): (string | Detected)[] {
  const action = new Map(decisions?.map((d) => [keyOf(d), d.action]));
  const placed = spans
    .filter((s) => s.start != null && s.end != null)
    .sort((a, b) => a.start! - b.start!);
  const out: (string | Detected)[] = [];
  let at = 0;
  for (const s of placed) {
    if (s.start! < at) continue; // overlaps a span already placed
    out.push(text.slice(at, s.start!), {
      key: keyOf(s),
      text: text.slice(s.start!, s.end!),
      type: category(s.category),
      how: `${s.method === "rule" ? "rules" : "model"} ${s.confidence.toFixed(2)}`,
      kept: decisions ? action.get(keyOf(s)) === "keep" : null,
    });
    at = s.end!;
  }
  out.push(text.slice(at));
  return out.filter((p) => p !== "");
}

export type Rewritten = { key: string; text: string; type: string; kind: Kind };

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The rewritten text, cut at each token and generalised phrase (Agent 3). */
export function rewrittenPieces(text: string, decisions: RoutingDecision[]): (string | Rewritten)[] {
  const byText = new Map<string, RoutingDecision>();
  for (const d of decisions) {
    const r = d.action === "keep" ? "" : (d.replacement ?? "").toLowerCase();
    if (r && !byText.has(r)) byText.set(r, d);
  }
  if (!byText.size) return [text];
  const pattern = [...byText.keys()].sort((a, b) => b.length - a.length).map(escape).join("|");
  return text
    .split(new RegExp(`(${pattern})`, "gi"))
    .map((p, i): string | Rewritten => {
      if (i % 2 === 0) return p;
      const d = byText.get(p.toLowerCase())!;
      return {
        key: keyOf(d),
        text: p,
        type: category(d.category),
        kind: d.action === "placeholder" ? "token" : "vague",
      };
    })
    .filter((p) => p !== "");
}

/** writer -> The writer; writer.wife -> The writer's wife; patient.mother -> The patient's mother; person_2 -> Person 2 */
export function subjectLabel(subject: string): string {
  if (subject === "patient") return "The patient";
  if (subject.startsWith("patient.")) return `The patient's ${subject.slice(8).replaceAll("_", " ")}`;
  if (subject === "writer") return "The writer";
  if (subject.startsWith("writer.")) return `The writer's ${subject.slice(7).replaceAll("_", " ")}`;
  return subject.charAt(0).toUpperCase() + subject.slice(1).replaceAll("_", " ");
}
