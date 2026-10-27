import type { AttackDefinition, AttackKind, AttackRound, RoundLeakage } from "@/api/client";
import { IMAGES } from "@/lib/images";

export type Meta = {
  color: string;
  bg: string;
  tag: string;
  blurb: string;
  unit: string;
  image: string;
};

const KINDS: Record<AttackKind, Omit<Meta, "tag"> & { tag: (n: number) => string }> = {
  context_dump: {
    color: "#FF4D3D",
    bg: "#FFE8E4",
    tag: (n) => (n === 1 ? "1 message" : `${n} messages`),
    blurb: "A single prompt-injection message tells the model to print everything it was given.",
    unit: "Round",
    image: IMAGES.context_dump,
  },
  stepwise: {
    color: "#6D4AFF",
    bg: "#EEE9FF",
    tag: (n) => `${n} questions`,
    blurb: "Innocent-looking questions, each harmless alone, that together single out one person.",
    unit: "Round",
    image: IMAGES.stepwise,
  },
  interception: {
    color: "#2D8CFF",
    bg: "#E3F0FF",
    tag: () => "passive",
    blurb: "An eavesdropper silently reads the traffic sent to the model. It never asks a thing.",
    unit: "Packet",
    image: IMAGES.interception,
  },
};

export const ORDER: AttackKind[] = ["context_dump", "stepwise", "interception"];

export const metaOf = (a: AttackDefinition): Meta => {
  const k = KINDS[a.kind];
  return { ...k, tag: k.tag(a.queries.length) };
};

export type Side = "std" | "prot";
export const leakOf = (r: AttackRound, side: Side): RoundLeakage | null =>
  (side === "std" ? r.baseline_leakage : r.protected_leakage) ?? null;

export const pctOf = (l: RoundLeakage | null) => Math.round((l?.recovery_rate ?? 0) * 100);
