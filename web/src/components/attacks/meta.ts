import type { AttackDefinition, AttackKind } from "@/api/client";
import { IMAGES } from "@/lib/images";

export type Meta = {
  color: string;
  bg: string;
  tag: string;
  image: string;
  ready: boolean; // false greys the attack out
};

const KINDS: Record<AttackKind, Omit<Meta, "tag"> & { tag: (n: number) => string }> = {
  context_dump: {
    color: "#FF4D3D",
    bg: "#FFE8E4",
    tag: (n) => (n === 1 ? "1 message" : `${n} messages`),
    image: IMAGES.context_dump,
    ready: true,
  },
  stepwise: {
    color: "#6D4AFF",
    bg: "#EEE9FF",
    tag: (n) => `${n} questions`,
    image: IMAGES.stepwise,
    // needs the session monitor (Phase 3b) to show anything; AllowAllMonitor is in place for now
    ready: false,
  },
  interception: {
    color: "#2D8CFF",
    bg: "#E3F0FF",
    tag: () => "passive",
    image: IMAGES.interception,
    ready: true,
  },
};

export const ORDER: AttackKind[] = ["context_dump", "stepwise", "interception"];

export const metaOf = (a: AttackDefinition): Meta => {
  const k = KINDS[a.kind];
  return { ...k, tag: k.tag(a.queries.length) };
};
