import type { AttackDefinition, AttackKind } from "@/api/client";
import { IMAGES } from "@/lib/images";

export type Meta = {
  color: string;
  bg: string;
  tag: string;
  image: string;
};

const KINDS: Record<AttackKind, Omit<Meta, "tag"> & { tag: (n: number) => string }> = {
  context_dump: {
    color: "#FF4D3D",
    bg: "#FFE8E4",
    tag: (n) => (n === 1 ? "1 message" : `${n} messages`),
    image: IMAGES.context_dump,
  },
  stepwise: {
    color: "#6D4AFF",
    bg: "#EEE9FF",
    tag: (n) => `${n} questions`,
    image: IMAGES.stepwise,
  },
  interception: {
    color: "#2D8CFF",
    bg: "#E3F0FF",
    tag: () => "passive",
    image: IMAGES.interception,
  },
};

export const ORDER: AttackKind[] = ["context_dump", "stepwise", "interception"];

export const metaOf = (a: AttackDefinition): Meta => {
  const k = KINDS[a.kind];
  return { ...k, tag: k.tag(a.queries.length) };
};
