import { FileStack, Footprints, Link2, type LucideIcon, MessageSquareWarning, Radio } from "lucide-react";
import type { AttackDefinition, AttackKind } from "@/api/client";
import { IMAGES } from "@/lib/images";

export type Meta = {
  color: string;
  bg: string;
  tag: string;
  image: string;
  icon: LucideIcon; // shown when no image is set
  ready: boolean; // false greys the attack out
};

const KINDS: Record<AttackKind, Omit<Meta, "tag"> & { tag: (n: number) => string }> = {
  context_dump: {
    color: "#FF4D3D",
    bg: "#FFE8E4",
    tag: (n) => (n === 1 ? "1 message" : `${n} messages`),
    image: IMAGES.context_dump,
    icon: FileStack,
    ready: true,
  },
  stepwise: {
    color: "#6D4AFF",
    bg: "#EEE9FF",
    tag: (n) => `${n} questions`,
    image: IMAGES.stepwise,
    icon: Footprints,
    // answered normally until the session monitor sees the details add up to one person
    ready: true,
  },
  interception: {
    color: "#2D8CFF",
    bg: "#E3F0FF",
    tag: () => "passive",
    image: IMAGES.interception,
    icon: Radio,
    ready: true,
  },
  // the legal pipeline's attacks: asking for a client's details outright, and joining details
  // across files (healthcare has none of these)
  direct: {
    color: "#FF4D3D",
    bg: "#FFE8E4",
    tag: (n) => (n === 1 ? "1 question" : `${n} questions`),
    image: "",
    icon: MessageSquareWarning,
    ready: true,
  },
  linkage: {
    color: "#2D8CFF",
    bg: "#E3F0FF",
    tag: (n) => `${n} questions`,
    image: "",
    icon: Link2,
    ready: true,
  },
};

export const ORDER: AttackKind[] = ["context_dump", "direct", "stepwise", "linkage", "interception"];

export const metaOf = (a: AttackDefinition): Meta => {
  const k = KINDS[a.kind];
  return { ...k, tag: k.tag(a.queries.length) };
};
