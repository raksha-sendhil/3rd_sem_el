import { ShieldAlert, ShieldCheck } from "lucide-react";

export type Variant = "standard" | "protected";
export const VARIANTS: Variant[] = ["standard", "protected"];

export const SIDES = {
  standard: {
    name: "Standard RAG",
    icon: ShieldAlert,
    color: "#FF4D3D",
    shade: "#D92D20",
    bg: "#FFE8E4",
    ink: "#B42318",
    pipeline: "baseline",
    side: "std",
  },
  protected: {
    name: "Protected RAG",
    icon: ShieldCheck,
    color: "#0FB5A0",
    shade: "#08776A",
    bg: "#DDF7F2",
    ink: "#08776A",
    pipeline: "protected",
    side: "prot",
  },
} as const;
