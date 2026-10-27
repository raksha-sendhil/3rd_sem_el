export const AGENTS = {
  detect: {
    name: "Detect",
    step: "Agent 1",
    color: "#2D8CFF",
    bg: "#E3F0FF",
  },
  extract: {
    name: "Extract",
    step: "Agent 2",
    color: "#6D4AFF",
    bg: "#EEE9FF",
  },
  rewrite: {
    name: "Rewrite",
    step: "Agent 3",
    color: "#E08E00",
    bg: "#FFF3D1",
  },
} as const;

/** How Agent 3 changed a detail: a token (purple) or a vaguer phrase (amber). */
export const KIND = {
  token: { label: "Tokenised", color: "#6D4AFF", bg: "#EEE9FF" },
  vague: { label: "Generalised", color: "#E08E00", bg: "#FFF3D1" },
} as const;
export type Kind = keyof typeof KIND;

/** MEDICAL_CONDITION -> Medical condition */
export const category = (c: string) => c.charAt(0) + c.slice(1).toLowerCase().replaceAll("_", " ");

export const seconds = (ms: number) => `${(ms / 1000).toFixed(2)}s`;
