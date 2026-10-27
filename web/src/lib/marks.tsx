"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { motion } from "framer-motion";
import { useEffect, useState, type ReactNode } from "react";
import { api, type DocumentView } from "@/api/client";
import { AGENTS, KIND, category } from "./agents";

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={80}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={4}
          className="tip z-50 overflow-hidden rounded-md border border-[#E6E1D6] bg-white px-3 py-1.5 font-mono text-[11px] text-[#17162B] shadow-lg"
        >
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export type MarkKind = "leak" | "token" | "vague";
const CLASS: Record<MarkKind, string> = {
  leak: "mark-leak",
  token: "mark-token",
  vague: "mark-vague",
};

/** One highlighted detail with its tooltip; ``active`` rings it while its twin is hovered. */
export function Mark({
  kind,
  type,
  active,
  onHover,
  children,
}: {
  kind: MarkKind;
  type: string;
  active?: boolean;
  onHover?: (on: boolean) => void;
  children: ReactNode;
}) {
  const rewrite = KIND[kind === "token" ? "token" : "vague"];
  return (
    <Tip
      content={
        kind === "leak" ? (
          <span>
            <b className="text-[#D92D20]">exposed</b> · {type}
          </span>
        ) : (
          <span>
            <b style={{ color: rewrite.color }}>
              {AGENTS.rewrite.step} · {AGENTS.rewrite.name}
            </b>{" "}
            · {type}
          </span>
        )
      }
    >
      <span
        onMouseEnter={() => onHover?.(true)}
        onMouseLeave={() => onHover?.(false)}
        className={`${CLASS[kind]} ${active ? "mark-active" : ""}`}
      >
        {children}
      </span>
    </Tip>
  );
}

/** What the stored records say is private, so answers can be checked against it. */
export type Terms = {
  leaks: Map<string, string>; // lower-cased original detail -> category
  tokens: Map<string, string>; // placeholder token -> category
  vague: Map<string, string>; // lower-cased generalised phrase -> category
};

const EMPTY: Terms = { leaks: new Map(), tokens: new Map(), vague: new Map() };

export function termsOf(views: (DocumentView | null)[]): Terms {
  const t: Terms = { leaks: new Map(), tokens: new Map(), vague: new Map() };
  for (const v of views)
    for (const d of v?.record.rewriter.decisions ?? []) {
      if (d.action === "keep") continue;
      const type = category(d.category);
      const original = d.span_text.trim();
      if (original.length >= 2) t.leaks.set(original.toLowerCase(), type);
      const r = d.replacement?.trim();
      if (!r) continue;
      if (d.action === "placeholder") t.tokens.set(r, type);
      else t.vague.set(r.toLowerCase(), type);
    }
  return t;
}

export async function loadTerms(docIds: string[]): Promise<Terms> {
  const ids = [...new Set(docIds)];
  return termsOf(await Promise.all(ids.map((id) => api.document(id).catch(() => null))));
}

export function useTerms(docIds: string[]): Terms {
  const key = [...new Set(docIds)].sort().join(",");
  const [terms, setTerms] = useState<Terms>(EMPTY);
  useEffect(() => {
    let alive = true;
    loadTerms(key ? key.split(",") : []).then((t) => alive && setTerms(t));
    return () => {
      alive = false;
    };
  }, [key]);
  return terms;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const TOKEN = /^\[[A-Z_]+_\d+\]$/;

type Segment = { text: string; kind?: MarkKind; type?: string };

/** Splits ``text`` into plain runs and marked details. The standard side marks only leaks;
 * the protected side also marks tokens and generalised phrases. */
export function scan(text: string, terms: Terms, side: "std" | "prot"): Segment[] {
  const words = [...terms.leaks.keys(), ...(side === "prot" ? terms.vague.keys() : [])]
    .sort((a, b) => b.length - a.length)
    .map((w) => String.raw`(?<![\w])${escape(w)}(?![\w])`);
  const parts = [...(side === "prot" ? [String.raw`\[[A-Z_]+_\d+\]`] : []), ...words];
  if (!parts.length) return [{ text }];
  return text
    .split(new RegExp(`(${parts.join("|")})`, "gi"))
    .map((p, i): Segment => {
      if (i % 2 === 0) return { text: p };
      if (TOKEN.test(p)) return { text: p, kind: "token", type: terms.tokens.get(p) ?? "Token" };
      const lower = p.toLowerCase();
      if (terms.leaks.has(lower)) return { text: p, kind: "leak", type: terms.leaks.get(lower) };
      return { text: p, kind: "vague", type: terms.vague.get(lower) };
    })
    .filter((s) => s.text);
}

/** Distinct details of each kind in ``text``. */
export function stats(text: string, terms: Terms, side: "std" | "prot") {
  const seen = { leak: new Set<string>(), token: new Set<string>(), vague: new Set<string>() };
  for (const s of scan(text, terms, side)) if (s.kind) seen[s.kind].add(s.text.toLowerCase());
  return { leaks: seen.leak.size, tokens: seen.token.size, vague: seen.vague.size };
}

export function Highlighted({
  text,
  terms,
  side,
  animate = false,
}: {
  text: string;
  terms: Terms;
  side: "std" | "prot";
  animate?: boolean;
}) {
  return (
    <span>
      {scan(text, terms, side).map((s, i) => {
        const node = s.kind ? (
          <Mark kind={s.kind} type={s.type ?? ""}>
            {s.text}
          </Mark>
        ) : (
          s.text
        );
        return animate ? (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.25 }}
          >
            {node}
          </motion.span>
        ) : (
          <span key={i}>{node}</span>
        );
      })}
    </span>
  );
}
