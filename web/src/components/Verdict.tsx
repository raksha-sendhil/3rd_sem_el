"use client";

import { ShieldCheck, ShieldMinus, ShieldX } from "lucide-react";
import type { MonitorVerdict } from "@/api/client";
import { Tip } from "@/lib/marks";

const TONE = {
  allowed: { bg: "#DDF7F2", ink: "#08776A", icon: ShieldCheck },
  generalised: { bg: "#FFF3D1", ink: "#8A5A00", icon: ShieldMinus },
  blocked: { bg: "#FFE8E4", ink: "#B42318", icon: ShieldX },
};

/** The session monitor's verdict on a protected answer; the tooltip says why, and the pool is
 * how many records still match what the session knows about the person. */
export const Verdict = ({ monitor }: { monitor?: MonitorVerdict | null }) => {
  const verdict = monitor?.verdict ?? "allowed";
  const t = TONE[verdict];
  const pool = monitor?.candidate_pool;
  return (
    <Tip content={monitor?.reason || "session monitor"}>
      <span className="chip whitespace-nowrap" style={{ background: t.bg, color: t.ink }} data-testid="monitor-verdict">
        <t.icon className="h-3 w-3" strokeWidth={2.6} />
        {verdict}
        {pool != null && <span className="font-normal opacity-80">· {pool} match</span>}
      </span>
    </Tip>
  );
};

export const VERDICT_TONE = TONE;

/** "condition_0" -> "condition" */
export const detail = (key: string) => key.replace(/_\d+$/, "").replaceAll("_", " ");
