"use client";

import { ShieldCheck, ShieldMinus, ShieldX } from "lucide-react";
import type { MonitorVerdict } from "@/api/client";
import { Tip } from "@/lib/marks";

const TONE = {
  allowed: { bg: "#DDF7F2", ink: "#08776A", icon: ShieldCheck },
  generalised: { bg: "#FFF3D1", ink: "#8A5A00", icon: ShieldMinus },
  blocked: { bg: "#FFE8E4", ink: "#B42318", icon: ShieldX },
};

/** The session monitor's verdict on a protected answer (always "allowed" until Phase 3b). */
export const Verdict = ({ monitor }: { monitor?: MonitorVerdict | null }) => {
  const verdict = monitor?.verdict ?? "allowed";
  const t = TONE[verdict];
  return (
    <Tip content={monitor?.reason || "session monitor"}>
      <span className="chip" style={{ background: t.bg, color: t.ink }} data-testid="monitor-verdict">
        <t.icon className="h-3 w-3" strokeWidth={2.6} />
        {verdict}
      </span>
    </Tip>
  );
};

/** "condition_0" -> "condition" */
export const detail = (key: string) => key.replace(/_\d+$/, "").replaceAll("_", " ");
