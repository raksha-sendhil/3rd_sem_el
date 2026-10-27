"use client";

import { Loader2, Play } from "lucide-react";
import type { Meta } from "./meta";

export const Controls = ({
  meta,
  shown,
  total,
  running,
  onLaunch,
}: {
  meta: Meta;
  shown: number;
  total: number;
  running: boolean; // rounds are still arriving from the API
  onLaunch: () => void;
}) => (
  <div className="card flex flex-wrap items-center gap-3 p-3 pl-4" data-testid="attack-controls">
    <button
      data-testid="attack-play-button"
      onClick={onLaunch}
      disabled={running}
      className="flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-bold text-white shadow-md transition-transform hover:-translate-y-0.5 disabled:opacity-70 disabled:hover:translate-y-0"
      style={{ background: meta.color }}
    >
      {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
      {running ? "Running…" : shown ? "Run again" : "Launch attack"}
    </button>
    <div className="ml-auto flex items-center gap-3 font-mono text-xs text-[#4A4760]" data-testid="attack-progress">
      {running && (
        <span className="chip bg-[#FFE8E4] text-[#B42318]">
          <span className="pulse-dot bg-[#FF4D3D]" /> LIVE
        </span>
      )}
      <span>
        {meta.unit} {shown}/{total}
      </span>
      <div className="h-2 w-28 overflow-hidden rounded-full bg-[#EFECE4]">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${total ? (shown / total) * 100 : 0}%`, background: meta.color }}
        />
      </div>
    </div>
  </div>
);
