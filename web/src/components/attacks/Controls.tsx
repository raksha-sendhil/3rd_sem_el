"use client";

import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import type { ReactNode } from "react";
import type { Meta } from "./meta";

export const SPEEDS = [1, 2, 4];

const IconBtn = ({
  onClick,
  label,
  children,
  testid,
  disabled,
}: {
  onClick: () => void;
  label: string;
  children: ReactNode;
  testid: string;
  disabled?: boolean;
}) => (
  <button
    data-testid={testid}
    onClick={onClick}
    disabled={disabled}
    aria-label={label}
    className="grid h-10 w-10 place-items-center rounded-full border border-[#E6E1D6] bg-white text-[#17162B] transition-[transform,border-color] hover:-translate-y-0.5 hover:border-[#17162B] disabled:opacity-40"
  >
    {children}
  </button>
);

export const Controls = ({
  meta,
  shown,
  total,
  playing,
  live,
  speed,
  onPlay,
  onStep,
  onReset,
  onSpeed,
}: {
  meta: Meta;
  shown: number;
  total: number;
  playing: boolean;
  live: boolean; // rounds are still arriving from the API
  speed: number;
  onPlay: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeed: (s: number) => void;
}) => {
  const done = shown >= total;
  return (
    <div className="card flex flex-wrap items-center gap-3 p-3 pl-4" data-testid="attack-controls">
      <button
        data-testid="attack-play-button"
        onClick={onPlay}
        className="flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-bold text-white shadow-md transition-transform hover:-translate-y-0.5"
        style={{ background: meta.color }}
      >
        {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        {playing ? "Pause" : done ? "Replay" : shown ? "Resume" : "Launch attack"}
      </button>
      <IconBtn testid="attack-step-button" label="Step" onClick={onStep} disabled={done}>
        <StepForward className="h-4 w-4" />
      </IconBtn>
      <IconBtn testid="attack-reset-button" label="Reset" onClick={onReset}>
        <RotateCcw className="h-4 w-4" />
      </IconBtn>
      <div className="flex rounded-full bg-[#EFECE4] p-1" data-testid="attack-speed-group">
        {SPEEDS.map((s) => (
          <button
            key={s}
            data-testid={`attack-speed-${s}x`}
            onClick={() => onSpeed(s)}
            className={`rounded-full px-3 py-1 font-mono text-xs font-bold transition-colors ${
              speed === s ? "bg-white text-[#17162B] shadow-sm" : "text-[#6B6880]"
            }`}
          >
            {s}x
          </button>
        ))}
      </div>
      <div
        className="ml-auto flex items-center gap-3 font-mono text-xs text-[#4A4760]"
        data-testid="attack-progress"
      >
        {(playing || live) && (
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
};
