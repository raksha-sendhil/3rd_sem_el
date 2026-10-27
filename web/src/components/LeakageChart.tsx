"use client";

import { useEffect, useRef, useState } from "react";
import { SERIES, SIDES, type SeriesId } from "./common";

const HEIGHT = 220;
const M = { top: 12, right: 112, bottom: 30, left: 46 };
const LABEL_FONT = "var(--font-label), monospace";

type Props = {
  rounds: number; // length of the x axis
  values: Record<SeriesId, number[]>; // one value per round received so far
  yMax: number;
  ticks: number[];
  format: (v: number) => string;
  label: string; // for screen readers
};

/** One leakage measure per round, one line per pipeline, directly labelled. */
export function LeakageChart({ rounds, values, yMax, ticks, format, label }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(260, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const plotW = width - M.left - M.right;
  const plotH = HEIGHT - M.top - M.bottom;
  const x = (i: number) => M.left + (rounds <= 1 ? plotW / 2 : (i / (rounds - 1)) * plotW);
  const y = (v: number) => M.top + plotH - (v / Math.max(yMax, 1e-9)) * plotH;
  const shown = Math.max(values.baseline.length, values.protected.length);
  const last = shown - 1;

  // direct labels at the newest point, pushed apart when they would collide
  const labelY = { baseline: 0, protected: 0 };
  if (last >= 0) {
    for (const s of SIDES) labelY[s] = y(values[s][last] ?? 0);
    if (Math.abs(labelY.baseline - labelY.protected) < 18) {
      const mid = (labelY.baseline + labelY.protected) / 2;
      const top: SeriesId =
        (values.baseline[last] ?? 0) >= (values.protected[last] ?? 0) ? "baseline" : "protected";
      labelY[top] = mid - 9;
      labelY[top === "baseline" ? "protected" : "baseline"] = mid + 9;
    }
  }

  function onMove(e: React.MouseEvent<SVGRectElement>) {
    if (shown === 0) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left + M.left - 16;
    let best = 0;
    for (let i = 1; i < shown; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    setHover(best);
  }

  const flip = hover !== null && x(hover) + 210 > width;

  return (
    <div className="chart-wrap" ref={wrap}>
      <svg width={width} height={HEIGHT} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={M.left}
              x2={M.left + plotW}
              y1={y(t)}
              y2={y(t)}
              stroke={t === 0 ? "#141210" : "#d9cfbf"}
              strokeWidth={2}
              strokeDasharray={t === 0 ? undefined : "4 4"}
              shapeRendering="crispEdges"
            />
            <text
              x={M.left - 10}
              y={y(t) + 4}
              textAnchor="end"
              fontSize={11}
              fontFamily={LABEL_FONT}
              fill="#5e574d"
            >
              {format(t)}
            </text>
          </g>
        ))}
        {Array.from({ length: rounds }, (_, i) => (
          <text
            key={i}
            x={x(i)}
            y={M.top + plotH + 20}
            textAnchor="middle"
            fontSize={11}
            fontFamily={LABEL_FONT}
            fill="#5e574d"
          >
            {`R${i + 1}`}
          </text>
        ))}

        {hover !== null && (
          <line
            x1={x(hover)}
            x2={x(hover)}
            y1={M.top}
            y2={M.top + plotH}
            stroke="#8a6bff"
            strokeWidth={2}
            strokeDasharray="4 4"
            shapeRendering="crispEdges"
          />
        )}

        {SIDES.map((s) => {
          const pts = values[s].map((v, i) => [x(i), y(v)] as const);
          const size = (i: number) => (hover === i ? 14 : 11);
          return (
            <g key={s}>
              {pts.length > 1 && (
                <polyline
                  points={pts.map((p) => p.join(",")).join(" ")}
                  fill="none"
                  stroke={SERIES[s].color}
                  strokeWidth={3}
                  strokeLinejoin="miter"
                />
              )}
              {pts.map(([px, py], i) => (
                <rect
                  key={i}
                  x={px - size(i) / 2}
                  y={py - size(i) / 2}
                  width={size(i)}
                  height={size(i)}
                  fill={SERIES[s].color}
                  stroke="#141210"
                  strokeWidth={2}
                  shapeRendering="crispEdges"
                />
              ))}
            </g>
          );
        })}

        {last >= 0 &&
          SIDES.map((s) => (
            <text
              key={s}
              x={x(last) + 14}
              y={labelY[s] + 4}
              fontSize={11}
              fontFamily={LABEL_FONT}
              fill="#141210"
            >
              {SERIES[s].name}
            </text>
          ))}

        <rect
          x={M.left - 16}
          y={M.top}
          width={plotW + 32}
          height={plotH}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>

      {hover !== null && (
        <div
          className="chart-tip"
          style={flip ? { right: width - x(hover) + 14 } : { left: x(hover) + 14 }}
        >
          <div className="lbl" style={{ marginBottom: 4 }}>
            Round {hover + 1}
          </div>
          {SIDES.map((s) => (
            <div className="row" key={s}>
              <span className="sq" data-series={s} aria-hidden />
              {SERIES[s].name}
              <b>{values[s][hover] === undefined ? "-" : format(values[s][hover])}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
