"use client";

import { useEffect, useRef, useState } from "react";
import { SERIES, type SeriesId } from "./common";

const HEIGHT = 220;
const M = { top: 12, right: 112, bottom: 36, left: 36 };
const ORDER: SeriesId[] = ["baseline", "protected"];

type Props = {
  title: string;
  sub: string;
  rounds: number; // questions in the attack (x axis length)
  values: Record<SeriesId, number[]>; // one value per question shown so far
  yMax: number;
  unit: (v: number) => string; // "3 details", "1 person"
};

function ticks(max: number): number[] {
  const step = max <= 5 ? 1 : max <= 10 ? 2 : Math.ceil(max / 5);
  const out = [];
  for (let v = 0; v <= max; v += step) out.push(v);
  return out;
}

/** Line chart of one leakage measure per question, for both pipelines. */
export function LeakageChart({ title, sub, rounds, values, yMax, unit }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const plotW = width - M.left - M.right;
  const plotH = HEIGHT - M.top - M.bottom;
  const x = (i: number) => M.left + (rounds <= 1 ? plotW / 2 : (i / (rounds - 1)) * plotW);
  const y = (v: number) => M.top + plotH - (v / Math.max(yMax, 1)) * plotH;
  const shown = Math.max(values.baseline.length, values.protected.length);

  // direct labels at the last point, pushed apart when they would overlap
  const last = shown - 1;
  const labelY: Record<SeriesId, number> = { baseline: 0, protected: 0 };
  if (last >= 0) {
    for (const s of ORDER) labelY[s] = y(values[s][last] ?? 0);
    if (Math.abs(labelY.baseline - labelY.protected) < 16) {
      const mid = (labelY.baseline + labelY.protected) / 2;
      const upper: SeriesId =
        (values.baseline[last] ?? 0) >= (values.protected[last] ?? 0) ? "baseline" : "protected";
      labelY[upper] = mid - 8;
      labelY[upper === "baseline" ? "protected" : "baseline"] = mid + 8;
    }
  }

  function onMove(e: React.MouseEvent<SVGRectElement>) {
    if (shown === 0) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left + M.left;
    let best = 0;
    for (let i = 1; i < shown; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    setHover(best);
  }

  const tipLeft = hover === null ? 0 : x(hover) + 12;
  const flip = tipLeft + 180 > width;

  return (
    <figure style={{ margin: 0 }}>
      <figcaption>
        <div className="chart-title">{title}</div>
        <div className="chart-sub">{sub}</div>
      </figcaption>
      <div className="chart-wrap" ref={wrap}>
        <svg width={width} height={HEIGHT} role="img" aria-label={`${title}. ${sub}`}>
          {ticks(yMax).map((t) => (
            <g key={t}>
              <line
                x1={M.left}
                x2={M.left + plotW}
                y1={y(t)}
                y2={y(t)}
                stroke={t === 0 ? "#8d8d8d" : "#e0e0e0"}
                strokeWidth={1}
                shapeRendering="crispEdges"
              />
              <text x={M.left - 10} y={y(t) + 4} textAnchor="end" fontSize={12} fill="#525252">
                {t}
              </text>
            </g>
          ))}
          {Array.from({ length: rounds }, (_, i) => (
            <text
              key={i}
              x={x(i)}
              y={M.top + plotH + 20}
              textAnchor="middle"
              fontSize={12}
              fill="#525252"
            >
              {`Q${i + 1}`}
            </text>
          ))}

          {hover !== null && (
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={M.top}
              y2={M.top + plotH}
              stroke="#8d8d8d"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
          )}

          {ORDER.map((s) => {
            const pts = values[s].map((v, i) => [x(i), y(v)] as const);
            return (
              <g key={s}>
                {pts.length > 1 && (
                  <polyline
                    points={pts.map((p) => p.join(",")).join(" ")}
                    fill="none"
                    stroke={SERIES[s].color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                  />
                )}
                {pts.map(([px, py], i) => (
                  <circle
                    key={i}
                    cx={px}
                    cy={py}
                    r={hover === i ? 5.5 : 4}
                    fill={SERIES[s].color}
                    stroke="#ffffff"
                    strokeWidth={2}
                  />
                ))}
              </g>
            );
          })}

          {last >= 0 &&
            ORDER.map((s) => (
              <text
                key={s}
                x={x(last) + 12}
                y={labelY[s] + 4}
                fontSize={12}
                fill="#161616"
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
            style={flip ? { right: width - x(hover) + 12 } : { left: tipLeft }}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Question {hover + 1}</div>
            {ORDER.map((s) => (
              <div className="row" key={s}>
                <span className="swatch" data-series={s} aria-hidden />
                {SERIES[s].name}
                <b>{values[s][hover] === undefined ? "-" : unit(values[s][hover])}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
