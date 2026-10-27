"use client";

import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type ChartPoint = {
  label: string;
  std: number;
  prot: number;
  stdMatch?: number;
  protMatch?: number;
};

// the axis font is set in globals.css: SVG attributes cannot read the font variable
const axis = { fontSize: 11, fill: "#6B6880" };
const tip = {
  contentStyle: {
    borderRadius: 12,
    border: "1px solid #E6E1D6",
    fontFamily: "var(--font-mono)",
    fontSize: 12,
  },
};

const Frame = ({
  title,
  sub,
  children,
  testid,
}: {
  title: string;
  sub: string;
  children: ReactNode;
  testid: string;
}) => (
  <div className="card p-5" data-testid={testid}>
    <h3 className="font-display text-base font-bold text-[#17162B]">{title}</h3>
    <p className="mb-3 font-mono text-[11px] text-[#6B6880]">{sub}</p>
    <div className="h-56">{children}</div>
  </div>
);

export const AttackCharts = ({ data, facts }: { data: ChartPoint[]; facts: number | null }) => {
  const pools = data.flatMap((d) => [d.stdMatch ?? 1, d.protMatch ?? 1]);
  const top = 10 ** Math.max(1, Math.ceil(Math.log10(Math.max(...pools, 1))));
  const ticks = Array.from({ length: Math.log10(top) + 1 }, (_, i) => 10 ** i);
  return (
    <div className="grid gap-6 md:grid-cols-2" data-testid="attack-charts-container">
      <Frame
        title="Target details recovered"
        sub={
          facts
            ? `% of the ${facts} known facts the attacker now holds`
            : "% of the target's known facts the attacker now holds"
        }
        testid="chart-recovered"
      >
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ left: -18, right: 8, top: 8 }}>
            <defs>
              <linearGradient id="gStd" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#FF4D3D" stopOpacity={0.35} />
                <stop offset="1" stopColor="#FF4D3D" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gProt" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#0FB5A0" stopOpacity={0.35} />
                <stop offset="1" stopColor="#0FB5A0" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#EFEBE2" vertical={false} />
            <XAxis dataKey="label" tick={axis} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} tick={axis} axisLine={false} tickLine={false} unit="%" />
            <Tooltip {...tip} />
            <Area
              type="monotone"
              dataKey="std"
              name="Standard"
              stroke="#FF4D3D"
              strokeWidth={3}
              fill="url(#gStd)"
              animationDuration={500}
            />
            <Area
              type="monotone"
              dataKey="prot"
              name="Protected"
              stroke="#0FB5A0"
              strokeWidth={3}
              fill="url(#gProt)"
              animationDuration={500}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Frame>
      <Frame
        title="People who still match"
        sub="candidate pool, log scale — 1 means identified"
        testid="chart-matches"
      >
        <ResponsiveContainer>
          <LineChart data={data} margin={{ left: -8, right: 8, top: 8 }}>
            <CartesianGrid stroke="#EFEBE2" vertical={false} />
            <XAxis dataKey="label" tick={axis} axisLine={false} tickLine={false} />
            <YAxis
              scale="log"
              domain={[1, top]}
              ticks={ticks}
              tick={axis}
              axisLine={false}
              tickLine={false}
              allowDataOverflow
            />
            <Tooltip {...tip} />
            <Line
              type="stepAfter"
              dataKey="stdMatch"
              name="Standard"
              stroke="#FF4D3D"
              strokeWidth={3}
              dot={{ r: 4, fill: "#FF4D3D" }}
              animationDuration={500}
            />
            <Line
              type="stepAfter"
              dataKey="protMatch"
              name="Protected"
              stroke="#0FB5A0"
              strokeWidth={3}
              dot={{ r: 4, fill: "#0FB5A0" }}
              animationDuration={500}
            />
          </LineChart>
        </ResponsiveContainer>
      </Frame>
    </div>
  );
};
