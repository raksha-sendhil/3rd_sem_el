import { AGENTS } from "@/lib/agents";

const Node = ({
  x,
  y,
  w = 92,
  label,
  sub,
  fill,
  text = "#fff",
}: {
  x: number;
  y: number;
  w?: number;
  label: string;
  sub: string;
  fill: string;
  text?: string;
}) => (
  <g transform={`translate(${x},${y})`}>
    <rect width={w} height="46" rx="12" fill={fill} />
    <rect width={w} height="46" rx="12" fill="url(#gloss)" />
    <text x={w / 2} y="21" textAnchor="middle" fontSize="12.5" fontWeight="700" fill={text} className="font-display">
      {label}
    </text>
    <text x={w / 2} y="35" textAnchor="middle" fontSize="8.5" fill={text} opacity="0.85" className="font-mono">
      {sub}
    </text>
  </g>
);

const width = (label: string) => label.length * 6.4 + 12;

const Chip = ({ x, y, label, fill, color }: { x: number; y: number; label: string; fill: string; color: string }) => (
  <g transform={`translate(${x},${y})`}>
    <rect width={width(label)} height="18" rx="9" fill={fill} />
    <text x="6" y="12.5" fontSize="9" fill={color} className="font-mono" fontWeight="600">
      {label}
    </text>
  </g>
);

const cut = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

export type Sample = {
  leaks: string[]; // details of one record as Standard RAG passes them on
  token?: string; // the same record after Agent 3
  vague?: string;
};

export const PipelineGraphic = ({ records, sample }: { records: number | null; sample: Sample | null }) => {
  const leaks = (sample?.leaks ?? []).slice(0, 4).map((l) => cut(l, 11));
  return (
    <svg
      viewBox="0 0 760 250"
      className="h-auto w-full"
      data-testid="pipeline-graphic"
      role="img"
      aria-label="Standard and protected RAG pipelines"
    >
      <defs>
        <linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fff" stopOpacity="0.28" />
          <stop offset="0.5" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <g stroke="#CFC8B8" strokeWidth="2" fill="none">
        <path d="M112 125 C150 125 150 50 190 50 L430 50" />
        <path d="M112 125 C150 125 150 190 190 190" />
        <path d="M282 190 H300 M392 190 H410 M502 190 H520" />
        <path d="M522 50 H560 M612 190 H620 C640 190 640 190 640 190" />
      </g>
      <g fill="none" strokeWidth="3" strokeLinecap="round">
        <path className="flow" stroke="#FF4D3D" d="M112 125 C150 125 150 50 190 50 L430 50" />
        <path className="flow" stroke="#0FB5A0" d="M112 125 C150 125 150 190 190 190" />
        <path className="flow" stroke="#0FB5A0" d="M282 190 H300 M392 190 H410 M502 190 H520" />
        <path className="flow" stroke="#FF4D3D" d="M522 50 H560" />
        <path className="flow" stroke="#0FB5A0" d="M612 190 H640" />
      </g>
      <g transform="translate(18,92)">
        <rect x="8" y="-6" width="86" height="66" rx="12" fill="#FFE3A1" />
        <rect x="4" y="-2" width="86" height="66" rx="12" fill="#FFD266" />
        <rect width="94" height="66" rx="12" fill="#FFC53D" />
        <text x="47" y="30" textAnchor="middle" fontSize="13" fontWeight="800" fill="#5A3B00" className="font-display">
          Records
        </text>
        <text x="47" y="45" textAnchor="middle" fontSize="8.5" fill="#5A3B00" className="font-mono">
          {records === null ? "…" : `${records.toLocaleString()} private`}
        </text>
      </g>
      <text x="190" y="20" fontSize="10" fill="#B42318" className="font-mono" fontWeight="700" letterSpacing="1.5">
        STANDARD RAG
      </text>
      <text x="190" y="242" fontSize="10" fill="#08776A" className="font-mono" fontWeight="700" letterSpacing="1.5">
        PROTECTED RAG
      </text>
      <Node x={190} y={167} label={AGENTS.detect.name} sub="agent 1" fill={AGENTS.detect.color} />
      <Node x={300} y={167} label={AGENTS.extract.name} sub="agent 2" fill={AGENTS.extract.color} />
      <Node x={410} y={167} label={AGENTS.rewrite.name} sub="agent 3" fill="#F5A300" />
      <Node x={430} y={27} label="LLM" sub="raw context" fill="#17162B" />
      <Node x={520} y={167} label="LLM" sub="safe context" fill="#17162B" />
      <g transform="translate(560,18)">
        <rect width="186" height="64" rx="14" fill="#fff" stroke="#FFB4AB" strokeWidth="1.5" />
        {leaks.map((l, i) => (
          <Chip
            key={i}
            x={i % 2 === 0 ? 10 : 10 + width(leaks[i - 1]) + 6}
            y={i < 2 ? 10 : 36}
            label={l}
            fill="#FFE8E4"
            color="#B42318"
          />
        ))}
      </g>
      <g transform="translate(640,158)">
        <rect width="112" height="64" rx="14" fill="#fff" stroke="#9FE6DA" strokeWidth="1.5" />
        {sample?.token && <Chip x={10} y={10} label={cut(sample.token, 13)} fill="#EEE9FF" color="#4B2FD6" />}
        {sample?.vague && <Chip x={10} y={36} label={cut(sample.vague, 13)} fill="#FFF3D1" color="#8A5A00" />}
      </g>
    </svg>
  );
};
