import type { ReactNode } from "react";

export const PageHead = ({
  index,
  label,
  title,
  children,
  color,
}: {
  index: string;
  label: string;
  title: ReactNode;
  children?: ReactNode;
  color: string;
}) => (
  <div className="rise">
    <div className="mb-4 flex items-center gap-3 font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#6B6880]">
      <span className="rounded-md px-2 py-1 text-white" style={{ background: color }}>
        {index}
      </span>
      {label}
    </div>
    <h1 className="font-display text-4xl font-extrabold leading-[1.02] tracking-tight text-[#17162B] sm:text-5xl lg:text-6xl">
      {title}
    </h1>
    {children}
  </div>
);

export const Legend = () => (
  <div className="mt-6 flex flex-wrap gap-2 text-xs" data-testid="highlight-legend">
    <span className="mark-leak">leaked detail</span>
    <span className="mark-token">[TOKEN]</span>
    <span className="mark-vague">generalised</span>
  </div>
);

export const ErrorNote = ({ error }: { error: string | null }) =>
  error ? (
    <div
      role="alert"
      className="rounded-2xl border border-[#FFB4AB] bg-[#FFE8E4] px-5 py-3 text-sm font-medium text-[#B42318]"
    >
      {error}
    </div>
  ) : null;
