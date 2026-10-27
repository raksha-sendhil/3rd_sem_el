"use client";

import type { AttackDefinition } from "@/api/client";
import { metaOf } from "./meta";

export const AttackPicker = ({
  attacks,
  value,
  onChange,
}: {
  attacks: AttackDefinition[];
  value: string | null;
  onChange: (id: string) => void;
}) => (
  <div className="grid gap-4 md:grid-cols-3" data-testid="attack-selector-cards">
    {attacks.map((a, i) => {
      const m = metaOf(a);
      const on = a.attack_id === value;
      return (
        <button
          key={a.attack_id}
          data-testid={`attack-card-${a.attack_id}`}
          onClick={() => onChange(a.attack_id)}
          className="card rise group relative flex items-center gap-4 overflow-hidden p-4 text-left transition-[transform,box-shadow] duration-300 hover:-translate-y-1"
          style={{
            animationDelay: `${i * 80}ms`,
            boxShadow: on ? `0 0 0 3px ${m.color}, 0 18px 40px -18px ${m.color}` : undefined,
          }}
        >
          <div className="grid h-24 w-24 shrink-0 place-items-center rounded-2xl" style={{ background: m.bg }}>
            {m.image && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={m.image}
                alt=""
                className="h-24 w-24 object-contain mix-blend-multiply transition-transform duration-500 group-hover:rotate-3 group-hover:scale-110"
              />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span
                className="font-mono text-[10px] font-bold uppercase tracking-[0.18em]"
                style={{ color: m.color }}
              >
                0{i + 1}
              </span>
              <span className="chip" style={{ background: m.bg, color: m.color }}>
                {m.tag}
              </span>
            </div>
            <h3 className="mt-1 font-display text-lg font-bold text-[#17162B]">{a.name}</h3>
            <p className="mt-1 text-xs leading-relaxed text-[#6B6880]">{m.blurb}</p>
          </div>
        </button>
      );
    })}
  </div>
);
