"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, streamAttack, type AttackDefinition, type AttackRound } from "@/api/client";
import { AttackCharts, type ChartPoint } from "@/components/attacks/AttackCharts";
import { AttackPicker } from "@/components/attacks/AttackPicker";
import { Controls } from "@/components/attacks/Controls";
import { ORDER, leakOf, metaOf, pctOf } from "@/components/attacks/meta";
import { RoundsTable } from "@/components/attacks/RoundsTable";
import { TargetCard } from "@/components/attacks/TargetCard";
import { ErrorNote, PageHead } from "@/components/PageHead";
import { useTerms } from "@/lib/marks";

export default function Attacks() {
  const [attacks, setAttacks] = useState<AttackDefinition[]>([]);
  const [id, setId] = useState<string | null>(null);
  const [rounds, setRounds] = useState<AttackRound[]>([]); // shown as each one is answered
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stop = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .attacks()
      .then((list) => {
        const sorted = [...list].sort((a, b) => ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind));
        setAttacks(sorted);
        setId((cur) => cur ?? (sorted.find((a) => a.kind === "stepwise") ?? sorted[0])?.attack_id ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    return () => stop.current?.();
  }, []);

  const attack = attacks.find((a) => a.attack_id === id) ?? null;
  const terms = useTerms(rounds.flatMap((r) => [...r.baseline.doc_ids, ...r.protected.doc_ids]));

  function launch() {
    if (!attack) return;
    stop.current?.();
    setRounds([]);
    setError(null);
    setRunning(true);
    stop.current = streamAttack(attack.attack_id, {
      round: (r) => setRounds((rs) => [...rs, r]),
      done: () => setRunning(false),
      fail: (message) => {
        setRunning(false);
        setError(message);
      },
    });
  }

  const select = (next: string) => {
    if (running) return;
    stop.current?.();
    setRounds([]);
    setError(null);
    setId(next);
  };

  // a new array restarts the chart animation, so rebuild it only when a round arrives
  const chart: ChartPoint[] = useMemo(
    () => [
      { label: "R0", std: 0, prot: 0 },
      ...rounds.map((r, i) => ({
        label: `R${i + 1}`,
        std: pctOf(leakOf(r, "std")),
        prot: pctOf(leakOf(r, "prot")),
        stdMatch: leakOf(r, "std")?.candidate_pool,
        protMatch: leakOf(r, "prot")?.candidate_pool,
      })),
    ],
    [rounds],
  );

  return (
    <div className="space-y-8">
      <PageHead
        index="02"
        label="Attacks"
        color="#FF4D3D"
        title={
          <>
            Try to break it.
            <br />
            <span className="text-[#6D4AFF]">Watch who gives in.</span>
          </>
        }
      />
      <AttackPicker attacks={attacks} value={id} onChange={select} />
      <ErrorNote error={error} />
      {attack && (
        <>
          <Controls
            meta={metaOf(attack)}
            shown={rounds.length}
            total={attack.queries.length}
            running={running}
            onLaunch={launch}
          />
          <div className="grid gap-6 lg:grid-cols-2 xl:grid-cols-[1fr_1fr_340px]">
            <AttackCharts data={chart} />
            <TargetCard targetId={attack.target_person_id} rows={rounds} />
          </div>
          <RoundsTable
            attackId={attack.attack_id}
            meta={metaOf(attack)}
            intercept={attack.kind === "interception"}
            rows={rounds}
            terms={terms}
          />
        </>
      )}
    </div>
  );
}
