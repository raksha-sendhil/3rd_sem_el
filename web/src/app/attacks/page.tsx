"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ApiError, api, streamAttack, type AttackDefinition, type AttackRound } from "@/api/client";
import { AttackCharts, type ChartPoint } from "@/components/attacks/AttackCharts";
import { AttackPicker } from "@/components/attacks/AttackPicker";
import { Controls } from "@/components/attacks/Controls";
import { ORDER, leakOf, metaOf, pctOf } from "@/components/attacks/meta";
import { RoundsTable } from "@/components/attacks/RoundsTable";
import { TargetCard } from "@/components/attacks/TargetCard";
import { PageHead } from "@/components/PageHead";
import { useTerms } from "@/lib/marks";

type Status = "idle" | "streaming" | "done" | "error";

export default function Attacks() {
  const [attacks, setAttacks] = useState<AttackDefinition[]>([]);
  const [id, setId] = useState<string | null>(null);
  // rounds arrive from the API as each is answered; ``shown`` of them are revealed, paced by speed
  const [rounds, setRounds] = useState<AttackRound[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [shown, setShown] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [queued, setQueued] = useState(0); // step presses waiting for a round to arrive
  const [speed, setSpeed] = useState(1);
  const stop = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .attacks()
      .then((list) => {
        const sorted = [...list].sort((a, b) => ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind));
        setAttacks(sorted);
        setId((cur) => cur ?? (sorted.find((a) => a.kind === "stepwise") ?? sorted[0])?.attack_id ?? null);
      })
      .catch((e) =>
        toast.error("Could not load the attacks", {
          description: e instanceof ApiError ? e.message : String(e),
        }),
      );
    return () => stop.current?.();
  }, []);

  const attack = attacks.find((a) => a.attack_id === id) ?? null;
  const total = status === "done" ? rounds.length : (attack?.queries.length ?? 0);
  const visible = useMemo(() => rounds.slice(0, shown), [rounds, shown]);
  const terms = useTerms(rounds.flatMap((r) => [...r.baseline.doc_ids, ...r.protected.doc_ids]));

  function start() {
    if (!attack) return;
    stop.current?.();
    setRounds([]);
    setShown(0);
    setStatus("streaming");
    stop.current = streamAttack(attack.attack_id, {
      round: (r) => setRounds((rs) => [...rs, r]),
      done: () => setStatus("done"),
      fail: (message) => {
        setStatus("error");
        setPlaying(false);
        setQueued(0);
        toast.error(`${attack.name} failed`, { description: message });
      },
    });
  }

  function reset() {
    stop.current?.();
    stop.current = null;
    setRounds([]);
    setStatus("idle");
    setShown(0);
    setPlaying(false);
    setQueued(0);
  }

  const select = (next: string) => {
    reset();
    setId(next);
  };

  const play = () => {
    if (playing) return setPlaying(false);
    if (status === "idle" || status === "error") start();
    else if (shown >= total) setShown(0);
    setPlaying(true);
  };

  const step = () => {
    if (status === "idle" || status === "error") start();
    setQueued((q) => q + 1);
  };

  // reveal the next round on a timer while playing, waiting for it to arrive if needed
  useEffect(() => {
    if (!playing || !attack) return;
    if (shown >= total && total > 0) {
      setPlaying(false);
      const last = rounds[rounds.length - 1];
      const said = (who: string, hit: boolean | undefined) =>
        `${who} ${hit ? "identified" : "did not identify"} the target.`;
      toast.success(`${attack.name} finished`, {
        description: `${said("Standard RAG", last && leakOf(last, "std")?.identified)} ${said(
          "Protected RAG",
          last && leakOf(last, "prot")?.identified,
        )}`,
      });
      return;
    }
    if (shown >= rounds.length) return;
    const t = setTimeout(() => setShown((s) => s + 1), (shown === 0 ? 500 : 1700) / speed);
    return () => clearTimeout(t);
  }, [playing, shown, total, rounds, speed, attack]);

  useEffect(() => {
    if (queued > 0 && shown < rounds.length) {
      setShown((s) => s + 1);
      setQueued((q) => q - 1);
    }
  }, [queued, shown, rounds.length]);

  // a new array restarts the chart animation, so rebuild it only when a round is revealed
  const chart: ChartPoint[] = useMemo(
    () => [
      { label: "R0", std: 0, prot: 0 },
      ...visible.map((r, i) => ({
        label: `R${i + 1}`,
        std: pctOf(leakOf(r, "std")),
        prot: pctOf(leakOf(r, "prot")),
        stdMatch: leakOf(r, "std")?.candidate_pool,
        protMatch: leakOf(r, "prot")?.candidate_pool,
      })),
    ],
    [visible],
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
      {attack && (
        <>
          <Controls
            meta={metaOf(attack)}
            shown={shown}
            total={total}
            playing={playing}
            live={status === "streaming"}
            speed={speed}
            onPlay={play}
            onStep={step}
            onReset={reset}
            onSpeed={setSpeed}
          />
          <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
            <div className="min-w-0 space-y-6">
              <AttackCharts data={chart} />
              <RoundsTable
                attackId={attack.attack_id}
                meta={metaOf(attack)}
                intercept={attack.kind === "interception"}
                rows={visible}
                terms={terms}
              />
            </div>
            <TargetCard
              targetId={attack.target_person_id}
              rows={visible}
              terms={terms}
              intercept={attack.kind === "interception"}
            />
          </div>
        </>
      )}
    </div>
  );
}
