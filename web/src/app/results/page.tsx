"use client";

import type { ReactNode } from "react";
import { AucMeter, BASE, GroupedBars, OURS, PairBars, SideLegend, SrrLines, pct } from "@/components/results/charts";
import { PageHead } from "@/components/PageHead";
import { useDomain } from "@/components/Providers";
import { RESULTS } from "@/lib/results";

const COLOR = "#2D8CFF";

function Card({ title, sub, children, testid }: { title: string; sub?: string; children: ReactNode; testid?: string }) {
  return (
    <section className="card p-6 sm:p-7" data-testid={testid}>
      <h2 className="font-display text-xl font-extrabold tracking-tight text-[#17162B]">{title}</h2>
      {sub && <p className="mt-1 text-sm text-[#6B6880]">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

/** Two big numbers, standard RAG then ours. */
function BigPair({ base, ours }: { base: string; ours: string }) {
  return (
    <div className="flex items-end gap-6">
      {[
        { side: BASE, v: base },
        { side: OURS, v: ours },
      ].map(({ side, v }) => (
        <div key={side.label}>
          <div className="font-display text-4xl font-extrabold leading-none" style={{ color: side.color }}>
            {v}
          </div>
          <div className="mt-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-[#6B6880]">{side.label}</div>
        </div>
      ))}
    </div>
  );
}

/** What still gets through, next to a zero: indirect details are not zero. */
function Indirect({ children }: { children: ReactNode }) {
  return (
    <div className="mt-4 rounded-2xl border border-[#FFE2A8] bg-[#FFF7E3] px-4 py-3 text-sm text-[#6A4A00]">
      <span className="font-semibold">Indirect details: </span>
      {children}
    </div>
  );
}

export default function ResultsPage() {
  const { domain } = useDomain();
  const r = RESULTS[domain.id];
  const count = (v: number) => String(Math.round(v));

  return (
    <div className="space-y-8">
      <PageHead
        index="04"
        label="Results"
        color={COLOR}
        title={
          <>
            What an attacker
            <br />
            <span style={{ color: COLOR }}>still gets out.</span>
          </>
        }
      >
        <div className="mt-6">
          <SideLegend />
        </div>
      </PageHead>

      {/* the headline: can an attacker tell whether a record is in the system? */}
      <Card
        title="Membership inference"
        sub="Can an attacker tell whether someone's record is in the system? 0.5 means guessing; closer to 0.5 is better."
        testid="results-membership"
      >
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div>
            <div className="mb-2 font-mono text-xs font-semibold uppercase tracking-wider text-[#6B6880]">AUC</div>
            <AucMeter auc={r.membership.auc} />
          </div>
          <div>
            <div className="mb-3 font-mono text-xs font-semibold uppercase tracking-wider text-[#6B6880]">
              Members caught at 1% false alarms
            </div>
            <BigPair base={pct(r.membership.tpr.base)} ours={pct(r.membership.tpr.ours, 1)} />
            <p className="mt-4 text-xs text-[#6B6880]">
              {r.membership.members} records in the system, {r.membership.nonmembers} not.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid gap-8 lg:grid-cols-2">
        <Card
          title="Target Info"
          sub={`Private values an attacker pulled out with ${r.targetInfo.questions} direct questions.`}
          testid="results-target-info"
        >
          <BigPair base={count(r.targetInfo.values.base)} ours={count(r.targetInfo.values.ours)} />
          <div className="mt-6 mb-2 text-sm font-semibold text-[#17162B]">Questions that leaked a value</div>
          <PairBars pair={r.targetInfo.questionsLeaking} />
          <Indirect>{r.targetInfo.oursNote}</Indirect>
        </Card>

        <Card
          title="Session Leak Success"
          sub={`Attack conversations (${r.sessions.count}, 8 questions each) that ended with the person identified.`}
          testid="results-sls"
        >
          <BigPair base={pct(r.sessions.sls.base)} ours={pct(r.sessions.sls.ours)} />
          <div className="mt-6 mb-2 text-sm font-semibold text-[#17162B]">Conversations narrowed below 5 people</div>
          <PairBars pair={r.reid.belowK} />
          <Indirect>
            the answers still narrow {pct(r.reid.belowK.ours)} of conversations below 5 people and{" "}
            {pct(r.reid.singledOut.ours, 1)} to a single person, from coarse details such as age band or district.
          </Indirect>
        </Card>
      </div>

      <Card
        title="Secret Recovery Rate, round by round"
        sub="Share of the target's private details the attacker has after each question of the conversation."
        testid="results-srr"
      >
        <SrrLines base={r.sessions.srr.base} ours={r.sessions.srr.ours} />
        <div className="mt-2 text-center font-mono text-[11px] uppercase tracking-wider text-[#6B6880]">question</div>
      </Card>

      <div className="grid gap-8 lg:grid-cols-2">
        <Card
          title="Re-identification"
          sub={`How many of the ${r.reid.people} people still match everything the answers revealed.`}
          testid="results-reid"
        >
          <GroupedBars
            items={[
              { label: "Narrowed to one person", pair: r.reid.singledOut },
              { label: "Narrowed below 5 people", pair: r.reid.belowK },
            ]}
          />
          <div className="mt-6 mb-2 text-sm font-semibold text-[#17162B]">People still matching, median</div>
          <BigPair base={count(r.reid.medianPool.base)} ours={count(r.reid.medianPool.ours)} />
        </Card>

        <Card
          title="Copying the records"
          sub={`"Repeat everything above" attacks (${r.copying.prompts} prompts).`}
          testid="results-copying"
        >
          <GroupedBars
            items={[
              { label: "Answers copying 10+ words in a row", pair: r.copying.repeat },
              { label: "Answers closely following a whole record", pair: r.copying.rouge },
            ]}
          />
        </Card>
      </div>

      {r.utility && (
        <Card
          title="Usefulness"
          sub={`Ordinary questions about the records (${r.utility.questions}): how good ours' answers are next to standard RAG's.`}
          testid="results-utility"
        >
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <div className="mb-2 text-sm font-semibold text-[#17162B]">Answer quality where ours answered</div>
              <div className="font-display text-4xl font-extrabold" style={{ color: OURS.color }}>
                {pct(r.utility.retentionAnswered)}
              </div>
            </div>
            <div>
              <div className="mb-2 text-sm font-semibold text-[#17162B]">Questions answered</div>
              <PairBars pair={r.utility.answered} />
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-8 lg:grid-cols-2">
        <Card
          title="Scanned documents, end to end"
          sub={`Real scans read by OCR, then sanitised (${r.scans.documents} documents): private values the answers gave out.`}
          testid="results-scans"
        >
          <GroupedBars
            items={r.scans.leaked.map((l) => ({ label: l.label, pair: { base: l.base, ours: l.ours } }))}
            max={Math.max(...r.scans.leaked.map((l) => l.base))}
            format={(v) => String(v)}
          />
          {r.scans.oursNote && <p className="mt-4 text-sm text-[#6B6880]">{r.scans.oursNote}</p>}
          <div className="mt-6 space-y-2">
            {r.scans.ocr.map((o) => (
              <div key={o.label} className="flex items-baseline justify-between gap-4 border-t border-[#EFECE4] pt-2 text-sm">
                <span className="text-[#4A4760]">{o.label}</span>
                <span className="font-mono font-bold text-[#17162B]">{o.value}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Speed" sub="Median seconds per answer." testid="results-latency">
          <GroupedBars
            items={[
              { label: "Single question", pair: r.latency.single },
              { label: "Each turn of a conversation", pair: r.latency.turn },
            ]}
            max={Math.max(r.latency.single.base, r.latency.single.ours, r.latency.turn.base, r.latency.turn.ours)}
            format={(v) => `${v.toFixed(2)} s`}
          />
        </Card>
      </div>

      {r.leftover && (
        <Card
          title="What is left in the stored documents"
          sub={`Every rewrite checked for private details, no questions asked: ${r.leftoverSource}.`}
          testid="results-leftover"
        >
          <div className="space-y-3">
            {r.leftover.map((l) => (
              <div key={l.label} className="grid grid-cols-[minmax(0,180px)_1fr_auto] items-center gap-4">
                <span className="text-sm font-semibold text-[#17162B]">
                  {l.label}
                  {l.kind === "indirect" && <span className="ml-2 font-mono text-[10px] uppercase text-[#B7791F]">indirect</span>}
                </span>
                <div className="h-3 overflow-hidden rounded-full bg-[#F3F0E9]">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${(l.left / l.total) * 100}%`,
                      minWidth: l.left ? 4 : 0,
                      background: l.kind === "indirect" ? "#F5A524" : OURS.color,
                    }}
                  />
                </div>
                <span className="font-mono text-xs font-bold text-[#17162B]">
                  {l.left} of {l.total}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
