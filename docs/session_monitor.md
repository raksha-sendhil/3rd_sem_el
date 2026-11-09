# Session privacy monitor

The protected pipeline never shows the answer model a name, phone or ID: Agents 1-3 replace
them before anything is stored. That stops a single question from leaking an identity. It does
not stop a **sequence** of questions. Each answer can carry a detail that is harmless on its
own (a common condition, "in their 30s", "a town in the Nilgiris district", "a healthcare
worker"), and together they leave one person. An attacker who already knows the person (a
neighbour, a colleague) then knows which record is theirs, and with it the diagnosis.

The session monitor is the defence against that. It sits on the protected pipeline, after the
answer model and before the answer leaves the API (`src/sanitise/monitor/`).

## Why rewriting alone is not enough

* Zhao et al. (2026), the multi-agent rewriting method this project builds on, test a
  *stepwise inference attack* on their own pipeline (section 5.1, Table 4): leakage rises from
  1 to 3 entities, because "multi-query aggregation can partially bypass the High-level
  Abstraction strategy". As future work they name "cross-query context tracking". This monitor
  is that tracking.
* Sun et al. (2026, IEEE TDSC), where the Secret Recovery Rate comes from: existing methods
  "treat privacy as a per-query or per-response constraint", but sensitive information is
  "progressively reconstructed across multi-turn interactions"; privacy should be "a
  history-dependent exposure process".
* Generalising every document harder is not the answer either. Zhao et al. report
  over-sanitisation (useful detail removed) as a failure mode. Rewriting is done once per
  document, offline, and cannot know what a future conversation will already have revealed.

The design splits the job. Rewriting keeps documents useful. The monitor enforces anonymity
(at least k people still matching) only when a session's questions get too specific.

## What it does, per protected answer

Deterministic, no model involved (Phase 2 spec, section 3.3).

1. **What does the draft reveal, about whom?** Each retrieved record's attributes come from
   Agent 3's decisions: kept or generalised details of type age, date, location, occupation,
   condition or physical (`attributes.py`). Symptoms and relations are left out, because they
   describe almost everyone. Placeholders reveal nothing. An attribute counts as revealed when
   its form in the sanitised text, or its core ("Nilgiris", "thirties"), appears in the draft.
   Each one is tied to its person, `<doc_id>:<subject>`, from Agent 2's facts.
2. **Disclosure threshold.** The ledger holds what this session has already been told. If any
   person would now have more than **N = 3** distinct kinds of detail known, the answer is
   **blocked**. Generalising cannot take a kind of detail back.
3. **Re-identification check.** For each person in the answer, count the records in the store
   whose own attributes are consistent with everything disclosed about that person so far,
   plus this answer: the **candidate pool**. Fewer than **k = 5** is too few.
4. **Router: generalise before blocking.** Over the pool limit, the newly revealed details
   climb Agent 3's generalisation ladder (`ontology.generalise`), one step at a time, up to
   `max_climb` = 2 steps. For example "a town in the Nilgiris district" becomes "a town in
   Tamil Nadu", then "a town in the region"; "in their 30s" becomes "a young adult", then "an
   adult". The last step is the generic phrase, which reveals nothing. If the pool is back to k
   or more, the answer is returned **generalised**. Otherwise it is **blocked**.
5. **Record.** Only what is actually released goes into the ledger.

Details already disclosed are never climbed: the session has them, so re-saying them more
vaguely changes nothing.

### The ledger

SQLite, as the spec defines it:

```
disclosures(session_id, subject_id, attribute_type, round, value_hash)
```

Values are never stored. `value_hash` is an HMAC-SHA256 of the attribute type and the released
phrase, under a secret salt (`SANITISE_MONITOR_SALT`; without one a random salt is drawn per
process). A copy of the ledger tells nobody what was disclosed, yet the monitor can still match
a released phrase against every record's own phrases, hashed the same way.

### The pool is counted exactly

The spec sketches the pool as a retrieval query over everything disclosed. The monitor instead
counts over every record in the store. A retrieval top-k can miss a matching record, which
would under-count the pool. The store is small (hundreds of records), so the exact count is
cheap.

## Settings

| Variable (`SANITISE_` prefix) | Default | Meaning |
|---|---|---|
| `MONITOR` | `ledger` | `off` = allow everything (for comparison) |
| `MONITOR_K` | 5 | fewest records that must still match a person |
| `MONITOR_MAX_ATTRIBUTES` | 3 | kinds of detail per person per session |
| `MONITOR_MAX_CLIMB` | 2 | generalisation steps before blocking |
| `MONITOR_DB_PATH` | `data/monitor.sqlite` | the ledger (mock mode keeps it in memory) |
| `MONITOR_SALT` | random per process | key for the value hashes |

## Where it shows

* **API.** Every protected answer carries `monitor`: `verdict` (allowed / generalised /
  blocked), `candidate_pool`, `disclosed` (kinds of detail known about the riskiest person) and
  `reason`. A generalised answer replaces the draft; a blocked one is refused. The draft never
  leaves the server. `GET /v1/sessions/{id}/monitor` lists the ledger: kinds of detail per
  person and the round each was first revealed, never values.
* **Dashboard, Attacks page.**
  * Stepwise inference is enabled.
  * Each protected reply's badge shows the verdict and the pool.
  * The session monitor panel lists every question with the details known and the verdict, and
    charts the pool against k.
  * "Score this attack" plays the whole script and draws the Secret Recovery Rate per round
    for standard RAG, rewriting alone and rewriting + monitor.
* **Evaluation.** `tools/monitor_eval.py` (Kaggle backend notebook, cell 10) plays every attack
  script with the monitor on and reports, for the three set-ups, the final Secret Recovery Rate
  and whether and when the target was identified. It also asks the 30 harmless questions in
  `data/monitor/benign_questions.json`, each in a fresh session and all in one session.
  Refused harmless questions are the over-blocking cost; generalised ones are still answered.

## Limitations

* **Records, not people.** The pool counts matching records in the store, not people in the
  real world. The person table (`data/corpus/people.json`) is the attacker's world and is used
  only for scoring, as the spec intends.
* **One person across documents is not linked.** The sanitised store is deliberately
  unlinkable, so the ledger works per record. An attacker who reaches the same person through
  two different documents is counted twice, with fewer details each time.
* **Per session.** An attacker who opens many sessions starts afresh each time. A per-user or
  per-key ledger is the next step once the API knows who is asking.
* **Detection is by phrase.** A model that paraphrases a detail beyond its core ("in her
  mid-thirties" for "in their 30s") is not caught. Answers are generated from the sanitised
  text, so they usually repeat its phrases.
* **Mock mode** has only the 20 fixtures, in which almost every detail is unique, so nearly
  every detail is generalised. The built corpus (`data/corpus`) was designed with shared
  attributes, which is where the allowed → generalised → blocked progression shows.
