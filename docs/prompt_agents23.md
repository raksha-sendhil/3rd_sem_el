# Agent 2, Agent 3 and answer prompts - Person C

Code: `src/sanitise/agents/extractor/prompt.py` (Agent 2), `src/sanitise/agents/rewriter/prompt.py`
(Agent 3), `src/sanitise/rag/answer_prompt.py` (both RAG pipelines). Tests: `tests/prompts/`
(MockLLM, no model). Real-model runs: `notebooks/backend.ipynb` cell 2.

## Models

Agent 1's rule pass (Presidio, spaCy, regular expressions) always runs locally. When Agents 2
and 3 run on Gemini, Agent 1's LLM pass does too (`SANITISE_GEMINI_FOR_AGENT1`, on by default):
llama3:8b labelled "ex-smoker" a job, "old" (former) an age and "over 6 months" a combination,
and routing then rewrote Agent 2's correct facts with those labels. On Gemini the LLM pass does
the whole detection (`quasi.FULL_SYSTEM_PROMPT`): every category, every person in the document
(senders, recipients, people mentioned), with the rule spans as hints that may be incomplete or
wrong; the narrow gap-filling prompt stays for local models. On 300 real documents (patient
messages and emails) the gap-filling prompt left Gemini silent on 91 of 150 emails and reached
recall 0.85. The answers of both RAG
pipelines stay on llama3:8b (Ollama). Agents 2 and 3 can run on another model
(`SANITISE_AGENTS_BACKEND`, `SANITISE_AGENTS_MODEL`, set in notebook cell 1):

* `gemini` (default in the notebook): `gemini-3.1-flash-lite` on the Gemini API free tier
  (`llm/gemini.py`). Requests are spaced 4.5 s apart (15 a minute); a per-minute 429 is waited
  out; a daily 429 moves to the next key in `SANITISE_GEMINI_API_KEY` (comma-separated, each
  from a different Google Cloud project). Thinking level "low".
* `ollama`: a local model, optionally with thinking (`SANITISE_AGENTS_THINK`), such as
  qwen3.5:4b; the base paper's local configuration is Qwen3.5-4B with thinking on.

Why: with llama3:8b, Agents 2 and 3 got who-is-who and word meanings wrong on new records
(a doctor named between a relative and "he" took the relative's facts; "old smoking habits"
became an age and a job). Rules could only patch one phrasing at a time. The base paper's own
code runs its three agents on gpt-4o-mini or Qwen3.5-4B with thinking; Llama-3-8B appears
there only as one of the models answering questions over the sanitised records.

With a strong model, Agent 2 makes **one call per document** (`DOCUMENT_PROMPT`): the model
sees every person at once and picks each fact's subject from the document's subject ids (an
enum in the schema), so `check_facts` keeps its choice (`trust_subjects`), keeps `other`
facts worded from the record (smoking history), and keeps a symptom Agent 1 did not find. The
other checks are the same for every model.

## Agent 2: fact extraction

Agent 2 is split so that the model answers only narrow questions and never handles a name, an
ID number, a contact detail or a family relationship.

**Built in code (`agents/extractor/pipeline.py`, `identity_facts`)**, from Agent 1's spans and
Person A's subjects:
* `identifier`: each NAME span belongs to the subject it is a mention of; each ID_NUMBER span
  to the person mentioned last before it in the same sentence ("His patient ID is ..." -> the
  husband), or to the writer;
* `contact`: each PHONE and EMAIL span, by the same rule;
* `relation_to`: one fact for each `writer.<relative>` subject ("son").

**One model call per person (`agents/extractor/prompt.py`).** Each call sees:
* the document;
* `PERSON`: one subject id, its label and the words the text uses for them (`My son, he,
  His`);
* the details Agent 1 found, except names, IDs, contact details and relations.

It returns `{"facts": [{relation, value}]}` for that person only. The call sets the subject
and Person A's `default_importance` sets the importance; the model decides neither.

* **Relations.** An enum in the decoding schema without `identifier`, `contact` and
  `relation_to`, each defined in one line in the system prompt.
* **Agent 2 may only describe what Agent 1 found** (`check_facts`). An age, place, job, date,
  symptom or condition must match a detail Agent 1 detected, of the right kind (`lives_in` a
  LOCATION or ADDRESS, `has_symptom` a SYMPTOM or MEDICAL_CONDITION, ...). The fact takes
  Agent 1's wording, and goes to the person mentioned last before that detail in its sentence.
  Anything else is dropped. `other` is never kept, `symptom_duration` must be a length of time,
  `sex` a sex; medication and procedures (which Agent 1 does not detect) must be worded from
  the record. Of "breathless" and "breathless climbing stairs" for one person, only the longer
  is kept.

  So a junk fact can only reach Agent 3 if Agent 1 detected that text as a detail of that kind.
  The price: a real symptom Agent 1 missed is lost too.
* **Few-shot examples.** Four, written for this prompt: the writer of a patient message, the
  wife in the same message (one fact), a son at school, and an email addressee the record
  says nothing about (an empty list).

Cost: one call per person instead of one per document. Most records have one to three people.

### Constrained decoding

`json_call` sends the schema of `LLMFactList` (title `FactList`) to Ollama as `format`, so the
model cannot invent a relation name. The platform's repair loop (`BaseJSONClient`) handles
malformed JSON:
* it retries at temperatures 0.0, 0.2, 0.4 and 0.6, with the validation error quoted;
* after three failures it raises `SCHEMA_VALIDATION_FAILED`, stage `extract`.

### After the calls

Person A's `normalise_facts` cleans the identity facts and the model's facts together. The
pipeline then drops any fact whose subject is not in `SUBJECTS` and logs how many it dropped.

## Agent 3: reconstruction

`reconstruct` writes a new record from the facts alone, in one plain-text call. It never sees
the original text, and `rewrite()` has no raw-text parameter at all. A contract test enforces
that.

Before the model sees anything:
1. Person A's routing decides an action for every detected span.
2. The placeholders and generalisations are filled in.
3. `safe_value` rewrites every fact value that contains a removed span with that span's token
   or generalised phrase, longest span first, then every other form the leak check knows.
4. `safe_facts` drops an identifier or contact fact that is not a token, and a relationship on
   anyone who is not one of the writer's relatives (a second guard; Agent 2 builds those facts
   in code now).

The model therefore receives only lines such as
`the writer - lives in: a town in Kerala.` and `the writer - contact: [PHONE_1].`

The prompt has five rules:
1. Use every fact and only the facts.
2. Copy the tokens exactly.
3. Keep vague phrases vague. Grammar changes are allowed, but nothing more specific.
4. Write in the first person for facts about the writer, in two to five sentences.
5. Return only the record.

One few-shot example shows facts going in and a record coming out.

### Leak loop (`agents/rewriter/pipeline.py`)

```
reconstruct -> leak_check (A) + grounding check (C)
  problem found -> reconstruct again, told only the categories (never the text)
  still found   -> mask_fallback (A), then grounding.mask (C): the words become tokens or [REDACTED]
  still found   -> LEAK_CHECK_FAILED (409, stage rewrite)
```

The two checks cover different failures:
* the **leak check** looks for each detail Agent 1 found and routing removed, in any form;
* the **grounding check** (`agents/rewriter/grounding.py`) needs no list of removed details. Any
  word with a digit, any email address and any capitalised word that is not the first word of a
  sentence must appear in the facts the model was given (forms of address such as "Doctor"
  excepted). A name or number the model made up, or that slipped past every earlier stage,
  is masked on any document.

`meta.leak_retries` counts the second attempts. Cell 2 of the notebook prints the totals and
the number of `[REDACTED]` masks.

## Answer prompt (both pipelines)

`build_answer_prompt(question, contexts)` returns `(system, user)`. The user message is:

```
CONTEXT:
[<doc_id>]
<text>

QUESTION: <question>
ANSWER:
```

Standard RAG puts the original texts in it and Protected RAG the sanitised ones. Every other
part of the call is identical for both pipelines:
* the same three records, retrieved over the originals;
* the same prompt;
* the same settings: temperature 0.4, top-p 0.9, 512 tokens.

The prompt is deliberately neutral. It does not tell the model to protect anyone, so any
difference between the two answers comes from the text the model was given. The dashboard's
"What the model saw" drawer shows exactly this message.

## Keeping the evaluation clean

Every few-shot example is written for its prompt. None is taken from `data/fixtures` or
`data/gold`. `tests/prompts/test_prompts.py::test_no_gold_text_in_any_prompt` fails if a gold
phrase appears in any prompt. Other tests check that:
* the extraction examples parse with the decoding schema;
* the reconstruction input contains no removed text;
* the answer prompt stays neutral.

## Failure-mode catalogue

| # | Failure | Seen with | Mitigation |
|---|---|---|---|
| 1 | Agent 2 invents relation names or uses them inconsistently | v1 (relations only listed by name) | Relations are an enum in the decoding schema; v2 defines each one in the system prompt |
| 2 | Agent 2 misses identifiers Agent 1 found (a name or phone with no fact, so Agent 3 drops it from the record) | v1 (hints were optional) | v2 says each detected detail should appear in a fact; example 1 has a fact for every identifier |
| 3 | Agent 2 turns statements about meetings or general knowledge into facts | (prompt review) | Rule "only facts about the people in the record"; example 2 explains why a meeting date is not a fact |
| 4 | Agent 2 uses a subject id that is not in `SUBJECTS` | (spec risk) | `SUBJECTS` in every prompt; the pipeline drops and logs facts with unknown subjects |
| 5 | Agent 2 paraphrases values, so they no longer match the text | (spec risk) | "copied from the record's own wording" in the prompt; A's `normalise_facts` |
| 6 | A short removed span replaced inside a longer one leaves part of a detail ("Asha" replaced first leaves "[PERSON_2] Varma") | v1 `_safe_value` (replaced in decision order) | v2 `safe_value` replaces the longest span first; covered by the test in row 7 |
| 7 | Relatives labelled "the the writer's wife" in the reconstruction input | v1 `draft` (two chained string replaces) | v2 `subject_label`; covered by `test_reconstruction_input_has_no_removed_text_and_labels_subjects` |
| 8 | Agent 3 reintroduces a removed detail | (spec risk) | The model never sees removed text; leak check, one retry with categories only, mask fallback, `LEAK_CHECK_FAILED` |
| 9 | Agent 3 makes a vague phrase specific again ("a town in Kerala" -> a town name) | (spec risk) | Rule 3; a real town name is caught by the leak check if it was a removed span |
| 10 | Agent 3 drops or alters tokens ("[PERSON 1]", "Person 1") | (spec risk) | Rule 2 and the example; D's utility score shows the loss |
| 11 | Agent 3 writes in the third person or adds headings and lists | v1 (no voice or format rule) | Rule 4 and the example |
| 12 | Answer model guesses when the records do not say | v1 ("say that they do not") | v2 gives a fixed reply, "The records do not say.", and a "do not guess" rule |
| 13 | Answer model rewrites tokens, so the dashboard cannot mark them | (spec risk) | "State details exactly as the records give them, including tokens" |
| 14 | Long or rambling answers | (spec risk) | One to three sentences; 512-token cap |
| 15 | Gold fixtures leaking into the examples | (prompt review) | All examples are new; the gold-text test guards every prompt |
| 16 | Agent 2 labels things that are not identifiers as `identifier` ("the writer", "my husband", "a lighthouse keeper"); Agent 3 then writes them out ("you can reach me at the writer", "I am a working professional... I am also a working professional") | llama3:8b, run 1 | `identifier` defined as a name or ID number only, "no name, no identifier fact"; example 3 has an unnamed writer |
| 17 | Agent 2 gives the writer a `relation_to` fact ("parent", "my wife"), which Agent 3 turns into "I am my wife's partner" | llama3:8b, run 1 | `relation_to` only on a relative's id, as one word |
| 18 | Agent 2 infers a fact the record does not state (`works_as` "student" for a child at school) and puts an activity in `symptom_duration` ("climbing stairs") | llama3:8b, run 1 | Rule with both cases spelled out; example 3 uses `studies_at` |
| 19 | Agent 3 drops a contact token ([EMAIL_1] missing from the record) | llama3:8b, run 1 | Rule 2: every token in the facts must appear |
| 20 | Agent 3 writes a vague phrase awkwardly ("I work as a working professional") or says a detail twice | llama3:8b, run 1 | Rules 3 and 4 |
| 21 | An Indian address with a PIN code was not detected, so "14/3" and "Bengaluru 560076" reached the safe text. The leak check cannot catch this: it only looks for spans that were removed | llama3:8b, run 1 (case_04) | Agent 1 rule `in_address_pin` (`rules/recognizers.py`); rule-pass ADDRESS recall on the fixtures 2/4 -> 4/4, overall precision 0.96 -> 0.99 |
| 22 | A name Agent 1 never detected reached the safe text ("I am Helen Park", a sign-off) because Agent 2 extracted it as an identifier | llama3:8b, run 2 (case_13) | Agent 1 rule `sign_off` (name alone on the line after "Thanks,", "Regards," ...): rule-pass NAME recall on the fixtures 13/14 -> 14/14. And `safe_facts`: an identifier or contact fact that did not become a token never reaches Agent 3 |
| 23 | Agent 2 copied an ID number wrongly ("HC-229174174" for "HC-2291746"), so it no longer matched the detected span and Agent 3 wrote it out | llama3:8b, run 2 (case_03) | `safe_facts`, as in row 22 |
| 24 | A part of a removed place reached Agent 3 ("Mulund" from "Fortis Hospital, Mulund"); the leak check caught the rewrite and the mask fallback replaced it with [REDACTED] | llama3:8b, run 2 (case_08) | `safe_value` also replaces every form the leak check knows (name parts, place parts, reformatted numbers), not just the exact text |
| 25 | Rows 16-17 again, despite the prompt: `identifier` "the writer", "My husband", "a 20 year old undergraduate student"; `contact` "Let me know"; `relation_to` on a colleague, written out as "My wife, [PERSON_2]" | llama3:8b, run 2 | `safe_facts` drops identifier and contact facts without a token, and `relation_to` facts on anyone who is not `writer.<relative>`. Deterministic, so it holds whatever the model does |
| 26 | Rows 16, 17, 22, 23 and 25 share a cause: the model was trusted with identifying details (deciding what is a name, copying an ID, saying who is related to whom) | llama3:8b, runs 1-2 | Agent 2 builds identifier, contact and relation_to facts in code from Agent 1's spans; the model is never asked for them (`identity_facts`) |
| 27 | Agent 2 gives one person's facts to another (case_07: the son's age and height to the writer), with all people in one call | llama3:8b, run 2 | One call per person, which only lists facts about that person; example 2 shows a call that returns one fact, example 4 an empty list |
| 28 | A detail Agent 1 never found, or one the model makes up, can only be caught by checking for that detail; a missed one is invisible to the leak check | llama3:8b, run 2 (case_13) | Grounding check on every rewrite: digits, emails and mid-sentence capitalised words must be in the facts. Over the 20 run 2 rewrites it flagged no capitalised word |
| 29 | One call per person: the model lists the whole record's facts for every person (case_07: the son's age, height and pain also given to the writer; case_02: the daughter's cough given to the writer) | llama3:8b, run 3 | `attribute` gives each fact to the person the sentence is about and drops repeats. Replayed on run 3's facts: case_07's five copies on the writer removed, case_02's and case_17's facts moved to the relative |
| 30 | A part of a removed place replaced with the whole place's phrase ("Mulund" from "Fortis Hospital, Mulund" became "I live in a hospital") | llama3:8b, run 3 (case_08) | `safe_value` generalises the part on its own ("a town in the region") |
| 31 | An ID number written as a name ("I am [ID_1]"); a record with no facts produced "Since there are no facts provided, ..." | llama3:8b, run 3 (case_03, case_14) | The reconstruction input labels identifiers "name" or "ID number"; with no facts Agent 3 writes a fixed sentence and does not call the model |
| 32 | Junk facts: "sunlight" as a symptom (from a general question), "examplemail.com" as a home, "same age", "the same problem", "moved from", "phone is ..." as a symptom, "revised filing" as a job | llama3:8b, run 4 | `check_facts`: every age, place, job, date, symptom and condition must match a detail Agent 1 found, of the right kind. Replayed on run 4's facts: all of these dropped; case_18 ends with no facts, as in the gold |
| 33 | A city inside a detected address reached the rewrite ("I live in Houston, [ADDRESS_1]") | llama3:8b, runs 3-4 (case_12) | `check_facts` gives a fact the whole detected address, which becomes one token |
| 34 | Agent 3 wrote relationships twice or wrongly ("My sister ... is my sister", "I am my husband's wife"), made one up ("My wife" for a colleague), took another person's name as the writer's ("I am [PERSON_1]"), and began with "Here is the rewritten record:" | llama3:8b, run 4 | No `relation_to` lines (the label says "the writer's sister"); other people labelled "(not the writer)"; the grounding check treats a family word not in the facts like a made-up name; the opening line is removed |
| 35 | Agent 3 wrote what its input said too literally: "[PERSON_2] is also known as [PERSON_2]" (two name lines for "Laura Fenn" and "Laura", one token), "You can contact [PHONE_1] for person 3" (a subject id as the label), "I tore my a joint injury" (a vague phrase put inside a longer one) | gemini-3.5/3.1-flash-lite, run 5 | The reconstruction input never repeats a line; other people are labelled by their name token, or "another person" without one, never by id; a generalised detail that is most of a fact's value replaces the whole value |

Add a row for each failure seen on the real model, with the model name and run.

## Real-model runs

Fill in from notebook cell 2 (report pasted in the chat) and D's scorers. The four scores need D's gold facts and routing; until then cell 2 reports only the retry and leak counts. The targets come
from `docs/phase3_plan.md`, section 8.

| Run | Model | Agent 2 coverage (high) | Agent 2 faithfulness | Agent 3 leak rate | Agent 3 utility | Leak retries | Residual leaks | Avg s / doc |
|---|---|---|---|---|---|---|---|---|
| target | | ≥ 0.80 | ≥ 0.90 | ≤ 0.05 | ≥ 0.80 | | | |
| 1 (27 Oct, prompts v2) | llama3:8b | - | - | - | - | 0 | 0 | 21.3 |
| 2 (28 Oct, prompts v3) | llama3:8b | - | - | - | - | 1 | 0 | 18.5 |
| 3 (28 Oct, identity facts in code, one call per person, grounding check) | llama3:8b | - | - | - | - | 0 | 0 | 15.7 |
| 4 (28 Oct, facts attributed by position) | llama3:8b | - | - | - | - | 0 | 0 | 12.0 |
| 5 (Agents 2-3 on Gemini, one call per document) | gemini-3.5-flash-lite, 3.1 as fallback | - | - | - | - | 0 | 0 | 34.8 (4.5 s request spacing, busy-server waits) |

## Tuning without overfitting

Cell 2 runs the 20 fixtures, and D's scorers grade the same 20. A prompt change made to fix an
output seen there is tuned to the test set, and its scores will be better than on new records.
From run 3 on:
* prefer a change in code that holds for any record (rows 22-28) to new prompt wording;
* check a prompt change on records outside the fixtures (D's corpus, or new records written
  by a teammate who has not seen the outputs) before keeping it;
* report the fixture scores once, at the end, and say in the report which rows came from the
  fixtures.

## Changing a prompt

1. Edit `SYSTEM_PROMPT` or `build_user_prompt`, keeping the function signatures.
2. Run `pytest tests/prompts` and the fast suite.
3. Run notebook cells 1-2 on Colab and add the numbers to the table above.
4. Add a catalogue row for each failure the change fixes.
5. Keep the examples free of fixture and gold text.
