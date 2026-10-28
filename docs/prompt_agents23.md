# Agent 2, Agent 3 and answer prompts - Person C

Code: `src/sanitise/agents/extractor/prompt.py` (Agent 2), `src/sanitise/agents/rewriter/prompt.py`
(Agent 3), `src/sanitise/rag/answer_prompt.py` (both RAG pipelines). Tests: `tests/prompts/`
(MockLLM, no model). Real-model runs: `notebooks/backend.ipynb` cell 2.

## Agent 2: fact extraction

One constrained-JSON call per document. The model sees:

* the document;
* `SUBJECTS`: the people in it, from Person A's `canonical.subjects`, as ids (`writer`,
  `writer.wife`, `person_1`) with a label. Every fact must name one of these ids;
* `SENSITIVE DETAILS ALREADY DETECTED`: Agent 1's spans as `- CATEGORY: text` lines, with the
  instruction that each one should appear in a fact.

It returns `{"facts": [{subject, relation, value, importance}]}`.

* **Relations.** The system prompt defines every name in `schemas.RELATIONS` in one line
  (`lives_in`, `has_condition`, `has_symptom`, ...), so the model knows when to use which.
* **Values** copy the record's wording, so later stages can match a fact against the spans.
* **Importance** follows a fixed rule per relation (medical facts high; age, sex, place of
  treatment and dates medium; the rest low).
* **Identifying details are facts too.** Agent 3 needs to know a name or phone number exists
  so that it can write a complete record. It replaces them before its model sees them.
* **Few-shot examples.** Two, written for this prompt:
  * a patient message with a relative, a clinic, a condition, a medication and a phone number;
  * a work email, which shows that a meeting date is not a fact about a person.

### Constrained decoding

`json_call` sends the schema of `LLMFactList` (title `FactList`) to Ollama as `format`.
`relation` and `importance` are enums there, so the model cannot invent a relation name or an
importance level. The platform's repair loop (`BaseJSONClient`) handles malformed JSON:
* it retries at temperatures 0.0, 0.2, 0.4 and 0.6, with the validation error quoted;
* after three failures it raises `SCHEMA_VALIDATION_FAILED`, stage `extract`.

### After the call (`agents/extractor/pipeline.py`)

Person A's `normalise_facts` cleans the values. The pipeline then drops any fact whose subject
is not in `SUBJECTS` and logs how many it dropped.

## Agent 3: reconstruction

`reconstruct` writes a new record from the facts alone, in one plain-text call. It never sees
the original text, and `rewrite()` has no raw-text parameter at all. A contract test enforces
that.

Before the model sees anything:
1. Person A's routing decides an action for every detected span.
2. The placeholders and generalisations are filled in.
3. `safe_value` rewrites every fact value that contains a removed span with that span's token
   or generalised phrase, longest span first.

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
reconstruct -> leak_check (A)
  leak found  -> reconstruct again, told only the leaked categories (never the leaked text)
  still leaks -> mask_fallback (A): the leaked words are replaced by their tokens
  still leaks -> LEAK_CHECK_FAILED (409, stage rewrite)
```

`meta.leak_retries` counts the second attempts, and `residual_leaks` lists anything that only
the mask caught. Cell 2 of the notebook prints both totals.

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

Add a row for each failure seen on the real model, with the model name and run.

## Real-model runs

Fill in from notebook cell 2 (report pasted in the chat) and D's scorers. The four scores need D's gold facts and routing; until then cell 2 reports only the retry and leak counts. The targets come
from `docs/phase3_plan.md`, section 8.

| Run | Model | Agent 2 coverage (high) | Agent 2 faithfulness | Agent 3 leak rate | Agent 3 utility | Leak retries | Residual leaks | Avg s / doc |
|---|---|---|---|---|---|---|---|---|
| target | | ≥ 0.80 | ≥ 0.90 | ≤ 0.05 | ≥ 0.80 | | | |
| 1 (27 Oct, prompts v2) | llama3:8b | - | - | - | - | 0 | 0 | 21.3 |
| 2 (28 Oct, prompts v3) | llama3:8b | - | - | - | - | 1 | 0 | 18.5 |

## Changing a prompt

1. Edit `SYSTEM_PROMPT` or `build_user_prompt`, keeping the function signatures.
2. Run `pytest tests/prompts` and the fast suite.
3. Run notebook cells 1-2 on Colab and add the numbers to the table above.
4. Add a catalogue row for each failure the change fixes.
5. Keep the examples free of fixture and gold text.
