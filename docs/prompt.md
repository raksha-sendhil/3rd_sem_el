# LLM pass (stage 4-5) - Person B

Code: `src/sanitise/agents/detector/quasi.py`. Tests: `tests/quasi/` (MockLLM) and
`tests/integration/test_ollama_live.py::test_infer_quasi_live` (real model, `pytest -m ollama`).

## What it does

One call per document. The model sees the raw text plus the explicit identifiers the
rule pass already found, and returns:

| Kind | Categories | Position |
|---|---|---|
| Single contextual details that rules cannot enumerate | `MEDICAL_CONDITION`, `OCCUPATION`, `LOCATION` (named institutions, small places) | The model quotes the words; `ground()` finds every occurrence and gives each exact offsets |
| Names and ages the rule pass missed (safety net) | `NAME`, `AGE` | Same grounding; `merge_spans` drops any that repeat a rule hit |
| Quasi-identifier combinations | `QUASI_COMBO` | None (spec 8.2): a combination has no single position |

Rules own every other category, and are the first line for names and ages too. That split follows the annotation guide: a rule cannot
tell a diagnosis from a plain symptom (3.12), or a rare job from an ordinary one, but a
language model can (decision log D11).

## Prompt

* **System prompt** (`SYSTEM_PROMPT`): the role, the two kinds of item, definitions copied
  from the annotation guide (3.7 occupation, 3.10 institutions, 3.12 conditions, 3.13 the
  two-criteria test for combinations), the rule "copy single details exactly", and three
  few-shot examples.
* **Few-shot examples** are written for the prompt and are **not** taken from the 20 gold
  fixtures, so the evaluation is not contaminated. They cover: a rare job + small island +
  rare condition with a combination; an email with nothing to report (teaches the empty
  list); a child + named school + named hospital + condition. Example 1 also shows a plain
  symptom ("dry cough") being left out.
* **User prompt** (`build_user_prompt`): the document, then `ALREADY FOUND:` as
  `- CATEGORY: text` lines (or `(none)`), then a one-line task reminder
  (`TASK_REMINDER`) naming the three detail categories and combinations, then `OUTPUT:`.

## Constrained decoding

`json_call` sends the JSON schema of `LLMQuasiList` to Ollama as `format`, so generation
is grammar-constrained. `LLMQuasiList` has the same shape as the frozen `QuasiList` and the
same schema title (MockLLM scripts keyed on `"QuasiList"` keep working), but its `category`
enum is narrowed to the six categories above. The model cannot emit `DATE` for "pain".

## Repair loop (stage 5)

Owned by the platform (`sanitise.llm.base.BaseJSONClient`) so every agent gets it: parse ->
validate against the schema -> on failure retry with temperature 0.0, 0.2, 0.4, 0.6 and a
repair message quoting the validation error and the schema -> after 3 retries raise
`SCHEMA_VALIDATION_FAILED` (stage `llm_pass`). B supplies only prompt-specific behaviour.
Calls use `seed=42` and `num_predict=1024` (settings `SANITISE_LLM_SEED`,
`SANITISE_LLM_NUM_PREDICT`) so runs are reproducible and cannot run away.

## Post-processing (`to_spans`)

* Single details are grounded case-insensitively, on whole words, tolerant of different
  whitespace; a leading article or possessive is dropped first ("a lighthouse keeper" ->
  "lighthouse keeper", guide 3.7). Every occurrence gets a span (guide rule 5).
* A single detail that is not in the text is a hallucination and is dropped (D12).
* Combinations are deduplicated and capped at 3 per document (the gold set has at most 2).
* Afterwards `merge_spans` (Person A) drops anything that only restates a rule span and
  resolves overlaps by category precedence.

## Failure-mode catalogue

| # | Failure | Seen with | Mitigation |
|---|---|---|---|
| 1 | Wrong category for a single detail: pain -> `DATE`, job -> `RELATION`, age/height -> `DATE` | llama3:8b, Colab Day 6 | Narrowed category enum in the decoding schema; definitions in the prompt |
| 2 | Returns single details instead of combinations | llama3:8b | The prompt asks for both kinds explicitly; single details are now useful (grounded, scored) instead of wasted |
| 3 | Repeats an explicit identifier ("32 years old", "34 year old" as OCCUPATION) | llama3:8b, qwen2.5:1.5b | `ALREADY FOUND` list in the prompt; `merge_spans` restatement filter drops what slips through |
| 4 | Puts a description instead of a quote in a single detail ("small town (Coonoor)") | qwen2.5:1.5b | `ground()` cannot find it, so it is dropped and logged |
| 5 | Output varies between runs | llama3:8b | temperature 0.0 and fixed seed 42 |
| 6 | Malformed JSON / wrong enum value | (spec risk) | Constrained decoding; repair loop. Colab runs needed 0 retries |
| 7 | Too many overlapping combinations | (spec risk) | "At most two" in the prompt; hard cap of 3 |
| 8 | Plain symptom labelled `MEDICAL_CONDITION` | (expected) | Definition + counter-example in the prompt; shows up as `EXTRA` in `sanitise score --errors` |
| 9 | Very slow first call | all | Model loading; the notebook warms up once, `OLLAMA_KEEP_ALIVE=1h` keeps it in GPU memory |
| 10 | Echoes the ALREADY FOUND list and skips the conditions | qwen2.5:1.5b | `TASK_REMINDER` at the end of the user prompt: case_07 conditions went from 0/2 to 2/2 on the same model. Small models still label plain symptoms as conditions (#8) |
| 11 | Recognises a name the rules missed but labels it `OCCUPATION` ("She is Sunaina"), or an age written in words only inside a combination ("He is eleven") | llama3:8b | The four-category enum left no correct label. `NAME` and `AGE` added to the enum and prompt, with example 4 (a name and an age in words the rules missed); the rule pass also gained name phrasings and ages in words |
| 12 | Labels a hospital department as `OCCUPATION` ("pediatrics department") | llama3:8b | Example 4 states that a ward or department is part of the place, not a job |

## Choosing the model (spec section 9)

Run sections 8-9 of `notebooks/colab_live_test.ipynb`: it scores rules-only, rules +
llama3:8b and rules + qwen2.5:7b-instruct on the same 20 documents and writes
`docs/model_comparison.md`. Pick the model with the higher **recall** (a miss is a leak),
using precision and speed to break ties, and set it as `SANITISE_MODEL`.

## Changing the prompt

Edit `SYSTEM_PROMPT` / `build_user_prompt`, run `pytest tests/quasi`, then measure on Colab
(section 8). Keep the few-shot examples free of the gold fixtures.
