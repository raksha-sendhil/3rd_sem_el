# Working on Agent 1

The work is split by layer. File ownership is disjoint: edit only the files
you own, and change a seam only by whole-team agreement.

| Person | Role                 | Owns                                                                 |
|--------|----------------------|----------------------------------------------------------------------|
| A      | Deterministic Logic  | `rules/` package, the seams `agents/detector/rules.py` + `merge.py`, `tests/rules/`, `eval/`, `data/gold_draft/`, `data/gold_heldout/`, `data/real_chatdoctor_sample/`, `docs/rules.md` |
| B      | Model and Prompt     | `agents/detector/quasi.py` (prompt, few-shot examples, decoding schema, grounding), `tests/quasi/`, `docs/prompt.md` |
| C      | Platform             | `schemas.py`, `config.py`, `errors.py`, `loader.py`, `validation.py`, `results.py`, `llm/`, `agents/detector/pipeline.py`, `cli/`, `api/`, `tests/` scaffolding, `pyproject.toml` |
| D      | Data and Evaluation  | `evaluation/`, `tests/evaluation/`, `tools/build_corpus.py`, `data/fixtures/`, `data/gold/`, `docs/evaluation.md`, `docs/annotation_guide.md`, `docs/demo_script.md`, metrics report |

Paths under `src/sanitise/` are written without that prefix.

**Getting code in.** Work on your own branch inside this repository's folder
structure, run `pytest`, `pytest -m presidio` and `ruff check .`, then open a
pull request and have someone else review and merge it. Do not use GitHub's
"Add files via upload": it drops the folder structure. Never commit your own
copy of `schemas.py`, `pyproject.toml` or `README.md`.

## The seams (spec 8.2)

```python
# A
def rule_detect(text: str) -> list[SensitiveSpan]: ...
def merge_spans(
    rule_spans: list[SensitiveSpan], llm_spans: list[SensitiveSpan]
) -> list[SensitiveSpan]: ...
# B
async def infer_quasi(
    text: str, explicit: list[SensitiveSpan], llm: LLMClient
) -> list[SensitiveSpan]: ...


# C
class LLMClient(Protocol):
    async def json_call(
        self, system: str, user: str, schema: type[BaseModel], retries: int = 3
    ) -> BaseModel: ...


async def detect(doc_id: str, text: str, llm: LLMClient) -> DetectorResult: ...
# D
def score(pred: list[DetectorOutput], gold: list[DetectorOutput]) -> SpanMetrics: ...
```

Every seam now holds its real code (A: `rules/`, B: `quasi.py`, D: `evaluation/`).
Change bodies freely; keep the signatures. Extra keyword-only options with
defaults are fine (e.g. `rule_detect(text, *, layers=...)`).
`tests/contract/test_interfaces.py` checks every seam against whatever code is
in the repo, so run `uv run pytest tests/contract` before pushing.

## What `detect()` enforces at runtime

* **Rule pass** — every span has `method="rule"`, `start`/`end` set, inside the
  document, and `text[start:end]` equal to `span.text` (case and whitespace
  insensitive). Otherwise `OFFSET_OUT_OF_RANGE` / `SCHEMA_VALIDATION_FAILED`,
  stage `rule_pass`.
* **LLM pass** — every span has `method="llm"` and a non-empty `rationale`;
  offsets are optional but must fit the text if given.
* **Merge** — output re-validated and offset-checked the same way.
* `rule_detect` runs in a worker thread (`asyncio.to_thread`): keep it
  synchronous, build the Presidio `AnalyzerEngine` once and reuse it.

## Person A — working without a model

```bash
uv run sanitise detect -i data/fixtures/case_07.json --rules-only --verbose
curl -X POST localhost:8000/v1/detect/rules -H 'content-type: application/json' -d '{"text": "..."}'
```

## Person B — working against MockLLM

Design and failure-mode catalogue: [prompt.md](prompt.md).

Call the model only through `llm.json_call(system, user, LLMQuasiList)` (the
`QuasiList` shape with the category enum narrowed to the four LLM categories;
its schema title is still "QuasiList"). The
platform already sends that JSON schema to Ollama as `format`,
validates the reply, and retries up to 3 times with escalating temperature
(0.0, 0.2, 0.4, 0.6 by default) and a repair message quoting the validation
error. To change how that repair message is worded or how raw output is
parsed, write a function and pass it as a hook — do not re-implement the loop:

```python
def my_repair(error: str, raw_output: str, json_schema: dict) -> str: ...


client = OllamaClient.from_settings(settings, repair_builder=my_repair)
```

MockLLM lets you script exactly what "the model" says, including broken JSON:

```python
llm = MockLLM(
    by_schema={"QuasiList": ['{"quasi_identifiers": [{"category": "DIAGNOSIS"}]}', good_reply]}
)
spans = await infer_quasi(text, explicit, llm)
llm.calls[1].messages[-1]["content"]  # the repair instruction your prompt received
```

From the CLI: `--mock-responses examples/mock_responses.json`. Over HTTP:
`POST /v1/detect/llm` with `explicit[]` filled in by hand.

To compare models (spec section 9): `sanitise detect -i data/fixtures/ --model qwen2.5:7b-instruct --out out/qwen/`.

## Person D — scoring

`sanitise score --gold data/gold/ --pred out/` loads both directories (gold
files in the spec 8.3 format; predictions as written by `detect --out`),
checks every gold offset against the gold text, warns about unpaired
documents, and then calls `score(pred, gold)` (one-to-one matching; see the
docstring in `evaluation/scoring.py`). Add `--errors` to list misses and false
positives, `--report docs/metrics_report.md --model llama3:8b` to write the
metrics report. More in `docs/evaluation.md`.

`doc_id` is always the first 16 hex characters of the SHA-256 of the document
text, computed by `sanitise.loader.compute_doc_id`. Use the same function when
writing gold files so the ids line up.

## Configuration

All settings are in `src/sanitise/config.py`, overridable by `SANITISE_*`
environment variables or a `.env` file (see `.env.example`). Set
`SANITISE_LLM_BACKEND=mock` to run the API without Ollama.

## Tests

* `tests/unit`, `tests/integration` — platform; no model, Presidio or spaCy needed.
* `tests/contract` — the seams; A, B and D should keep these green.
* `@pytest.mark.ollama` — live tests, deselected unless you run `pytest -m ollama`.
* Shared fixtures (`SAMPLE_TEXT`, `fake_rule_detect`, `mock_llm`, `gold_dir`) are in `tests/conftest.py`.
