# Preventing Cumulative Privacy Leakage in RAG

A multi-agent pipeline that rewrites private documents before they are used in
retrieval-augmented generation. **Phase 2** builds and measures **Agent 1, the
Sensitive Information Detector**: a document goes in, a validated list of
sensitive spans comes out, scored against hand-annotated ground truth.

Everything runs locally: Ollama (llama3:8b, qwen2.5:7b-instruct fallback),
Presidio + spaCy, no paid APIs.

## Quick start

With pip (pinned, known-good versions):

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate    # Git Bash;  PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # includes spaCy en_core_web_lg (~400 MB)
pip install -e . --no-deps       # the `sanitise` command
```

Or with uv: `uv sync --extra spacy-model`.

Then the model runtime (not a Python package):

```bash
winget install Ollama.Ollama     # or https://ollama.com/download
ollama pull llama3:8b            # and/or qwen2.5:7b-instruct
sanitise health                  # is Ollama up and the model pulled?
cp .env.example .env             # optional; defaults work for a local Ollama
```

No Ollama yet? Every command takes `--mock` (MockLLM) instead.

```bash
# The Phase 2 demo command
uv run sanitise detect --input data/fixtures/case_07.json --verbose

# Same thing without a model, with scripted LLM replies
uv run sanitise detect -i examples/case_demo.json --verbose --mock-responses examples/mock_responses.json

# Batch: write one <doc_id>.json per document, then score against the gold set
uv run sanitise detect -i data/fixtures/ --out out/
uv run sanitise score --gold data/gold/ --pred out/

# HTTP API on http://127.0.0.1:8000 (interactive docs at /docs)
uv run sanitise serve
```

`sanitise detect` options: `--verbose`, `--out DIR`, `--rules-only`, `--model NAME`,
`--mock`, `--mock-responses FILE`, `--index N` (record in a multi-record file),
`--concurrency N`, `--skip-too-long`, `--json`. Exit codes: `0` ok, `1` pipeline
error (prints the error envelope), `2` usage error, `3` a stage not implemented yet.

## How Agent 1 works

```text
document ─► load (doc_id = sha256[:16], ≤ 2000 chars)
         ─► rule_detect   Presidio + spaCy + regex   → explicit spans with exact offsets   (Person A)
         ─► infer_quasi   one constrained-JSON call  → conditions, jobs, places (quoted, then
                          grounded to exact offsets) + QUASI_COMBO combinations (no offsets)  (Person B)
                ↳ validate against the schema; on failure retry ≤ 3× with rising temperature
                  and the validation error fed back as a repair instruction             (Person C, shared loop)
         ─► merge_spans   union, restatement filter, category precedence                   (Person A)
         ─► DetectorResult { output, meta: timings, retries, counts }                       (Person C)
         ─► score         precision / recall / F1 vs gold, overall + per category           (Person D)
```

`detect()` validates every stage boundary: spans are re-validated with Pydantic,
rule spans must have `method="rule"` and offsets that match the text, LLM spans
must have `method="llm"` and a rationale. A violation raises a typed error with
the stage that caused it, rather than surfacing as a wrong score later.

## HTTP API (spec 8.4)

| Method | Path                | Request                       | Response                              |
|--------|---------------------|-------------------------------|---------------------------------------|
| POST   | `/v1/detect`        | `{doc_id?, text, options?}`   | `DetectorResult`                      |
| POST   | `/v1/detect/rules`  | `{doc_id?, text}`             | `DetectorOutput` — rule pass only     |
| POST   | `/v1/detect/llm`    | `{doc_id?, text, explicit[]}` | `DetectorOutput` — LLM pass only      |
| POST   | `/v1/score`         | `{pred[], gold[]}`            | `SpanMetrics`                         |
| GET    | `/v1/health`        | —                             | `{status, ollama_reachable, model, pipeline_version}` |

`options` = `{llm: bool = true, model?: str, retries?: int}`. `doc_id` defaults to the
SHA-256-derived id. Errors use the spec 8.5 envelope:

| code                       | HTTP | when                                             |
|----------------------------|------|--------------------------------------------------|
| `DOC_TOO_LONG`             | 413  | text over `SANITISE_MAX_DOC_CHARS` (2000)       |
| `SCHEMA_VALIDATION_FAILED` | 502  | model output never validated after all retries  |
| `LLM_UNREACHABLE`          | 503  | Ollama down, or model not pulled                |
| `LLM_TIMEOUT`              | 504  | no answer within `SANITISE_LLM_TIMEOUT_S`       |
| `OFFSET_OUT_OF_RANGE`      | 500  | a stage produced offsets that do not fit the text |

Malformed request bodies return FastAPI's standard `422`; an unimplemented stage returns `501`.

## Repository layout

```text
src/sanitise/
  schemas.py              FROZEN shared contract (spec 8.1 + seam models)          C
  config.py               pydantic-settings; SANITISE_* env vars / .env           C
  errors.py               error codes + envelope (spec 8.5)                        C
  loader.py               stage 1: ChatDoctor / Enron / fixture loading, doc_id    C
  validation.py           offset checks shared by pipeline and gold loader         C
  results.py              write predictions, read predictions + gold files          C
  llm/
    base.py               LLMClient protocol, retry/repair loop, call accounting   C
    ollama.py             Ollama client (JSON-schema constrained decoding)         C
    mock.py               MockLLM — scriptable, same validation path as Ollama     C
    factory.py            build the configured client                              C
  agents/detector/
    pipeline.py           detect() orchestration                                    C
    rules.py              rule_detect()   — seam, re-exports sanitise.rules         A
    merge.py              merge_spans()   — seam, re-exports sanitise.rules         A
    quasi.py              infer_quasi()   — prompt, few-shot, grounding            B
  rules/                  Presidio + spaCy + regex layers, normalise, merge        A
  evaluation/
    scoring.py            score(), matching, error list, markdown report           D
    validate_gold.py      gold-file checker (offsets, doc_ids, overlaps)           D
  cli/                    `sanitise` command (Typer + Rich)                         C
  api/                    FastAPI app                                               C
tests/
  unit/  integration/     platform tests (MockLLM, fake Ollama transport)          C
  contract/               seam tests run against everyone's current code            C
  rules/                  rule pass tests (most need spaCy: -m presidio)            A
  evaluation/             scorer + gold validator tests                            D
data/
  fixtures/  gold/        the 20 fixtures and the official gold set                 D
  gold_draft/  gold_heldout/  real_chatdoctor_sample/   rule-pass dev data          A
eval/                     rule-pass dev scorer, ablation, RESULTS.md                A
tools/build_corpus.py     regenerates data/fixtures + data/gold                     D
docs/                     ownership, conventions, rules.md (A), evaluation.md,
                          annotation_guide.md, demo_script.md (D)                   all
examples/                 a demo record and a MockLLM script
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for who owns what and how to
work against the seams.

## Development

```bash
pytest                        # fast suite: no model, no spaCy load
pytest -m presidio            # Presidio + en_core_web_lg environment check (~1 min)
pytest -m ollama              # live tests against the real Ollama model (slow on CPU)
ruff check . && ruff format --check .
```

**No GPU?** An 8B model on a CPU-only laptop can take minutes per call. Open
[notebooks/colab_live_test.ipynb](notebooks/colab_live_test.ipynb) in Google
Colab (T4 GPU runtime) and run all cells: it installs Ollama and the models, lets you
paste any document and see the detected spans highlighted, scores the 20 gold
documents (`docs/metrics_report.md`), compares llama3:8b with qwen2.5:7b-instruct
(`docs/model_comparison.md`), and runs the test suites and the HTTP API.

Design notes per layer: [docs/rules.md](docs/rules.md) (A), [docs/prompt.md](docs/prompt.md) (B),
[docs/evaluation.md](docs/evaluation.md) and [docs/annotation_guide.md](docs/annotation_guide.md) (D).
