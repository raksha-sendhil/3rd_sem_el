# Data and Evaluation (Agent 1) - Person D

Everything Person D owns. Paths are relative to the repository root.

```
data/fixtures/case_01..20.json            input documents  {doc_id, source, text}
data/gold/case_01..20.json                gold annotations (spec section 8.3 format)
docs/annotation_guide.md                  the specification for what the detector must catch
docs/demo_script.md                       the 4-minute demo, with Q&A and fallbacks
src/sanitise/evaluation/scoring.py        score(), matching, report generator
src/sanitise/evaluation/validate_gold.py  catches offset typos, bad doc_ids, overlaps
tests/evaluation/test_score.py            24 tests
tools/build_corpus.py                     regenerates fixtures + gold from marked-up text
```

`SpanMetrics` and every other model come from the shared `src/sanitise/schemas.py`.

## Run it

```bash
pytest tests/evaluation -q                                                # 24 pass
python -m sanitise.evaluation.validate_gold --gold data/gold --fixtures data/fixtures
sanitise detect --input data/fixtures/ --out out/                         # detector output
sanitise score --gold data/gold/ --pred out/ --errors                     # table is printed by default
sanitise score --gold data/gold/ --pred out/ --report docs/metrics_report.md --model llama3:8b
```

The last command writes the metrics report (headline, per-category table, error list, limitations) for you.
`python -m sanitise.evaluation.scoring --gold ... --pred ... --table --errors` still works as a standalone scorer.

## What is finished and what is not

| Item | State |
|---|---|
| Scorer, tests, validator, report generator | Done and tested |
| 20 fixtures + gold annotations | Done. Reviewed by C (`reviewed_by: "C"`), 3 fixes from the review (D13-D15); 17 `SYMPTOM` spans added 22 Oct (D21); `--strict` validation passes: 141 spans, 0 errors |
| Annotation guide | v1.1, signed off; decision log D1-D17 |
| Demo script | Complete; the `<fill in>` items take the numbers from the Colab run |
| Metrics report | `docs/metrics_report.md`, generated on Colab by `notebooks/colab_live_test.ipynb` |

## Notes for the write-up

1. **The fixtures are synthetic** (written in ChatDoctor and Enron style, fictional people). Accepted for Phase 2 (D17). To replace one with a real record, hand-annotate it by following the guide. Do not copy its predicted spans.
2. **Gold-set composition:** 16 ChatDoctor-style and 4 Enron-style; 2 documents have no sensitive content (cases 14, 18; case_10 has had one `SYMPTOM` span since D21); 11 documents carry 12 quasi-combos in total; 141 gold spans, 17 of them `SYMPTOM`. All documents are short (longest is 384 characters), whereas real records run longer and are messier, so expect the detector to score worse on real data than on this set.
3. **The spec's demo shows 143 gold spans; this set has 141.** That figure in the spec was illustrative. Report whatever the harness prints.
4. **Re-generating:** edit the marked-up text in `tools/build_corpus.py` and run `python tools/build_corpus.py`; never hand-edit `data/gold/`.

## What you need from teammates

- **Person C:** the CLI must have `sanitise detect --input <file> --out <dir>` writing one JSON per document (a `DetectorResult` or a `DetectorOutput`, both load) and `sanitise score` should use this scorer. `doc_id` must be SHA-256 first 16 hex of the raw `text`. *(Done at integration: `sanitise score` calls `score()` and has `--errors`, `--report` and `--model`.)*
- **Person A:** confirm offsets are Python indices, end-exclusive, on the exact `text` string in the fixture (no stripping or newline normalisation before detection).
- **Person B:** `QUASI_COMBO` predictions must have `start`/`end` = `None`, and any `category` must be a valid `PIICategory`.

## Known limitations of the scoring (say them in the report)

- Quasi-identifiers are scored on category only, so quasi recall is an upper bound.
- One-to-one matching is stricter than the spec's literal wording; it prevents inflation.
- 20 documents means per-category figures are indicative only.
