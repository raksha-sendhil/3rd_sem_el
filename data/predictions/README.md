# Saved detector runs

`llama3_8b/` holds Agent 1's output (rules + llama3:8b) for the 20 fixtures, one
`DetectorResult` per document, produced on a Colab T4 GPU on 18 Oct 2026 by section 8 of
`notebooks/colab_live_test.ipynb`. Scored with
`sanitise score --gold data/predictions/llama3_8b_gold --pred data/predictions/llama3_8b`:
recall 0.93, precision 0.82, F1 0.87 (124 gold spans).

`llama3_8b_gold/` is a copy of `data/gold` as it was on that date, before the `SYMPTOM`
category (annotation guide D21) added 17 spans. The saved run is always scored against
this copy, because it was made before `SYMPTOM` existed. The prompt used for it also
contained a few phrases from the gold set (docs/prompt.md, failure 13), so treat these
figures as slightly optimistic and re-run for current numbers.

`notebooks/scores.ipynb` shows this run instantly when `RUN_LIVE` is unticked.
