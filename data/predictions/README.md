# Saved detector runs

`llama3_8b/` holds Agent 1's output (rules + llama3:8b) for the 20 fixtures, one
`DetectorResult` per document, produced on a Colab T4 GPU on 18 Oct 2026 by section 8 of
`notebooks/colab_live_test.ipynb`. Scored with `sanitise score --gold data/gold --pred
data/predictions/llama3_8b`: recall 0.93, precision 0.82, F1 0.87 (124 gold spans).

`notebooks/scores.ipynb` shows this run instantly when `RUN_LIVE` is unticked.
