# Health evaluation, run 1 (5h scale, 11 Nov 2026)

The first full health run on Kaggle (`notebooks/eval_health_kaggle.ipynb`, `SCALE = "5h"`, commit
`f61d9b7`). It compares no privacy (standard RAG) with ours (Agents 1-3 + session monitor), using
the same questions and the same answer model (llama3:8b, top 3 documents). Definitions are in
`docs/evaluation_metrics.md`.

| File | What it is |
|---|---|
| `results/REPORT.md` | the report, regenerated from this run's answers with the measures added on 12 Nov (answer rate, utility where ours answered, re-identification by details, utility by kind of scan, OCR time) |
| `results/REPORT_as_run.md`, `results/results_as_run.json` | the report exactly as the run wrote it |
| `results/results.json`, `results/figures/` | every number and figure of the regenerated report |
| `answers/` | every question with both pipelines' answers |
| `data/` | the questions asked (Target Info, repeat, membership, utility, attack sessions) |
| `e2e/` | the 25 end-to-end scans: what OCR read, and the labelled private values and questions |

## Main numbers

| Metric | No privacy | Ours |
|---|---|---|
| Target Info: distinct private values leaked (100 questions) | 126 | 0 |
| Questions leaking a private value | 95% | 0% |
| ROUGE Prompt rate (close paraphrase of a record) | 30% | 0% |
| Repeat Prompt rate (10+ words copied) | 32% | 30% |
| Membership inference AUC (TPR at 1% FPR) | 0.78 (52%) | 0.62 (23%) |
| Secret Recovery Rate, round 8 (60 sessions) | 50% | 0% |
| Session Leak Success (target identified) | 81.7% | 0% |
| People still matching the answers' details, median | 1 | 9 |
| Sessions narrowed to one person | 88% | 3% |
| Sessions narrowed below k = 5 | 95% | 30% |
| Utility retention, ChatDoctor | 100% | 35% (63% where ours answered) |
| Utility retention, end to end: gold text / scan + OCR | 100% | 61% / 56% |
| Latency, single question (median) | 5.2 s | 4.9 s |
| OCR time per scanned document (median; one page each) | | 22 s |

All differences between the pipelines are significant at p < 0.001, except the Repeat Prompt rate.

## Known limitation of this run

The utility questions were worded around each record's original wording: ChatDoctor questions
quoted the patient, and scan questions named hospitals and section headings. Ours rewrites that
wording, so it often answered "The records do not say" even with the right record retrieved.

`notebooks/eval_health_rerun_utility_kaggle.ipynb` asks questions about the content instead.
Its results go in `../run2_utility/`.
