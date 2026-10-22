# Phase 2 Demo Script - Agent 1: Sensitive Information Detector

**Length:** about 4 minutes. **Rehearse 5 times on Day 7, then stop changing code.**

## Before you start (do this 30 minutes before the audience arrives)

The demo runs in `notebooks/colab_live_test.ipynb` on a Colab T4 GPU (an 8B model is too
slow on a laptop CPU). Open it, set the runtime to T4 GPU, and **Run all**. This installs
everything, warms up the model, scores all 20 documents (section 8) and runs the model
comparison (section 9), so every number is on screen before you start talking.

Check before presenting:
* section 5 printed `"status": "ok"` and a warm-up time;
* section 8 printed the headline and wrote `docs/metrics_report.md`;
* section 12 downloaded `agent1_results.zip` (your fallback if the session dies).

Running locally instead (with a GPU):

```bash
python -m sanitise.evaluation.validate_gold --gold data/gold --fixtures data/fixtures --strict   # 0 errors
sanitise detect --input data/fixtures --out out/      # pre-generate all 20 predictions
```

## The script

### 1. Set the problem (30 s)
> "Agent 3 can only remove what Agent 1 finds, and the session monitor can only track attribute types Agent 1 names. So Agent 1 is the foundation. Today we show it working and, unlike the base paper, we show a measured precision and recall."

### 2. Run it on one document (60 s)
Notebook section 6, interactive box: pick `case_07` and press **Detect** (or section 7 for
the terminal view, `sanitise detect --input data/fixtures/case_07.json --verbose`).
Point at three things on screen:
1. **Rule pass**: exact identifiers with character offsets (`32 years old`, `6 ft 3 inches`, `My son`).
2. **LLM pass** (dashed borders): the medical findings (`enlarged lymph nodes`, `leukocytosis`), grounded to exact positions, and the combinations rules cannot see: *"left-sided abdominal pain + bilateral pelvic pain"*. Say: "No regex finds this. This is the contribution."
3. **Merged output**: validation OK, retries count, total time.

### 3. Show a case where the LLM matters most (45 s)
Pick `case_06` in the box (tick **rules only** first to show what rules alone miss, then untick).

> "A lighthouse keeper on a single island with a skin condition. Every word is individually harmless. Together, that is one person."

### 4. Show it does not over-flag (30 s)
Pick `case_10` (no sensitive content). Then, if asked, paste a document of your own.
> "A plain headache question. Nothing flagged. The gold set includes three documents like this so we can measure false positives honestly."

### 5. The measured result (60 s)
Scroll to notebook sections 8 and 9 (already run), or locally:
```bash
sanitise score --gold data/gold/ --pred out/ --errors
```
Expected output shape:
```
Precision 0.xx  Recall 0.xx  F1 0.xx  (20 documents, 141 gold spans)
```
Show the model comparison table: the *rules only* row against *rules + LLM* is the
measured value of the LLM pass.

Say:
> "Recall is the figure we optimise, because a missed span is a leak and a false positive only costs a little utility. The scoring is against 20 documents we annotated by hand, each reviewed by a second person, with the rules written down in an annotation guide."

### 6. Close (15 s)
> "Phase 3 builds Agents 2 and 3 and the session monitor on top of this validated detector."

## Likely questions and honest answers

| Question | Answer |
|---|---|
| Why only 20 documents? | Enough for a credible figure in one week; fewer than ten would not support a claim. Per-category numbers are indicative only; we say so in the report. |
| Is the gold set independent of the detector? | Yes. Annotated from the text and the guide before any detector output was seen, then reviewed span by span by a second team member (3 corrections, logged as D13-D15). The prompt's few-shot examples are not taken from the gold set. |
| How are quasi-identifiers scored? | By category only, because an inferred combination has no single position. So quasi recall is an upper bound. We state this limitation. |
| Why count age as sensitive? | Stricter than HIPAA Safe Harbor on purpose; age is a core quasi-identifier and later stages need it tracked. Documented in the guide. |
| Why Llama 3 8B? | It is what the base paper benchmarked. We compared it with Qwen 2.5 7B on all 20 fixtures: see `docs/model_comparison.md` *(read out the recall of each)*. |
| How do you stop the LLM inventing things? | Its output is grammar-constrained to a JSON schema with only four allowed categories, every quoted detail must be found verbatim in the document (otherwise it is dropped), and merge removes anything that only repeats a rule hit. |
| What does it miss? | Read the top 3 misses from `--errors` and have an answer ready for each. |
| Are the documents real patients? | The fixtures are synthetic, written in ChatDoctor / Enron style, so no real personal data is in the repository. *(Edit this answer if you swap in real records.)* |

## If something breaks live
- **Ollama is slow:** use the pre-generated `out/` files and say "this one was run earlier"; run only the score step live.
- **JSON validation fails:** show the retry count: it demonstrates the repair loop working. Then move on.
- **Score looks low:** do not defend it. Run `--errors` and show what was missed. A honest error analysis is more convincing than a perfect number.
