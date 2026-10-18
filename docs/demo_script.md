# Phase 2 Demo Script - Agent 1: Sensitive Information Detector

**Length:** about 4 minutes. **Rehearse 5 times on Day 7, then stop changing code.**

## Before you start (2 minutes of setup, do this before the audience arrives)

```bash
ollama list                                   # model is pulled and loaded
python -m sanitise.evaluation.validate_gold --gold data/gold --fixtures data/fixtures --strict   # must say 0 errors
rm -rf out && mkdir out
# pre-generate predictions for all 20 documents so the score step is instant:
for f in data/fixtures/*.json; do sanitise detect --input "$f" --out out/; done
```

Warm the model with one throwaway call so the first live run is not slow.
Have `data/fixtures/case_07.json` ready. Keep a terminal tab with a pre-recorded fallback output.

## The script

### 1. Set the problem (30 s)
> "Agent 3 can only remove what Agent 1 finds, and the session monitor can only track attribute types Agent 1 names. So Agent 1 is the foundation. Today we show it working and, unlike the base paper, we show a measured precision and recall."

### 2. Run it on one document (60 s)
```bash
sanitise detect --input data/fixtures/case_07.json --verbose
```
Point at three things on screen:
1. **Rule pass**: exact identifiers with character offsets (`32 years old`, `6 ft 3 inches`, `My son`).
2. **LLM pass**: the combinations rules cannot see: *"left-sided abdominal pain + bilateral pelvic pain"*. Say: "No regex finds this. This is the contribution."
3. **Merged output**: validation OK, retries count, total time.

### 3. Show a case where the LLM matters most (45 s)
```bash
sanitise detect --input data/fixtures/case_06.json --verbose
```
> "A lighthouse keeper on a single island with a skin condition. Every word is individually harmless. Together, that is one person."

### 4. Show it does not over-flag (30 s)
```bash
sanitise detect --input data/fixtures/case_10.json --verbose     # no sensitive content
```
> "A plain headache question. Nothing flagged. The gold set includes three documents like this so we can measure false positives honestly."

### 5. The measured result (60 s)
```bash
sanitise score --gold data/gold/ --pred out/
```
Expected output shape:
```
Precision 0.xx  Recall 0.xx  F1 0.xx  (20 documents, 121 gold spans)
```
The per-category table is printed under the headline; add `--errors` to list the misses.

Say:
> "Recall is the figure we optimise, because a missed span is a leak and a false positive only costs a little utility. The scoring is against 20 documents we annotated by hand, each reviewed by a second person, with the rules written down in an annotation guide."

### 6. Close (15 s)
> "Phase 3 builds Agents 2 and 3 and the session monitor on top of this validated detector."

## Likely questions and honest answers

| Question | Answer |
|---|---|
| Why only 20 documents? | Enough for a credible figure in one week; fewer than ten would not support a claim. Per-category numbers are indicative only; we say so in the report. |
| Is the gold set independent of the detector? | Yes. Annotated from the text and the guide before any detector output was seen. |
| How are quasi-identifiers scored? | By category only, because an inferred combination has no single position. So quasi recall is an upper bound. We state this limitation. |
| Why count age as sensitive? | Stricter than HIPAA Safe Harbor on purpose; age is a core quasi-identifier and later stages need it tracked. Documented in the guide. |
| Why Llama 3 8B? | It is what the base paper benchmarked. Qwen 2.5 7B was compared on five fixtures on Day 1. *(fill in the result)* |
| What does it miss? | Read the top 3 misses from `--errors` and have an answer ready for each. |
| Are the documents real patients? | The fixtures are synthetic, written in ChatDoctor / Enron style, so no real personal data is in the repository. *(Edit this answer if you swap in real records.)* |

## If something breaks live
- **Ollama is slow:** use the pre-generated `out/` files and say "this one was run earlier"; run only the score step live.
- **JSON validation fails:** show the retry count: it demonstrates the repair loop working. Then move on.
- **Score looks low:** do not defend it. Run `--errors` and show what was missed. A honest error analysis is more convincing than a perfect number.
