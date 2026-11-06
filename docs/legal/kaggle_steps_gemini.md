# Running the legal pipeline with every agent on Gemini

This guide covers branch `law-version-2` and `notebooks/legal_kaggle_gemini.ipynb`. Agents 1-3
and the answers all run on `gemini-3.5-flash`, so the run shows how Gemini behaves in place of
Gemma 4. For the Ollama and Gemma 4 run, use [kaggle_steps.md](kaggle_steps.md) on branch
`a/legal-pipeline`.

## What is different

| | `a/legal-pipeline` | `law-version-2` |
|---|---|---|
| Agent 1 LLM pass | Gemma 4 + LoRA adapter | Gemini |
| Agents 2-3 | Gemma 4 through Ollama | Gemini |
| Answers | Gemini | Gemini |
| Search | bge-m3 + BM25, local | the same |
| Fine-tuning | yes | none: Gemini cannot be fine-tuned |
| Raw documents sent to Google | no | **yes** |

**Fictional data only.** The agents read the original documents, so this branch breaks the
design's rule that the outside model only sees cleaned text. The privacy scores measure the
answer step and say nothing about what the agents sent to Google. Report them that way, and
never run this branch on real client documents.

The medical domain is unchanged and still uses Ollama.

## Steps

1. **Create the notebook.** In Kaggle, choose Create → New Notebook, then File → Import Notebook
   and upload `notebooks/legal_kaggle_gemini.ipynb`.
2. **Set the session options.** Accelerator **GPU T4**, Internet **On**, Persistence
   **Files only**.
3. **Add the keys.** Under Add-ons → Secrets, add `GEMINI_API_KEY` and optionally
   `GEMINI_API_KEY_2` to `_9`, then tick **Attach**. `HF_TOKEN` is not needed.
4. **Run the cells from top to bottom.**

| Cell | Expect | If not |
|---|---|---|
| Packages | no red error text at the end | if the clone fails, check that `law-version-2` is on GitHub |
| Keys and settings | `N Gemini key(s) loaded`, then the profile with `"llm_backend": "gemini"` | `0 key(s)`: the secret is not attached |
| Generate | `300 files, 3453 spans, 0 errors, 300 warnings` | |
| Preflight | `[OK  ]` lines and `All required checks passed.` | fix any `FAIL` first |
| Check | spans, facts, safe text and an answer for each of three documents | see below |
| Build | one line per document, then `30 sanitised, 0 failed` | re-run: finished documents are skipped |
| Behaviour | the model, fallbacks, retries, leaks and time per stage | see below |
| Evaluate | the report table, then four charts | re-run: replies already received are cached |
| Export | `bundle.zip` with its size | |

Go through the Check output with [Part D of the main guide](kaggle_steps.md#part-d-checking-each-agent).

## Requests and limits

| Run | Requests, roughly | At 10 a minute |
|---|---|---|
| 30 documents, `MI_PROBES = 10` (the default) | 240 | 30-40 minutes |
| 300 documents, `MI_PROBES = 100` | 1,500 | 3 hours or more |

These counts come from a mock run, so real runs with repair retries will be somewhat higher.
Check your key's limits in AI Studio first: a free key's daily limit may not cover the full run
in one day, so it may take several days or several keys.

- **Per-minute limit:** the key rests for 60 seconds and the next key is used. When every key is
  resting, the call waits, up to `SANITISE_GEMINI_MAX_KEY_WAIT_S` (300 seconds), and tries again.
- **Daily quota:** the key rests for a day. Once every key has run out, the remaining documents
  fail quickly with `all N Gemini keys are rate-limited`. They are not lost: re-run the build
  cell when the quota returns.

To go from the trial to the full run, set `DOCS = None` and `MI_PROBES = 100`, re-run the
settings cell, then the build cell and the cells after it.

## What to look at

Compare the behaviour cell and `REPORT.md` with the Gemma 4 run:

- **Agent 1 recall and precision:** Gemini has no fine-tuning, so these show what prompting alone
  achieves.
- **Fell back:** a high count means the agent scores mostly reflect the fallback rules.
- **Residual leaks:** should be 0 or very small.
- **Time per document:** with Gemini this is mostly the 6-second gap between calls at 10 a minute,
  so it says little about the model's own speed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `400 INVALID_ARGUMENT` on the Check cell | Gemini rejected the JSON schema or the model name. Note the full message; `response_json_schema` needs a recent `google-genai`, so re-run the packages cell |
| `404` naming the model | change `gemini_model` in `config/domains.json` |
| many `agent3_in_place` fallbacks | the rewrite is dropping the advice or failing the leak check; look at one document with the Check cell |
| `all N Gemini keys are rate-limited` on every document | the daily quota has run out; re-run when it returns or add a key |
