# Running the agents on Kaggle, step by step

This guide is for the team. It covers how to run the legal pipeline on Kaggle, how to check
each agent, and where to put the API keys so that a run doesn't stop when one key runs out.

## 0. Before you open Kaggle (once)

1. **Push the code.** The notebook clones branch `a/legal-pipeline` from GitHub, so the
   branch must be pushed. If the repository is private, upload the repository as a Kaggle
   Dataset instead and change the `git clone` line in the first code cell to copy from
   `/kaggle/input/...`.
2. **Gemini keys.** Each teammate creates one key at Google AI Studio
   (https://aistudio.google.com, "Get API key"). Up to nine keys can be used. See section 6.
3. **Hugging Face token** (only for fine-tuning Agent 1). Sign in at huggingface.co, open
   `google/gemma-4-E4B-it` and accept the licence, then create a read token under Settings ->
   Access Tokens.
4. **Kaggle account.** Phone-verify the account. Kaggle does not give GPUs or internet access
   without verification.

## 1. Create the notebook

1. On kaggle.com choose Create -> New Notebook, then File -> Import Notebook and upload
   `notebooks/legal_kaggle.ipynb` from this repository.
2. In the right-hand panel set **Accelerator: GPU T4 x2**, **Internet: on** and
   **Persistence: Files only** (so outputs survive a restart).
3. Under **Add-ons -> Secrets**, add each secret, then tick it for this notebook:

   | Secret name | Value |
   |---|---|
   | `GEMINI_API_KEY` | the first Gemini key |
   | `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9` | more keys (optional; teammates' own keys) |
   | `HF_TOKEN` | the Hugging Face read token |

   Never paste a key into a cell or commit it. Cells read the keys from secrets, and saved
   versions of the notebook never contain them.

## 2. Run the cells in order

| Cell | What it does | Done when |
|---|---|---|
| Packages | clones the repository, installs packages and spaCy `en_core_web_lg` | no red errors; `pip install -e .` finishes |
| Keys and settings | loads the Gemini keys and `HF_TOKEN`, prints the legal profile from `config/domains.json` | prints "N Gemini key(s) loaded" with N ≥ 1 |
| Ollama | installs Ollama, starts it on GPU 0, pulls `gemma4:e4b` | `ollama pull` reports success |
| Generate data | writes `data/legal/` and the ontology tables, validates the gold set | "300 files, ... 0 errors" |
| **Check one document** | runs one intake note through Agents 1, 2 and 3 and asks Gemini one question | see section 3 |
| Fine-tune: before, train, after | the Agent 1 before score, QLoRA training, the after score | recall after > recall before |
| Build | Agents 1-3 over all 300 documents into `/kaggle/working/store.sqlite` | "300 sanitised, 0 failed" |
| Evaluate, report | both pipelines, attacks, metrics, then `REPORT.md` and charts | the report table is printed |
| Export | `bundle.zip` for the dashboard | the file is listed with its size |

Click **Save Version** at the end ("Save & Run All" is not needed; "Quick Save" keeps the
outputs). Everything is written to `/kaggle/working`.

Kaggle gives about 30 GPU hours a week and up to 12 hours per session. Fine-tuning (1-2 h)
and building (1-2 h) can run in separate sessions. Re-running the build cell skips documents
that are already in the store, and Gemini answers come from the cache, so a restart loses at
most one document.

## 3. Checking each agent

The **Check one document** cell prints, for `c001_intake_note`:

- **Agent 1** (rules, then the LLM pass, then merging and de-duplication): every span with
  its category and whether a rule or the LLM found it. Names, phone, email, PAN, case number,
  court, amount, dates, address, occupation and city should all be there, each once.
- **Agent 2** (fact extraction): `client` facts such as `has_legal_issue`, `filed_in`,
  `claim_amount` and `identifier`. In an intake note there is no advice yet; try
  `c001_advice_email.json` to see `advised` and `documents_requested`.
- **Agent 3** (routing, rewrite, leak check): the safe text with `[CLIENT_1]`, `[PHONE_1]`,
  `[CASE_1]`, "a magistrate's court", an amount band, and "middle-aged" / "a professional"
  (the age, job and city climb two steps because together they identify the client).
  `residual leaks: none` means the leak check passed.
- **Answer:** one Gemini reply built from the safe text only.

To check a different document, change the file name in the cell. After the full build, every
document's record is in `/kaggle/working/processed/<doc_id>.json`.

Agent 1's LLM pass alone, before and after fine-tuning, is scored by the Part 3 cells, and
their numbers are in `agent1_before.json` and `agent1_after.json`.

## 4. Switching domains and models

The pipeline (Agent 1 detection and merging, Agent 2 facts, Agent 3 routing, rewrite and leak
check, search, answer, monitor) is the same for every domain. A domain changes the prompts,
the data and the models, and they are all set in **`config/domains.json`**:

```json
"legal": {
  "prompts": "legal",
  "model": "gemma4:e4b",              // Agents 2-3 through Ollama
  "answer_backend": "gemini",         // answers from cleaned text only
  "gemini_model": "gemini-3.5-flash", // or gemini-3.8-flash
  "agent1_backend": "same",           // "transformers" = Gemma 4 + the LoRA adapter
  "agent1_adapter": null              // e.g. /kaggle/working/agent1_lora
}
```

- **On Kaggle**, the build cell switches the legal profile's Agent 1 to the fine-tuned
  adapter by editing this file.
- **In the dashboard**, the domain dropdown in the header switches every request to that
  domain's prompts, data and models (the API reads `?domain=legal`). Hovering over the
  dropdown shows which models the chosen domain uses.
- **From code or curl**, add `?domain=legal` to any request, or send the header
  `X-Sanitise-Domain: legal`. `GET /v1/domains` lists the domains and their models.
- **Adding the third domain:** add an entry to `config/domains.json` with its models and data
  paths. Until its own prompts are written it reuses the `prompts` family you name, so it runs
  straight away.

## 5. Running it on a laptop instead

```powershell
$env:SANITISE_LLM_BACKEND = "mock"; sanitise serve       # every domain, no model
cd web; npm install; npm run dev                         # http://localhost:3000
python -m sanitise.legal all --mock                      # the whole legal evaluation, no model
```

To play back a Kaggle run, unzip `bundle.zip`, set `$env:SANITISE_REPLAY_DIR = "bundle"` and
run `sanitise serve`.

## 6. API keys: where they go and how they rotate

**Where the keys go**

| Where you run | Put keys in |
|---|---|
| Kaggle | Add-ons -> Secrets: `GEMINI_API_KEY`, `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9` |
| Laptop | `.env` (gitignored): `GEMINI_API_KEYS=key1,key2,key3`, or `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, ... |

**How rotation works** (`src/sanitise/llm/gemini.py`, `KeyPool`)

- Calls go to the keys in turn.
- When a key gets a rate-limit error (HTTP 429, RESOURCE_EXHAUSTED), it rests for
  `SANITISE_GEMINI_KEY_COOLDOWN_S` seconds (default 60) and the same call is retried
  immediately on the next key.
- When the error says the **daily** quota is used up, that key rests for 24 hours.
- `SANITISE_GEMINI_DAILY_TOKEN_BUDGET` (default 0 = off) sets a token cap per key per day.
  A key that reaches it is skipped until the next day (UTC). Use it to keep one key in
  reserve, or to share a key fairly.
- Only when every key is resting does a call fail, with a message saying when the first key
  is free again. Re-run the cell then; finished answers are not repeated.

**Not running out of tokens**

- **Cache:** every Gemini answer is cached in `/kaggle/working/gemini_cache`. Re-running the
  evaluation, or resuming after a failure, costs no tokens for answers already received.
  Keep that folder between sessions (Persistence: Files only).
- **Usage:** `gemini_cache/usage.json` records tokens per key per day. Keys appear as
  8-character fingerprints, never the key itself.
- **Rate:** `SANITISE_GEMINI_RPM` (default 10) spaces calls across all keys. Lower it if every
  key keeps hitting per-minute limits.
- **Size of a run:** one full evaluation is roughly 600 Gemini calls: 80 for QA, 84 for
  attacks, 40 for extraction and 400 for membership inference. At about 2,000 tokens each,
  that is roughly a million tokens (an estimate). `--mi-limit 50` halves the largest part,
  as the spec's fallback allows.
- Only the answer step and the optional rewording use Gemini. Agents 1-3 run on Gemma 4
  inside Kaggle and use no API tokens.

**Replacing or revoking a key**

- To replace one, edit the secret's value in Add-ons -> Secrets and re-run the keys cell.
- If a key leaks (for example, it was pasted into a cell or a commit), delete it in AI Studio
  straight away and create a new one.
- Use the team's own keys, one per person. Creating extra Google projects just to multiply
  the free quota may break Google's terms. If four keys are not enough, use the paid tier for
  the final run rather than more free projects.

## 7. If something goes wrong

| Symptom | Fix |
|---|---|
| `no Gemini API key` | the secret is not ticked for this notebook; re-run the keys cell |
| `all N Gemini keys are rate-limited` | wait the time it states, then re-run (the cache keeps finished answers); lower `SANITISE_GEMINI_RPM` |
| `ollama pull` 404 for `gemma4:e4b` | check the tag on ollama.com and change `model` in `config/domains.json` |
| 401/403 downloading `google/gemma-4-E4B-it` | accept the licence on Hugging Face; check the `HF_TOKEN` secret |
| training loss is `nan` | add `--fp32` to the train command |
| CUDA out of memory in the build | Agent 1 must be on `cuda:1` and Ollama on GPU 0 (the cells set both) |
| session stopped mid-build | re-run the setup cells, then the build cell; it resumes |
