# Setting up and checking the system, step by step

This guide is for the whole team. Work through it in order; each step says what to type, what
you should see, and what to do if you see something else.

- **Part A**: the laptop. Install, run the tests, run the whole legal pipeline without a
  model, and open the dashboard. About 30 minutes, no GPU needed.
- **Part B**: accounts and keys (Gemini, Hugging Face, Kaggle). About 15 minutes.
- **Part C**: the Kaggle notebook with the real models, cell by cell. 3-5 hours of GPU time
  over one or two sessions.
- **Part D**: checking each agent on its own.
- **Part E**: checking the whole system: the report, the attacks and the dashboard.
- **Part F**: API keys, rotation and not running out of tokens.
- **Part G**: domains: switching, models per domain, and adding the third domain.
- **Part H**: troubleshooting.

Commands are for Windows PowerShell unless marked; Git Bash, macOS and Linux differ only in
how the virtual environment is activated.

---

## Part A: the laptop

### A1. Prerequisites

| Tool | Version | Check with |
|---|---|---|
| Python | 3.11 | `py -3.11 --version` |
| Git | any recent | `git --version` |
| Node.js | 20 or newer (only for the dashboard) | `node --version` |
| Ollama | optional; only to run real models locally | `ollama --version` |

### A2. Get the code

```powershell
git clone https://github.com/raksha-sendhil/3rd_sem_el.git
cd 3rd_sem_el
git checkout a/legal-pipeline
git pull
```

If you already have the repository, run `git fetch` and `git checkout a/legal-pipeline`,
then `git pull`.

### A3. Install

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt     # includes spaCy en_core_web_lg (about 400 MB)
pip install -e . --no-deps          # the `sanitise` command
pip install google-genai matplotlib # Gemini client and the report charts
```

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Check:** `python -c "import sanitise, spacy; spacy.load('en_core_web_lg'); print('ok')"`
prints `ok`.

### A4. Run the tests

```powershell
python -m pytest -q                       # fast suite, about 10 s
python -m pytest -q -m "not ollama"       # everything except live-model tests, about 30 s
```

**Expect:** `441 passed, 3 deselected` for the second command. The 3 deselected tests need a
live Ollama model and are expected to be skipped.

The tests cover:

- the legal regexes and their exact offsets
- party roles and placeholders
- the generalisation ladders
- the leak check
- the session monitor
- the data generator (sizes, offsets, candidate pools)
- the seven metrics
- Gemini key rotation
- per-request domain switching in the API
- one end-to-end mock run

### A5. Check the setup

```powershell
python -m sanitise.legal preflight
```

Each line is `OK`, `FAIL` (needs fixing before the real run) or `--` (optional here). On a
laptop without keys or Ollama you will see:

```
Code and data
  [OK  ] spaCy en_core_web_lg
  [OK  ] legal corpus: 300 documents in data\legal\corpus
  [OK  ] ontology tables: data/ontology/
Domains (config/domains.json)
  [OK  ] medical: agents llama3:8b, answers same, agent 1 same
  [OK  ] legal: agents gemma4:e4b, answers gemini-3.5-flash, agent 1 same
Gemini (answers)
  [FAIL] 0 key(s): add GEMINI_API_KEY (Kaggle secret or .env)
Ollama (Agents 2-3: gemma4:e4b)
  [FAIL] server: http://localhost:11434
...
```

On the laptop the two `FAIL` lines are fine: the mock run below needs neither. On Kaggle,
every required line must be `OK`.

### A6. Run the whole legal pipeline without a model

```powershell
python -m sanitise.legal all --mock
```

This runs every stage: generate the data (the same files as committed), the ontology tables,
then Agents 1-3 over 300 documents (rule pass plus a deterministic rewrite in place of the
models), the evaluation with a canned answer model, the report and the dashboard bundle.
**Expect** (about 40 seconds):

```
== build
300 sanitised, 0 already in the store, 0 failed, 0 with residual leaks (25 s, 0.1 s per document)
== evaluate
QA accuracy baseline 0.00 protected 0.00; results in data\legal\results
== report
data\legal\results\REPORT.md
== export
data\legal\bundle
```

Open `data\legal\results\REPORT.md`. In a mock run these rows should already meet their
targets:

| Row | Mock value |
|---|---|
| Extraction: exact identifier matches (protected) | 0 |
| PII leakage rate (protected) | 0.000 (baseline about 0.27) |
| Recall@4 | about 0.97 |
| Agent 1: recall on direct identifiers | about 0.99 |
| Agent 1: precision | about 0.99 |
| Agent 3: leak rate | 0.000 |
| Agent 3: routing accuracy | about 0.90 |
| Agent 3: documents with a direct identifier left | 0 |

In the attack table, every protected row should say `never`, and `stepwise_c001` with the
monitor should list blocked rounds `2, 3, 4`.

QA accuracy, utility retention, Agent 2, membership inference and latency will show `**no**`
in a mock run. That is expected: the canned answer model only quotes the first document, and
there is no Agent 2 model. These rows only mean something after the Kaggle run (Part C).

The outputs (`store.sqlite`, `results/`, `bundle/`, `processed/`) are gitignored.

### A7. Look at one document through every agent

```powershell
python -m sanitise.legal check --mock --doc c001_advice_email --question "What did we advise?"
```

This prints five blocks. Part D explains what to check in each one. In mock mode:

- **AGENT 1** lists 6 spans: the case number, `Bose`, the date, `Emerald Enterprises`, the
  court, and the lawyer `Sharmila Venkatesan`. `gold spans missed: ['cheque bounce']` is
  expected, because the legal issue is the LLM pass's job.
- **AGENT 3 safe text** reads naturally:
  `Dear Mr. [CLIENT_1]`, `in February 2025`, `your dispute with a trading business`,
  `listed before a magistrate's court`, `Adv. [PERSON_1]`. The advice paragraph is unchanged.
- **leak check** says `no residual leaks`.

Other documents to try are any file name in `data\legal\corpus`, for example
`c001_intake_note`, `c001_case_note`, `c002_client_enquiry` or `c045_intake_note`.

### A8. The dashboard

Use two terminals, both in the repository folder with the virtual environment active.

```powershell
# terminal 1: the API, no model needed
$env:SANITISE_LLM_BACKEND = "mock"; sanitise serve
```

```powershell
# terminal 2: the dashboard
cd web
copy .env.example .env.local     # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install                      # first time only
npm run dev                      # open http://localhost:3000
```

Check each of these:

1. The header has a **Domain** dropdown with Legal and Medical. Hovering over it shows the
   models that domain uses.
2. **Legal → Documents**: 48 documents from the 20-client demo slice. Open one; the original
   and the sanitised version appear side by side, with every change marked.
3. **Legal → Compare**: ask "What did we advise in past cheque-bounce matters where the cheque
   was returned for insufficient funds?" Both answers appear. The payload inspector shows
   names on the baseline side and placeholders on the protected side.
4. **Legal → Attacks**: nine attacks (direct, linkage and stepwise × `c001`, `c002`,
   `c003`). Run `stepwise_c001`: the protected side is refused from round 2.
5. **Medical**: the original fixtures and the three medical attacks come back, so switching
   domains works.

Mock answers are canned quotes. Real answers come from replaying the Kaggle run (Part E3).

---

## Part B: accounts and keys

### B1. Gemini API keys (answers only)

1. Each teammate goes to https://aistudio.google.com, signs in and chooses **Get API key**.
   If asked, create the key in a new project.
2. Copy the key (it starts with `AIza`). Never paste it into a notebook cell, a chat or a
   commit.
3. Send it to the person running Kaggle through a private channel. Better still, each person
   adds their own key to the shared notebook's secrets (B3).

One key is enough to start; two to four make rate limits a non-issue (Part F).

### B2. Hugging Face (only for fine-tuning Agent 1)

1. Create an account at https://huggingface.co.
2. Open the `google/gemma-4-E4B-it` model page and accept the licence. Approval is usually
   immediate.
3. Under Settings → Access Tokens, create a **Read** token (it starts with `hf_`).

### B3. Kaggle

1. Create an account and verify your phone number (Settings → Phone verification). Without
   it, Kaggle offers no GPU and no internet.
2. Create → New Notebook. Then File → Import Notebook → upload
   `notebooks/legal_kaggle.ipynb` from your copy of the repository.
3. Right-hand panel → **Session options**:
   - **Accelerator: GPU T4 x2.** Two GPUs: Ollama on one, the fine-tuned Agent 1 on the
     other.
   - **Internet: On.**
   - **Persistence: Files only.** `/kaggle/working` then survives restarts.
4. **Add-ons → Secrets**: add each secret, then tick **Attach** for this notebook.

   | Label | Value |
   |---|---|
   | `GEMINI_API_KEY` | key 1 |
   | `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, ... up to `_9` | more keys (optional) |
   | `HF_TOKEN` | the Hugging Face token |

   Labels must match exactly, in capitals.

---

## Part C: the Kaggle notebook, cell by cell

Run the cells from top to bottom. The times are for a T4.

### C1. Packages (cell 1, about 5 minutes)

Clones `a/legal-pipeline` into `/kaggle/working/repo`, installs the packages and the spaCy
model.

- **Expect:** no red error text at the end.
- **If the clone fails** (private repository): upload the repository as a Kaggle Dataset
  (Add Input → Upload), and replace the `git clone` line with
  `!cp -r /kaggle/input/<dataset>/3rd_sem_el /kaggle/working/repo`.

### C2. Keys and settings (cell 2)

- **Expect:** `N Gemini key(s) loaded` with N equal to the number of `GEMINI_API_KEY*`
  secrets you attached, followed by the legal profile printed from `config/domains.json`.
- **If it says `0 Gemini key(s)` or warns about `HF_TOKEN`:** the secret is not attached.
  Go to Add-ons → Secrets, tick Attach, and re-run the cell.

### C3. Ollama and Gemma 4 (cell 3, about 5 minutes)

Installs Ollama, starts it on GPU 0 and downloads `gemma4:e4b`.

- **Expect:** `ollama pull` ends with `success`.
- **If the pull fails with "not found":** check the exact Gemma 4 tag on ollama.com, then
  change `"model"` in `config/domains.json` and in this cell.

### C4. Generate the data (cell 4, about 1 minute)

- **Expect:** `300 files, 3453 spans, 0 errors, 300 warnings`. The warnings only say the gold
  files have not been human-reviewed yet (docs/legal/gold_review.md).

### C5. Preflight (cell 5)

`python -m sanitise.legal preflight`. **Every line must be `OK`**, apart from `--` for:

- **LoRA adapter:** it does not exist until Part 3 has run.
- **Agent 1 LLM pass:** it uses the base model until then.

Fix any `FAIL` before going on: the long cells will fail the same way, only later.

### C6. Check the agents on one document (cell 6, 2-5 minutes)

Runs `check` on `c001`'s intake note, advice email and case note with the real models, and
asks Gemini one question each time. Read the output against the checklists in Part D before
starting the long runs. This is the cheapest place to catch a wrong prompt or model.

### C7. Fine-tune Agent 1

**Free the GPUs first** (the cell just before the "before" eval). After the agent checks,
Ollama keeps Gemma 4 loaded, about 9 GB, and training needs one whole T4. The cell stops
Ollama and prints the memory used on each GPU; both should show only a few MiB. If one
still shows several GB, use Run → Restart & clear cell outputs, re-run cells 1-2 (not the
Ollama cell), then run the free-the-GPUs cell again. Fine-tuning does not need Ollama; the
build step starts it again.


| Cell | What it does | Time | Expect |
|---|---|---|---|
| eval (before) | the base Gemma 4 on the 40 held-out documents | about 20 min | `P ... R ... F1 ...`, written to `agent1_before.json` |
| train | QLoRA, 400 documents, 1 epoch | 1-2 h | a loss line every 10 steps, falling steadily; ends with `adapter saved to /kaggle/working/agent1_lora` |
| eval (after) | the tuned model on the same 40 | about 20 min | a before/after table, then `Done when: recall after > recall before -> True` |

Watching the loss:

- **Falls from about 1-2 towards 0.1-0.3:** good.
- **Flat or rising:** the learning rate is too high, or the labels are wrong. Stop and tell
  person C.
- **`nan`:** re-run the train cell with `--fp32` added.

If the session ends during training, re-run C1-C3 and then the train cell with
`--resume /kaggle/working/agent1_checkpoints/checkpoint-<latest>`.

### C8. Build the store (cell 12, 1-2 hours)

First run the cell that restarts Ollama (it was stopped for fine-tuning); `ollama list`
should show `gemma4:e4b`. The build cell then switches the legal profile's Agent 1 to the
adapter (it edits
`config/domains.json`), then sanitises all 300 documents: Agent 1 on GPU 1, Agents 2-3
through Ollama on GPU 0.

- **Expect:** one line per document (`[17] c007_intake_note.json: ok`), then
  `300 sanitised, 0 failed, N with residual leaks`.
- **N with residual leaks** should be 0 or very small. Any document listed there needs a look
  in Part D.
- **If it stops:** re-run C1-C3 and this cell. Finished documents are skipped.

### C9. Evaluate and report (cells 14-15, about 1 hour)

This runs both pipelines on the 40 QA questions, the 9 attacks with and without the monitor,
the 20 extraction prompts and the 200 membership probes (roughly 600 Gemini calls), then
writes `REPORT.md` and the charts.

- **Expect:** the report table, then four charts.
- **If it stops** with `all N Gemini keys are rate-limited`: wait the time it states and
  re-run the cell. Answers already received come from the cache (Part F).

### C10. Export (cell 17)

- **Expect:** `bundle.zip` with its size.
- Download it from the right-hand panel: Output → `/kaggle/working/bundle.zip`.
- Then click **Save Version → Quick Save** so the outputs are kept.

---

## Part D: checking each agent

`python -m sanitise.legal check --doc <file>` (add `--mock` on a laptop) prints each agent's
output. Go through these checklists with the original text beside you.

### D1. Agent 1: detection (rules + LLM pass + merge)

Each line shows: category, the exact text, `rule` or `llm`, and the character positions.

- [ ] **Direct identifiers, every one:** the client's full name and surname, the opposing
      party, any witness, the lawyer and the judge as NAME; phone; email; PAN and Aadhaar as
      ID_NUMBER; bank account and IFSC as FINANCIAL; the case number; the full address as
      one ADDRESS span.
- [ ] **Quasi-identifiers:** age, occupation, city (LOCATION), dates, the court, the amount
      (MONEY), the company (ORGANIZATION, without `M/s`), any `Project …` code name.
- [ ] **The legal issue** (`cheque bounce`, `land dispute`, …) as LEGAL_ISSUE. Only the LLM
      pass finds this, so in mock mode it appears under `gold spans missed`.
- [ ] **Merge:** no span appears twice and no two overlap. A COURT span such as
      "Pune District Court" is not also split into a LOCATION.
- [ ] **QUASI_COMBO** (LLM pass only): for an intake note or case note, one combination such
      as `age (52) + occupation (cardiologist) + city (Pune) + issue (cheque bounce)`.
- [ ] `gold spans missed:` is empty, apart from LLM-only items in mock mode.

**Scored version:** the Agent 1 rows of `REPORT.md` give recall on direct identifiers
(target ≥ 0.95) and precision (target ≥ 0.80) over all 300 documents. `agent1_before.json`
and `agent1_after.json` score the LLM pass alone. To score the rule pass alone, run
`python eval/score_rules.py data/legal/gold --show`; it lists every miss and false positive.

### D2. Agent 2: facts

Each line shows subject, relation, importance and value.

- [ ] Subjects are `client`, `opposing_party`, `witness` or `person_<n>` (lawyers, judges).
- [ ] **High-importance facts are present:**
  - `has_legal_issue` and `filed_in` (the court) in every document type;
  - `advised` and `documents_requested` in advice emails;
  - `advised`, `outcome` and `next_step` in case notes.
- [ ] Every value is copied from the document; nothing is invented. This is the spec's
      faithfulness target.

**Scored version:** the Agent 2 rows of `REPORT.md`, coverage of high-importance facts
(target ≥ 0.80) and faithfulness (target ≥ 0.90), against `data/legal/gold_facts/`.

### D3. Agent 3: routing

Each line shows category, action and the replacement.

- [ ] **placeholder:** NAME (`[CLIENT_1]` for the client, `[PARTY_1]` for the other side,
      `[WITNESS_1]`, `[PERSON_n]` for lawyers and judges), PHONE, EMAIL, ID_NUMBER,
      FINANCIAL, CASE_NUMBER, ADDRESS, MATTER_CODENAME.
- [ ] **generalise:**
  - court → "a magistrate's court" / "a High Court" / "a district court";
  - amount → a band such as "between ₹10 lakh and ₹1 crore";
  - company → its sector ("a real-estate developer");
  - date → month and year.
- [ ] When age, occupation and city appear together, each climbs **two** steps:
      "middle-aged", "a professional", "a city in western India".
- [ ] **keep:** the legal issue, unless it is one of the four rare issues (maritime salvage,
      geographical indication, aircraft lease, cross-border insolvency), which are
      generalised.
- [ ] The surname alone ("Dear Mr. Bose") gets the same token as the full name when both are
      in the document.

### D4. Agent 3: the safe text and the leak check

- [ ] Search the safe text for every direct identifier you ticked in D1. None may appear,
      in any spelling or spacing.
- [ ] The advice, documents, outcome and next step are still there, word for word. This is
      what the QA questions look for.
- [ ] The text reads naturally (with the real model it is rewritten from the facts; in mock
      mode the details are replaced in place).
- [ ] `leak check: no residual leaks`.

**Scored version:** the Agent 3 rows of `REPORT.md`: leak rate (target ≤ 0.05), routing
accuracy (≥ 0.85), utility (≥ 0.80), and documents with a direct identifier left (must be 0).

### D5. The answer step

- [ ] The `ANSWER` block is built from the safe text only. It may contain `[CLIENT_1]` but
      never a real name.
- [ ] For the advice question, it states the advice.

---

## Part E: checking the whole system

### E1. The report (`/kaggle/working/results/REPORT.md`)

Every protected row has a target and a **Met** column. What to expect from a real run:

| Metric | Target (protected) | Note |
|---|---|---|
| Extraction ROUGE-L / exact identifier matches / leaked chunks | ≤ 0.20 / 0 / ≤ 5% | the baseline will be much higher |
| PII leakage rate | ≤ 1% | |
| Membership inference AUC / TPR at 1% FPR | ≤ 0.60 / ≤ 5% | a value a little above 0.5 is expected and honest: the issue and the advice are kept |
| QA accuracy | within 10 points of the baseline; retention ≥ 0.90 | the cost of protection |
| Online latency | ≤ 1.2× | |
| Recall@4 | ≥ 0.85 | identical for both pipelines |

Write down every `**no**`. Each one is either a real finding for the report or something to
fix; the agent-level rows show which agent is responsible.

### E2. Attack outcomes

| Attack | Baseline | Protected without monitor | Protected with monitor |
|---|---|---|---|
| direct_c00x | identified in round 1-2 | never | never |
| linkage_c00x | identified in round 1 | never | never |
| stepwise_c00x | identified by round 3-4 | never, or later than the baseline | never; refused from the round where fewer than 5 clients would match |

For `stepwise_c001` the monitor should block round 2 ("which one involves a doctor?"),
because 12 cheque-bounce clients in Maharashtra drop to 3 doctors.

### E3. Replay the Kaggle run in the dashboard

1. Unzip `bundle.zip` into the repository folder, so that you have `3rd_sem_el\bundle\`.
2. Start the API:

   ```powershell
   $env:SANITISE_DOMAIN = "legal"; $env:SANITISE_REPLAY_DIR = "bundle"; sanitise serve
   ```

3. Start the dashboard with `cd web; npm run dev` and open http://localhost:3000.
4. The Documents tab shows all 300 records with Gemma 4's real output. Compare and Attacks
   replay the recorded Gemini answers exactly, monitor verdicts included.
5. A question that was not part of the run gets a canned quote instead. For the demo, use
   the QA questions and the attacks (docs/legal/demo_script.md).

---

## Part F: API keys, rotation and tokens

### F1. Where keys go

| Where you run | Put keys in |
|---|---|
| Kaggle | Add-ons → Secrets: `GEMINI_API_KEY`, `GEMINI_API_KEY_2` … `GEMINI_API_KEY_9` |
| Laptop | `.env` in the repository folder (gitignored): `SANITISE_GEMINI_API_KEYS=key1,key2,key3` (the `.env` file only feeds `SANITISE_` settings; plain `GEMINI_API_KEY*` names work as real environment variables) |

`preflight` lists the keys it found as 8-character fingerprints, never the keys themselves.

### F2. How rotation works

Rotation is implemented by `KeyPool` in `src/sanitise/llm/gemini.py`.

- Calls take the keys in turn.
- **Per-minute limit** (HTTP 429 / RESOURCE_EXHAUSTED): that key rests for
  `SANITISE_GEMINI_KEY_COOLDOWN_S` seconds (default 60), and the same call is retried
  immediately on the next key.
- **Daily quota** (the error mentions "per day"): that key rests for 24 hours.
- **Your own daily cap:** `SANITISE_GEMINI_DAILY_TOKEN_BUDGET=200000` (for example) retires a
  key for the day once it has used that many tokens. 0 means no cap.
- Only when every key is resting does a call fail, with a message saying when the first key
  is free again.

### F3. Not running out

- **Cache.** Every Gemini answer is saved in `/kaggle/working/gemini_cache`. Re-running a cell
  or resuming after an error costs nothing for answers already received. Keep
  Persistence = Files only so the cache survives.
- **Usage.** `gemini_cache/usage.json` shows tokens per key per day, by fingerprint.
- **Rate.** Lower `SANITISE_GEMINI_RPM` (cell 2) if keys keep hitting per-minute limits.
- **Size.** A full evaluation is about 600 calls and roughly a million tokens (an estimate).
  Adding `--mi-limit 50` to the evaluate command removes about a third of the calls.
- Only answering uses Gemini. Agents 1-3 run on Gemma 4 inside Kaggle and use no API tokens.
- Kaggle's GPU allowance is about 30 hours a week, separate from Gemini. Run fine-tuning and
  the build in different sessions if needed; both resume.

### F4. Replacing, revoking and the rules

- **To replace a key:** edit the secret's value in Add-ons → Secrets, then re-run cell 2.
- **If a key leaks** (pasted into a cell, a commit or a chat): delete it in AI Studio at once
  and create a new one.
- **Use the team's own keys, one per person.** Creating extra Google projects only to
  multiply the free quota may break Google's terms. If four keys are not enough for the
  final run, switch one key to the paid tier.

---

## Part G: domains

The pipeline is the same in every domain: Agent 1 detects (rules, then the LLM pass, then
merging and de-duplication), Agent 2 extracts facts, Agent 3 routes, rewrites and
leak-checks, and then come search, the answer and the session monitor. A domain changes only
the context: its prompts, its data and its models.

- **Models per domain:** `config/domains.json`. For example, the legal profile sets
  `"model": "gemma4:e4b"` for Agents 2-3, `"answer_backend": "gemini"` and
  `"gemini_model": "gemini-3.5-flash"` for answers, and `"agent1_backend"` /
  `"agent1_adapter"` for Agent 1. A field set to `null` falls back to the global setting.
- **Choosing a domain:**
  - in the dashboard, the header dropdown;
  - in the API, `?domain=legal` on any request or the header `X-Sanitise-Domain: legal`;
  - `GET /v1/domains` lists the domains and their models;
  - `SANITISE_DOMAIN` sets the default.
- **Checking:** run `python -m sanitise.legal preflight`; the Domains block shows each
  domain's models. In the dashboard, hover over the dropdown.
- **Adding the third domain:**
  1. Copy the legal entry in `config/domains.json` and give it a new name.
  2. Point its data paths (`store_path`, `attacks_dir`, `clients_path`, `mock_corpus`) at its
     own folders.
  3. Set `"prompts"` to the closest existing family until its own prompts are written.

---

## Part H: troubleshooting

| Symptom | Cause and fix |
|---|---|
| `441 passed` but fewer, or errors importing spaCy | the spaCy model is missing: `python -m spacy download en_core_web_lg` |
| preflight `FAIL ... 0 key(s)` on Kaggle | secret not attached to this notebook; Add-ons → Secrets → Attach, re-run cell 2 |
| preflight `FAIL model gemma4:e4b ... not pulled` | re-run cell 3; check the tag on ollama.com |
| `all N Gemini keys are rate-limited` | wait the time given, then re-run; add keys or lower `SANITISE_GEMINI_RPM` |
| 401 / 403 downloading `google/gemma-4-E4B-it` | accept the licence on Hugging Face; check the `HF_TOKEN` secret is attached |
| training loss `nan` | add `--fp32` to the train command |
| CUDA out of memory when fine-tuning ("Process … has 8.9 GiB in use") | Ollama still holds Gemma 4 on the GPU: run the free-the-GPUs cell (`pkill -f "ollama serve"`), check `nvidia-smi`, restart the session if memory is still used |
| CUDA out of memory during the build | Agent 1 must be on `cuda:1` and Ollama on GPU 0, as the cells set; restart the session and run C1-C3 then C8 |
| `LLM_TIMEOUT` during the build | raise `SANITISE_LLM_TIMEOUT_S` in cell 2 (Gemma on a T4 can take over a minute on long documents) |
| session ended mid-build or mid-evaluation | re-run C1-C3, then the same cell; finished work is skipped or cached |
| dashboard says "API unreachable" | the API is not running, or `web/.env.local` has the wrong `NEXT_PUBLIC_API_URL` |
| `400 UNKNOWN_DOMAIN` | the domain name is not in `config/domains.json` |
| dashboard shows medical data under Legal | start the API from the repository folder, so it finds `config/domains.json` and `data/legal` |
