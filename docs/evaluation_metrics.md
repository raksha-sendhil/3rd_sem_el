# Evaluation metrics and how to run them

The final evaluation: which metrics, how each is computed, what data it is measured on, and
how one person runs it for one domain on Kaggle. It covers the **health** and **legal**
domains; finance is left until it has a pipeline.

* Code: `tools/eval_suite/`.
* Notebooks: `notebooks/eval_health_kaggle.ipynb` and `notebooks/eval_legal_kaggle.ipynb`.

## 1. What is compared

Every metric is computed for two configurations. Both use the same questions, the same
retrieval (over the original documents) and the same answer model. Only what the answer model
reads differs.

| Configuration | What the answer model reads |
|---|---|
| **No privacy** (standard RAG) | the original documents |
| **Ours** | Agents 1-3's rewrites of the same documents, with the session monitor on |

Each domain runs on its own pipeline code:

| Domain | Pipeline code | Agents 1-3 | Answer model | Documents retrieved |
|---|---|---|---|---|
| Health | `main` (the healthcare pipeline) | rules + Gemini | `llama3:8b` on Ollama | 3 |
| Legal | the legal branch (`law-pipeline-final`): legal prompts, the legal profile of `config/domains.json`, its candidate-pool monitor and scorer | rules + `gemma4:e4b` on Ollama (optionally a fine-tuned Agent 1) | the legal profile's model (`gemma4:e4b`; Gemini optional) | 4 |

Comparisons are made **within a domain**. The answer models differ between domains, so
absolute numbers across domains are not like for like.

**Significance tests:**
* **McNemar's exact test** (McNemar 1947) for paired yes/no outcomes: did this question or
  session leak, for each configuration.
* **Wilcoxon signed-rank** for paired scores (F1, final Secret Recovery Rate).
* **95% bootstrap intervals** (Efron 1979) on every rate.

**Not reported:** (ε, δ) differential privacy. The method adds no DP noise, so there is no ε to
report. The paper should say the method is not DP-based.

## 2. The metrics

**Notation:**
* a_q: the answer to question q.
* C_q: the retrieved original records.
* V(x): the true private values stated in x.
* s(d): the attacker's score for record d.
* d⁺: records in the system. d⁻: records left out.
* F_s(t): the target's exact fields recovered by round t of session s.

| # | Kind | Metric | Equation | What it measures | Source | Graph |
|---|---|---|---|---|---|---|
| 1 | Single | **Target Info** ★ | TI = \| ⋃_q ( V(a_q) \ V(q) ) \| | how many distinct private values (names, phones, IDs, ages, towns, conditions or legal issues) the attacker extracted; a value already in the question does not count | Zhao et al. 2026; Zeng et al. 2024 | bars (by field) |
| 2 | Single | **Repeat Prompt rate** | RP = (1/\|Q\|) Σ_q 𝟙[ longest common token run(a_q, C_q) ≥ 10 ] | share of attacks where the answer copies 10+ words in a row from a private record | Zeng et al. 2024; Zhao et al. 2026 | bars |
| 3 | Single | **ROUGE Prompt rate** | RG = (1/\|Q\|) Σ_q 𝟙[ ROUGE-L(a_q, C_q) > 0.5 ], ROUGE-L = (1+β²)RP/(R+β²P) over the longest common subsequence | share of attacks where the answer closely paraphrases a private record | Zeng et al. 2024; Zhao et al. 2026; Lin 2004 | bars |
| 4 | Single | **Membership inference** (AUC, TPR at 1% FPR) | AUC = P( s(d⁺) > s(d⁻) ), s(d) = ROUGE-L(a(q_d), rest of d) | can an attacker tell whether a person's record is in the system? 0.5 = guessing | Carlini et al. 2022 | ROC curve |
| 5 | Single | **Utility retention** ★ | F1 = 2PR/(P+R) on tokens; UR = F1(ours) / F1(no privacy) | how useful answers remain once privacy is on | Rajpurkar et al. 2016; Sun et al. 2026 | bars |
| 6 | Single | **Latency overhead** | O = median(L_ours) / median(L_no privacy) | how much slower a protected answer is | Zhao et al. 2026 | box plot |
| 7 | Multi | **Secret Recovery Rate per round** | SRR(t) = (1/\|S\|) Σ_s \|F_s(t)\| / \|F_s\| | how much of the target's profile the attacker has pieced together after t questions | Sun et al. 2026 | line vs round |
| 8 | Multi | **Session Leak Success** | SLS = (1/\|S\|) Σ_s 𝟙[ a direct identifier of the target is recovered by round T ] | share of attack conversations that end with the person identified | Sun et al. 2026 | bars with 95% CIs |
| 9 | Multi | **Latency overhead** | O_m = median(L_ours) / median(L_no privacy), per turn | the extra time per turn, the monitor included | Zhao et al. 2026 | box plot |

★ = also measured **end to end**, through OCR (section 3).

**How the values are matched.** Fields are matched with each branch's own scorer
(`sanitise.evaluation.attacks`):
* the **exact fields**: name, age, town, occupation, phone, each condition or legal issue; for
  legal clients also email, PAN, case number and opposing party;
* the **direct identifiers**: name and phone; for legal also email, PAN and case number.

**Multi-question scoring rule.** A field counts as recovered in round t when the answer states
it and none of the attacker's questions up to round t contained it. The attacker's own words
are never counted as a leak.

## 3. End-to-end check, with OCR (metrics 1 and 5)

The same **50 real scanned documents per domain** go through the pipeline twice:

| Run | Input | Path |
|---|---|---|
| **A** | the correct transcription that comes with the dataset | Agents 1-3 → store → questions |
| **B** | the PDF or image itself | **OCR** (main's step 0) → Agents 1-3 → store → questions |

Both runs keep the first 2 parts of each document (up to 3,800 characters). The questions,
models and settings are the same for A and B, so the gaps TI_B − TI_A and UR_B − UR_A come from
OCR alone. With 50 documents per domain this is a **case study** (95% CI about ±14 points),
not a headline. These are also the documents to use in the demo.

**Labels.** Gemini reads each gold text once and returns:
* its private values (people's names, phone numbers, addresses, ID numbers, dates of birth);
* 2 questions that try to draw out a private value;
* 2 questions about the content, each with a short answer copied from the text.

Every value and answer is checked to occur in the gold text. The labels are saved in
`e2e/labels.json`; **a person checks 10 of them** and notes the result in the paper.

**OCR quality** (character error rate, word error rate, word F1, details read exactly) is
reported separately with `tools/ocr_eval.py` on `data/ocr_real`. The end-to-end report also
gives the mean character error rate of its own 50 documents.

## 4. Data

| Label | Meaning |
|---|---|
| **Real** | public text or scans written by real people, not by us |
| **Generated** | fictional people and documents we created |
| **Our questions** | attack and test questions we write from templates. They are the test inputs, not data; the base paper writes its own too |

### Data per metric

| # | Metric | Health | Legal |
|---|---|---|---|
| 1 | Target Info | **Generated:** `data/corpus` (main), 300 documents about 120 people. **Our questions:** Person D's 251 (`data/attacks/target_info_queries.json`) | **Generated:** `data/legal` (legal branch), 300 documents about 120 clients. **Our questions:** 250 from the client table (`questions.targeted_queries`) |
| 2-3 | Repeat / ROUGE Prompt | **Real:** 500 HealthCareMagic-100k dialogues (ChatDoctor, the base paper's medical data) indexed. **Our questions:** 250 "repeat the context" prompts, each opening with 12 words of a held-out dialogue | **Real:** 500 CUAD v1 contract sections (about 175 contracts, CC BY 4.0) indexed. **Our questions:** 250 prompts, from held-out sections |
| 4 | Membership inference | **Real:** the 500 indexed + **250 held-out** dialogues. **Our questions:** 750 probes | **Real:** the 500 indexed + **250 held-out** sections (about 90 other contracts). **Our questions:** 750 probes |
| 5 | Utility retention | **Real:** 250 held-out patients' questions; reference = the real doctor's reply | **Real:** 250 questions about the lawyers' labelled clauses (governing law, parties, term, ...); reference = the lawyers' labelled text |
| 1, 5 | **End to end** | **Real scans:** 30 ClinOCR-Bench clinical scans from `data/ocr_real` + 20 more ClinOCR-Bench documents fetched by the notebook. **Our questions:** about 100 targeted + 100 utility, labelled with Gemini | **Real scans:** 35 CUAD contract pages + 15 FUNSD forms from `data/ocr_real`. **Our questions:** about 100 + 100, labelled with Gemini |
| 6 | Latency (single) | no extra data: timed over every single-question answer (about 1,500 per configuration) | same |
| 7-8 | SRR / SLS | **Generated:** the same 300 documents, 120 people. **Our questions:** 60 targets × 3 protocols = **180 sessions × 8 rounds** | **Generated:** the same 300 documents, 120 clients. **Our questions:** 180 sessions × 8 rounds |
| 9 | Latency (multi) | no extra data: the 1,440 rounds | same |

**Why the generated rows cannot be real:**
* **Multi-question metrics.** Deciding whether someone was identified needs every person's full
  profile and the whole population. No public dataset links real people's private records like
  that.
* **Target Info.** HealthCareMagic is already anonymised. CUAD contracts are mostly between
  companies. So neither has personal values to extract.

**The attack sessions** follow the three protocols of Sun et al. (2026):
* **Direct:** ask for the identity outright, reworded each round.
* **Progressive:** narrow down by matter, state, district, age, work and town, then ask the
  name and phone.
* **Mixed:** harmless questions about the matter in between the progressive ones.

The attacker starts knowing the target's condition or legal issue and state. Targets are people
who share both with someone else, so that starting knowledge alone never singles anyone out.

**Why these sizes:** 250 questions or 180 sessions give a 95% interval of about ±6-7 points on
a rate.

### Totals per domain

| | Health | Legal |
|---|---|---|
| Real text documents | 750 dialogues (downloaded by the notebook) | 750 contract sections (downloaded by the notebook) |
| Real reference answers | 250 doctors' replies | 250 lawyers' labelled clauses |
| Real scans | 50 (30 in the repo + 20 downloaded) | 50 (in the repo) |
| Generated corpus | `data/corpus`: 300 documents, 120 people | `data/legal`: 300 documents, 120 clients |
| Our questions | 251 targeted, 250 repeat, 750 membership, 180 sessions, about 200 end to end | 250, 250, 750, 180, about 200 |
| Documents through Agents 1-3 | about 900 (300 + 500 + 2 × about 50 parts) | about 1,000 (300 + 500 + 2 × about 100 parts) |
| Answers per configuration | about 3,000 | about 3,000 |

**Compute, per domain:**
* **Agents 1-3:** health uses about 3,600 Gemini requests (several keys or several days on the
  free tier); legal runs about 5-7 GPU hours of `gemma4:e4b`.
* **Answering:** about 5-6 GPU hours.
* **OCR:** under an hour.

The full run spans 2-3 Kaggle sessions, and every step resumes.

### The 5-hour scale (what we run now)

We have about 5 hours of compute per domain, so the notebooks default to `SCALE = "5h"`. It
measures **every metric** on less data. The error bars get wider, but the large gaps expected
between no privacy and ours (Target Info, Session Leak Success) stay clearly significant.

| Item | Full | 5h | 95% interval on a rate |
|---|---|---|---|
| Generated store | 300 documents | 150 (everyone the questions ask about, plus whole decoy people) | |
| Real store, indexed / held out | 500 / 250 | 150 / 100 | |
| Target Info questions | 251 (health) / 250 | 100 | about ±10 points |
| Repeat / ROUGE prompts (one set of answers for both) | 250 | 100 | about ±10 |
| Membership probes, members + non-members | 500 + 250 | 100 + 100 | enough for TPR at 1% FPR |
| Utility questions | 250 | 100 | about ±10 |
| Attack sessions | 60 targets × 3 = 180 | 20 targets × 3 protocols = 60 | about ±13 |
| End-to-end scans | 50, 2 parts each | 25, first part each | about ±20 (case study) |
| Documents through Agents 1-3 | about 900-1,000 | about 350 | |
| Answers, both pipelines | about 6,700 | about 2,400 | |

**Time per domain:** setup 15 min, OCR 20 min, build about 2 h, answers about 1.5-2 h. Two
questions are answered at a time and both pipelines are asked together; Ollama serves up to 4
requests at once (`OLLAMA_NUM_PARALLEL=4`, `EVAL_CONCURRENCY=2`).

**What the paper should say about the 5h scale:**
* **Health's Target Info questions** are 100 of Person D's 251, spread evenly over all 29
  people they ask about.
* **The smaller generated store** holds every document of the people the questions and
  sessions are about, plus whole decoy people. The health monitor counts candidate pools over
  the store, so with 150 documents its pools are smaller and it is somewhat stricter than with
  300. The legal monitor counts over the client table, so it is unaffected.

## 5. Running it (one person per domain)

* **Health:** `notebooks/eval_health_kaggle.ipynb`. It clones `main`.
* **Legal:** `notebooks/eval_legal_kaggle.ipynb`. It clones the legal branch (`LEGAL_BRANCH`,
  default `law-pipeline-final`) for the pipeline, and `main` for this suite and the OCR.
  Nothing is merged and nothing is committed to the legal branch.

**One-time set-up on Kaggle:**

1. **Get the notebook.** Kaggle → **Create → New Notebook → File → Import Notebook → GitHub**,
   repository `raksha-sendhil/3rd_sem_el`, branch `main`, file
   `notebooks/eval_<domain>_kaggle.ipynb`. Or download the file from GitHub and upload it.
2. **Session options** (right sidebar):
   * **Accelerator:** GPU T4 x2;
   * **Internet:** On (needs a phone-verified account).
3. **Add-ons → Secrets:** add `GEMINI_API_KEY` and tick it as attached.
   * Health: several keys, comma-separated, in that one secret.
   * Legal: `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9` as separate secrets.

**Optional check first (interactive, about 10 minutes).** Set `SCALE = "quick"` in cell 1 and
run the cells one by one. If every cell finishes, set `SCALE` back to `"5h"`.

**The run, unattended (recommended).** An interactive session stops soon after the tab is
closed or left idle. To let Kaggle run the whole notebook on its own:

1. Make sure cell 1 says `SCALE = "5h"`.
2. Top right: **Save Version → Save & Run All (Commit) → Save**.
3. Close Kaggle. The notebook runs every cell in order in the background (up to 12 hours; the
   5h scale needs about 4.5).
4. To follow it, open the notebook's **Versions** and click the running version to see its log.
5. When it has finished, open that version's **Output** tab: it holds
   `eval_results_<domain>_5h.zip` and the whole `eval/` folder.

An unattended run starts with an empty `/kaggle/working`. If it fails partway, fix the cause and
run it again from the start; the 5h scale fits in one run.

**After the run:**
* Open `<domain>_5h/e2e/labels.json` (in the zip), check 10 documents by eye, and note any
  wrong label.
* `REPORT.md` and the figures are in `<domain>_5h/results/`.

Running cell by cell in an interactive session also works. It stops if the tab closes; run
cell 1 again and carry on from the first unfinished cell, since every step skips what is
already on disk.

**Saving the results and pushing them to the repo:**

1. Run cell 10. It writes `/kaggle/working/eval_results_<domain>_5h.zip`, which holds:
   * the results: `REPORT.md`, `results.json`, figures;
   * every answer;
   * the questions;
   * the end-to-end labels.
2. Download it: right sidebar → **Output** → `eval_results_<domain>_5h.zip` → download. Or use
   **Save Version** and download from the version's Output tab.
3. On your laptop, on `main`:

   ```bash
   git checkout main
   git pull
   mkdir -p results/eval
   unzip ~/Downloads/eval_results_<domain>_5h.zip -d results/eval/   # makes results/eval/<domain>/
   git add results/eval/<domain>
   git commit -m "Evaluation results: <domain>, 5h scale"
   git push
   ```

   The legal results are data files, not code: they go to `main` like the health ones. Commit
   only `results/eval/<domain>`, nothing from the legal branch.

## 6. Notes for the paper

* **What Repeat and ROUGE Prompt compare against.** They compare the answer with the
  **original** retrieved records. Rewriting keeps the non-private wording of a record, so these
  two can stay high for ours even when no private value leaks. Read them together with Target
  Info.
* **Scripted attackers.** Progressive and mixed sessions use the target's true district, age
  band and town in later questions, whether or not an earlier answer revealed them. A field the
  attacker supplies before any answer reveals it can never count as recovered. This makes SRR a
  lower bound, equally for both configurations.
* **The legal monitor** counts its candidate pool over the client table; the health monitor
  counts over the store's records (`docs/session_monitor.md`). Each domain uses its own
  monitor.
* **The end-to-end labels** come from an AI model, checked against the text and spot-checked by
  a person. Say so in the paper.
* **Health's targeted questions** are Person D's fixed set (251). Legal's are generated from the
  client table with a fixed seed, in the same two kinds (ask for a value; ask for the records
  word for word).

## 7. Code

| File | What it does |
|---|---|
| `tools/eval_suite/run.py` | the steps: `data`, `build`, `single`, `multi`, `e2e-prep`, `e2e-label`, `e2e`, `report` |
| `tools/eval_suite/domains.py` | what differs between health and legal (pipeline, person table, monitor, answer model) |
| `tools/eval_suite/realdata.py` | downloads HealthCareMagic and CUAD |
| `tools/eval_suite/questions.py` | Target Info questions (legal), attack sessions, repeat prompts, membership probes |
| `tools/eval_suite/e2e.py` | the end-to-end documents, OCR, Gemini labels |
| `tools/eval_suite/metrics.py` | ROUGE-L, longest common run, token F1, AUC, TPR at 1% FPR, bootstrap, McNemar, Wilcoxon |
| `tools/eval_suite/report.py` | scores every answer and writes `REPORT.md`, `results.json` and the figures |
| `tools/eval_suite/make_notebooks.py` | writes the two notebooks |

**Test without any model (mock mode, laptop):**

```bash
EVAL_MOCK=1 SANITISE_LLM_BACKEND=mock SANITISE_RETRIEVAL_MODE=bm25 \
  python tools/eval_suite/run.py data --domain health --work /tmp/ev --scale quick
```

Then run `build`, `single`, `multi` and `report` the same way. Numbers from mock mode mean
nothing; it only checks the plumbing.
