"""Write notebooks/legal_kaggle_gemini.ipynb: the legal pipeline with every agent on Gemini.

The notebook is generated so its cells stay in step with the code; edit the CELLS list below and
run ``python tools/legal/make_notebook_gemini.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from make_notebook import REPO, notebook

BRANCH = "law-version-2"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Private Matter Assistant - Kaggle notebook, all agents on Gemini

Agents 1-3 and the answers all run on `gemini-3.5-flash`. There is no Ollama, no Gemma 4 and no
fine-tuning here (Gemini cannot be fine-tuned). Only search stays local: bge-m3 and BM25.

**Raw text goes to Google.** The agents read the original documents, so this run breaks the
design's rule that the outside model only sees cleaned text. That is acceptable here because the
corpus is fictional. Never run this branch on real client documents.

**Settings:** Accelerator **GPU T4** (one GPU is enough, for bge-m3), Internet **on**,
Persistence **Files only**.
**Secrets** (Add-ons -> Secrets, each ticked for this notebook; never paste a key into a cell):
`GEMINI_API_KEY`, optionally `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9`.
Step-by-step: docs/legal/kaggle_steps_gemini.md.

| Part | Cells | Requests (30 documents) | Time at 10 a minute |
|---|---|---|---|
| Setup | packages, keys, data, preflight | 0 | ~10 min |
| Check | three documents through every agent | ~12 | ~2 min |
| Build | Agents 1-3 (resumes) | ~90 | ~10 min |
| Evaluate | both pipelines, attacks, metrics, charts | ~135 | ~15 min |
| Export | the dashboard bundle to download | 0 | ~1 min |

The first run covers 30 documents. To run all 300 (about 1,500 requests), set `DOCS = None` in
the settings cell and run the build cell again: finished documents are skipped.""",
    ),
    (
        "code",
        f"""# Packages and the code
!git clone -q -b {BRANCH} {REPO} /kaggle/working/repo 2>/dev/null || (cd /kaggle/working/repo && git pull -q)
%cd /kaggle/working/repo
!pip install -q -U transformers google-genai presidio-analyzer sentence-transformers chromadb \\
    rank_bm25 rapidfuzz rouge-score matplotlib
!pip install -q -e .
!python -m spacy download en_core_web_lg""",
    ),
    (
        "code",
        """# Keys from Kaggle secrets (never typed into a cell), and the settings
import json, os
from kaggle_secrets import UserSecretsClient

DOCS = 30        # documents to build; None = all 300
MI_PROBES = 10   # membership probes per group; 100 for the full evaluation

secrets = UserSecretsClient()
keys = []
for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]:
    try:
        keys.append(secrets.get_secret(name))
    except Exception:
        pass  # not attached to this notebook
os.environ["GEMINI_API_KEYS"] = ",".join(keys)  # rotated when one hits its limit
print(f"{len(keys)} Gemini key(s) loaded")

W = "/kaggle/working"
LIMIT = f"--limit {DOCS}" if DOCS else ""
os.environ.update({
    "SANITISE_DOMAIN": "legal",
    "SANITISE_GEMINI_CACHE_DIR": f"{W}/gemini_cache",  # cached replies + per-key token use
    "SANITISE_GEMINI_RPM": "10",                       # calls per minute, across all keys
    "SANITISE_GEMINI_KEY_COOLDOWN_S": "60",            # rest after a per-minute limit
    "SANITISE_GEMINI_MAX_KEY_WAIT_S": "300",           # all keys resting: wait, do not fail
    "SANITISE_GEMINI_DAILY_TOKEN_BUDGET": "0",         # per key per day; 0 = no budget
    "SANITISE_RETRIEVAL_MODE": "hybrid",
    "SANITISE_LLM_TIMEOUT_S": "300",
})
# The models live in config/domains.json. Print them so the run log records what was used.
print(json.dumps(json.load(open("config/domains.json"))["legal"], indent=1))""",
    ),
    (
        "code",
        """# Generate the fictional data with the fixed seed, and the ontology tables
!python -m sanitise.legal generate --data data/legal
!python -m sanitise.legal ontology
!python -m sanitise.evaluation.validate_gold --gold data/legal/gold | tail -3""",
    ),
    (
        "code",
        """# Preflight: every required line must say OK (keys, spaCy, data)
!python -m sanitise.legal preflight""",
    ),
    (
        "code",
        """# Check the agents on one document at a time before the longer runs. A schema or model problem
# shows up here, after a few requests, instead of on every document of the build.
!python -m sanitise.legal check --doc c001_intake_note --question "What is this matter about?"
!python -m sanitise.legal check --doc c001_advice_email --question "What did we advise?"
!python -m sanitise.legal check --doc c001_case_note --question 'What was the outcome?'""",
    ),
    (
        "markdown",
        """## Build: Agents 1-3 on Gemini

Re-run this cell after a restart or a quota error: it skips documents already in the store, and
replies already received come from the cache.""",
    ),
    (
        "code",
        """!python -m sanitise.legal build --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results {LIMIT}""",
    ),
    (
        "code",
        """# How Gemini behaved: compare these with the Gemma 4 run
import glob, statistics
from collections import Counter

records = [json.load(open(p)) for p in glob.glob("/kaggle/working/processed/*.json")]
print(f"{len(records)} documents; models: {Counter(r['meta']['model'] for r in records)}")
print(f"fell back: {Counter(f for r in records for f in r['meta']['fallback']) or 'none'}")
print(f"repair retries: {sum(r['meta']['retries'] for r in records)}")
print(f"documents with residual leaks: {sum(bool(r['residual_leaks']) for r in records)}")
if records:
    for stage in ("detect_ms", "extract_ms", "rewrite_ms", "total_ms"):
        print(f"median {stage}: {statistics.median(r['meta'][stage] for r in records):.0f}")
print("tokens per key per day:", open("/kaggle/working/gemini_cache/usage.json").read())""",
    ),
    (
        "markdown",
        """## Evaluate: online path, attacks and metrics

Both pipelines answer with Gemini from the same four documents; every attack runs with and
without the session monitor; then the metrics, the agent-level scores and the charts. Scores from
a partial build only cover the documents built.""",
    ),
    (
        "code",
        """!python -m sanitise.legal evaluate --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results --mi-limit {MI_PROBES}
!python -m sanitise.legal report --out /kaggle/working/results
from IPython.display import Markdown, display
display(Markdown(open("/kaggle/working/results/REPORT.md").read()))""",
    ),
    (
        "code",
        """from IPython.display import Image
for chart in ("privacy_utility", "membership_roc", "recall_at_k", "pii_leakage"):
    display(Image(f"/kaggle/working/results/charts/{chart}.png"))""",
    ),
    (
        "markdown",
        """## Export for the dashboard

Download `bundle.zip`, unzip it on the laptop and start the API with
`SANITISE_REPLAY_DIR=<bundle>`. Then click **Save Version** so the outputs are kept.""",
    ),
    (
        "code",
        """!python -m sanitise.legal export --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results --bundle /kaggle/working/bundle
!cd /kaggle/working && zip -qr bundle.zip bundle && ls -lh bundle.zip""",
    ),
]


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks" / "legal_kaggle_gemini.ipynb"
    text = json.dumps(notebook(CELLS), indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
