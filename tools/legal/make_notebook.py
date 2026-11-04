"""Write notebooks/legal_kaggle.ipynb, the one Kaggle notebook of the legal build spec.

OWNER: Person C. The notebook is generated so its cells stay in step with the code; edit the
CELLS list below and run ``python tools/legal/make_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/raksha-sendhil/3rd_sem_el.git"
BRANCH = "a/legal-pipeline"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Private Matter Assistant - Kaggle notebook (legal build spec)

Everything that sees raw (fictional) client text runs here: Gemma 4 E4B through Ollama for
Agents 2-3, Gemma 4 E4B + our LoRA adapter through `transformers` for Agent 1's LLM pass, and
bge-m3 for search. Only cleaned text goes to the Gemini API for the protected answers.

**Settings:** Accelerator **GPU T4 x2**, Internet **on**, Persistence **Files only**.
**Secrets** (Add-ons -> Secrets, each ticked for this notebook; never paste a key into a cell):
`GEMINI_API_KEY`, optionally `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9` (rotated when one runs
out), and `HF_TOKEN` (Gemma 4 is gated on Hugging Face). Step-by-step: docs/legal/kaggle_steps.md.

| Part | Cells | Time (T4) |
|---|---|---|
| 1 Setup | install, key, Ollama + `gemma4:e4b` | ~10 min |
| 1 Data | generate clients, corpus, gold, training set | ~1 min |
| 3 Fine-tune Agent 1 | before score, QLoRA, after score | ~1-2 h + 2 x 20 min |
| 2 Build | Agents 1-3 over 300 documents (resumes) | ~1-2 h |
| 4 Evaluate | retrieval, both pipelines, attacks, metrics, charts | ~1 h (Gemini rate limit) |
| 14 Export | the dashboard bundle to download | ~1 min |

Every output goes to `/kaggle/working`; click **Save Version** at the end. Fine-tuning and
building can run in separate sessions: both resume from what is on disk.""",
    ),
    (
        "code",
        f"""# Part 1, step 3: packages and the code
!git clone -q -b {BRANCH} {REPO} /kaggle/working/repo 2>/dev/null || (cd /kaggle/working/repo && git pull -q)
%cd /kaggle/working/repo
!pip install -q -U transformers accelerate peft bitsandbytes google-genai \\
    presidio-analyzer sentence-transformers chromadb rank_bm25 rapidfuzz rouge-score matplotlib
!pip install -q -e .
!python -m spacy download -q en_core_web_lg""",
    ),
    (
        "code",
        """# Part 1, step 4: keys from Kaggle secrets (never typed into a cell), and the settings
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
keys = []
for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]:
    try:
        keys.append(secrets.get_secret(name))
    except Exception:
        pass  # not attached to this notebook
os.environ["GEMINI_API_KEYS"] = ",".join(keys)  # rotated when one hits its limit
print(f"{len(keys)} Gemini key(s) loaded")
try:  # Gemma 4 on Hugging Face is gated: accept the licence, then add HF_TOKEN as a secret
    os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
except Exception:
    print("No HF_TOKEN secret: Part 3 needs it to download google/gemma-4-E4B-it")

W = "/kaggle/working"
os.environ.update({
    "SANITISE_DOMAIN": "legal",
    "SANITISE_LLM_BACKEND": "ollama",
    "SANITISE_GEMINI_CACHE_DIR": f"{W}/gemini_cache",  # cached answers + per-key token use
    "SANITISE_GEMINI_RPM": "10",                       # calls per minute, across all keys
    "SANITISE_GEMINI_KEY_COOLDOWN_S": "60",            # rest after a per-minute limit
    "SANITISE_GEMINI_DAILY_TOKEN_BUDGET": "0",         # per key per day; 0 = no budget
    "SANITISE_RETRIEVAL_MODE": "hybrid",
    "SANITISE_LLM_TIMEOUT_S": "300",
})
# The models per domain live in config/domains.json (legal: gemma4:e4b for Agents 2-3,
# Gemini for answers). Print them so the run log records what was used.
import json
print(json.dumps(json.load(open("config/domains.json"))["legal"], indent=1))""",
    ),
    (
        "code",
        """# Part 1, step 5: Gemma 4 for Agents 2-3 through Ollama on GPU 0 (GPU 1 stays free for Agent 1)
!curl -fsSL https://ollama.com/install.sh | sh
!CUDA_VISIBLE_DEVICES=0 nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
import time; time.sleep(5)
!ollama pull gemma4:e4b""",
    ),
    (
        "code",
        """# Step 1 (D): generate the fictional data with the fixed seed, and B's ontology tables
!python -m sanitise.legal generate --data data/legal
!python -m sanitise.legal ontology
!python -m sanitise.evaluation.validate_gold --gold data/legal/gold | tail -3""",
    ),
    (
        "code",
        """# Check the agents on one document before the long runs: Agent 1 (rules + LLM pass +
# merge), Agent 2 (facts), Agent 3 (routing + rewrite + leak check), one answer from Gemini
import asyncio, json
from sanitise.agents.pipeline import sanitise
from sanitise.config import get_settings
from sanitise.domain import settings_for, use_domain
from sanitise.llm.factory import build_answer_llm, build_llm_client

doc = json.load(open("data/legal/corpus/c001_intake_note.json"))
settings = settings_for(get_settings(), "legal")
agents = build_llm_client(settings)


async def check():
    with use_domain("legal"):
        record = await sanitise(doc["doc_id"], doc["text"], agents)
        print("ORIGINAL
" + doc["text"])
        print("
AGENT 1 spans:")
        for s in record.detector.spans:
            print(f"  {s.category.value:15} {s.text!r:45} {s.method}")
        print("
AGENT 2 facts:")
        for f in record.extractor.facts:
            print(f"  {f.subject:15} {f.relation:20} {f.value!r} ({f.importance.value})")
        print("
AGENT 3 safe text:
" + record.rewriter.safe_text)
        print("
residual leaks:", record.residual_leaks or "none")
        answer = await build_answer_llm(settings, agents).text_call(
            "Answer in one sentence.", "CONTEXT:
" + record.rewriter.safe_text
            + "

QUESTION: What did we advise?
ANSWER:")
        print("
GEMINI on the safe text:", answer)

await check()""",
    ),
    (
        "markdown",
        """## Part 3 - fine-tune Gemma 4 E4B for Agent 1 (QLoRA)

1. Score the base model on the 40 held-out documents (the *before* score).
2. Train: the loss is logged every 10 steps and should fall steadily. A flat or rising loss
   means the learning rate is too high or the labels are misformatted; `nan` means switch to
   `--fp32`.
3. Score the tuned model on the same 40 documents (the *after* score).

Optional: `python tools/legal/augment.py --n 150` writes Gemini-reworded training documents;
add `--extra data/legal/train_aug/sft.jsonl` to the train command.""",
    ),
    (
        "code",
        """!python tools/legal/finetune_agent1.py eval --out /kaggle/working/agent1_before.json""",
    ),
    (
        "code",
        """!python tools/legal/finetune_agent1.py train --adapter /kaggle/working/agent1_lora""",
    ),
    (
        "code",
        """!python tools/legal/finetune_agent1.py eval --adapter /kaggle/working/agent1_lora \\
    --out /kaggle/working/agent1_after.json
import json
before = json.load(open("/kaggle/working/agent1_before.json"))
after = json.load(open("/kaggle/working/agent1_after.json"))
for name in ("precision", "recall", "f1"):
    print(f"{name:9} before {before[name]:.3f}  after {after[name]:.3f}")
print("Done when: recall after > recall before ->", after["recall"] > before["recall"])""",
    ),
    (
        "markdown",
        """## Part 2 - the offline pipeline (steps 2-9)

Agent 1 = rules + the fine-tuned Gemma 4 on GPU 1; Agents 2-3 = Gemma 4 through Ollama;
leak check; SQLite store plus one JSON file per document. Re-run the cell after a restart: it
skips documents already in the store.""",
    ),
    (
        "code",
        """# Switch Agent 1 to the fine-tuned model in the legal profile (config/domains.json)
import json
cfg = json.load(open("config/domains.json"))
cfg["legal"].update(agent1_backend="transformers", agent1_adapter="/kaggle/working/agent1_lora")
json.dump(cfg, open("config/domains.json", "w"), indent=2)
os.environ["SANITISE_AGENT1_DEVICE"] = "cuda:1"
!python -m sanitise.legal build --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results""",
    ),
    (
        "markdown",
        """## Part 4 - online path, attacks and metrics (steps 10-13)

Hybrid search (bge-m3 in ChromaDB + BM25, reciprocal rank fusion k = 60, top 4) over the
originals; both pipelines answer with Gemini from the same four documents; every attack runs
with and without the session monitor; then the seven metrics, the agent-level scores and the
charts. Gemini answers are cached in `/kaggle/working/gemini_cache`, so a re-run after a rate
limit costs nothing.""",
    ),
    (
        "code",
        """!python -m sanitise.legal evaluate --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results
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
        """## Step 14 - export for the dashboard

Kaggle does not accept incoming connections, so download `bundle.zip`, unzip it on the laptop
and start the API with `SANITISE_REPLAY_DIR=<bundle>`; the dashboard then plays back this run
(every question, comparison and attack replays exactly). See docs/legal/demo_script.md.""",
    ),
    (
        "code",
        """!python -m sanitise.legal export --data data/legal --store /kaggle/working/store.sqlite \\
    --out /kaggle/working/results --bundle /kaggle/working/bundle
!cd /kaggle/working && zip -qr bundle.zip bundle && ls -lh bundle.zip""",
    ),
]


def notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks" / "legal_kaggle.ipynb"
    path.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
