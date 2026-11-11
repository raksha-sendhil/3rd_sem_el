"""Write notebooks/legal_agents_kaggle.ipynb: Agents 1-3 only, on real legal documents and on
your own files.

OWNER: Person C. No store, search, answers, attacks, session monitor, fine-tuning or dashboard:
those belong to the full notebook (notebooks/legal_kaggle.ipynb) and to other team members.
Edit the CELLS list below and run ``python tools/legal/make_agents_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/raksha-sendhil/3rd_sem_el.git"
BRANCH = "law-pipeline-final"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Legal privacy agents on real documents (Agents 1-3 only)

This notebook runs only the three agents and shows what each one does:

| Agent | What it does | What you see |
|---|---|---|
| 1 Detect | rules (regex, Presidio, spaCy) + a model pass find names, addresses, IDs, case numbers, money... | the original with each span coloured by category |
| 2 Extract | turns the document into facts (`subject - relation - value`, with importance) | a facts table |
| 3 Rewrite | keeps, generalises or replaces each span, rewrites the text, then runs a leak check | the safe text, and what happened to each span |

**The documents** (`DOC_SET` in cell 1):

- `"indian300"`: 300 real Indian judgments and orders from the official open datasets of the
  Supreme Court (90, from 2023-2025) and nine High Courts (Karnataka, Delhi, Bombay, Kerala,
  Madras, Calcutta, Telangana, Gujarat, Gauhati). Each is the start of the judgment, at most
  1,800 characters, so it goes through the agents in one part. They were picked and checked by
  `tools/legal/build_indian_corpus.py`.
- `"curated"`: 14 hand-picked documents (12 Indian, 1 UK, 1 US) with checks written by hand.
- `"both"`: all 314.

**What counts as a good result.** Each document lists identifiers that must disappear
(`must_hide`) and legal substance that must survive (`must_keep`). For the 300, the names come
from the court's own case metadata (petitioner and respondent), not from our agents, together
with phone numbers, PAN, vehicle numbers and S/o names found by pattern. The legal terms are the
sections, Acts and words such as "bail" found in the text. The notebook reports both, first for
the rules alone (a baseline that needs no model) and then for the full agents.

**Kaggle settings** (right-hand panel): Accelerator **GPU T4 x2** (one GPU is used), Internet
**on**. Run the cells from top to bottom. Step-by-step help: `docs/legal/agents_notebook.md`.

| Step | Time |
|---|---|
| Setup: code, packages, Gemma 4 through Ollama | ~10 min |
| Rules-only baseline on 300 documents | ~2 min |
| Agents 1-3 with Gemma 4, 30 documents (`LIMIT = 30`) | ~30-45 min on a T4 |
| Agents 1-3 with Gemma 4, all 300 | ~5-7 h: one session, or two (finished documents are kept) |
| Your own documents | ~1-2 min per 1,800 characters |""",
    ),
    (
        "code",
        """# 1. Settings: the only cell you need to edit
BACKEND = "ollama"        # "ollama": Gemma 4 on this notebook's GPU; nothing leaves the notebook
                          # "gemini": Google's API (needs a GEMINI_API_KEY secret); the original
                          #           text is sent to Google, so use it for public documents only
                          # "mock":   rules only, no model (the baseline; runs in seconds)
MODEL = None              # None = the legal profile's model (gemma4:e4b / gemini-3.5-flash)
DOC_SET = "indian300"     # "indian300" (300 Indian judgments), "curated" (14), or "both"
LIMIT = 30                # first N documents for a trial run; None = the whole set
MAX_CHARS = 1800          # longer documents are split into parts of at most this many characters
RUN_REAL_DOCS = True      # False = skip the set and run only your own documents
ONLY = None               # e.g. ["in08_kar_hc_motor_accident"]: just these ids
OUT = "/kaggle/working/agent_runs"   # one JSON per document; re-running skips finished ones""",
    ),
    (
        "code",
        f"""# 2. The code and its packages
!git clone -q -b {BRANCH} {REPO} /kaggle/working/repo 2>/dev/null || (cd /kaggle/working/repo && git pull -q)
%cd /kaggle/working/repo
!pip install -q -e . pypdf python-docx google-genai
!python -m spacy download en_core_web_lg -q 2>/dev/null || python -m spacy download en_core_web_lg
!git log -1 --format="code version: %h %s\"""",
    ),
    (
        "code",
        """# 3. The model
import os, subprocess, time, urllib.request
os.environ["SANITISE_DOMAIN"] = "legal"
os.environ["SANITISE_GEMINI_CACHE_DIR"] = "/kaggle/working/gemini_cache"

def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

if BACKEND == "ollama":
    if not ollama_up():
        subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        log = open("/kaggle/working/ollama.log", "a")
        # Kaggle refuses "!ollama serve &", so start it from Python; it survives pressing Stop
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, start_new_session=True)
        for _ in range(60):
            if ollama_up():
                break
            time.sleep(2)
    subprocess.run(["ollama", "pull", MODEL or "gemma4:e4b"], check=True)
    print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
elif BACKEND == "gemini":
    from kaggle_secrets import UserSecretsClient
    secrets, keys = UserSecretsClient(), []
    for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]:
        try:
            keys.append(secrets.get_secret(name))
        except Exception:
            pass  # that secret is not attached to this notebook
    os.environ["GEMINI_API_KEYS"] = ",".join(keys)
    os.environ["SANITISE_GEMINI_RPM"] = "10"
    print(f"{len(keys)} Gemini key(s) loaded" if keys else "No key: add GEMINI_API_KEY under Add-ons -> Secrets")
else:
    print("mock: rules only, no model")""",
    ),
    (
        "code",
        """# 4. The real documents
import asyncio, json
import pandas as pd
from IPython.display import HTML, Markdown, display
from sanitise.config import get_settings
get_settings.cache_clear()  # pick up the settings above, also after re-running them
from sanitise.legal.real_docs import (
    from_text, load_inputs, load_set, render_html, run_agents, summary_rows, totals,
    write_report,
)
pd.set_option("display.max_colwidth", 80)

real = load_set(DOC_SET)
if ONLY:
    real = [d for d in real if d.id in ONLY]
everything_in_set = real
real = real[:LIMIT] if LIMIT else real
table = pd.DataFrame([{
    "id": d.id, "court": d.meta["court"], "type": d.meta["doc_type"], "date": d.meta["date"],
    "chars": len(d.text), "must_hide": len(d.must_hide), "must_keep": len(d.must_keep),
} for d in everything_in_set])
print(f"{len(everything_in_set)} documents in the set; running {len(real)} (LIMIT = {LIMIT})")
display(table["court"].value_counts().rename("documents per court"))
display(table.head(20))""",
    ),
    (
        "markdown",
        """## Baseline: the rules alone

Agent 1's rule pass, a fixed rewrite that applies the placeholders and generalisation ladders,
and no model. This is what the agents have to beat.""",
    ),
    (
        "code",
        """baseline = await run_agents(real, backend="mock", out_dir=f"{OUT}/rules_only",
                            max_chars=MAX_CHARS, progress=None)
display(pd.Series(totals(baseline), name="rules only"))""",
    ),
    (
        "markdown",
        """## Agents 1-3 with the model

Each document goes through Agent 1 (rules + model + merge), Agent 2 and Agent 3 (rewrite + leak
check), part by part. One line is printed per document. If the session stops, run this cell
again: finished documents are loaded from `OUT`, not run again.""",
    ),
    (
        "code",
        """results = []
if RUN_REAL_DOCS and BACKEND != "mock":
    results = await run_agents(real, backend=BACKEND, model=MODEL, max_chars=MAX_CHARS,
                               out_dir=f"{OUT}/{BACKEND}")
    display(pd.DataFrame({"rules only": totals(baseline), "agents": totals(results)}))
    display(pd.DataFrame(summary_rows(results)).set_index("document"))""",
    ),
    (
        "markdown",
        """### How to read the table

- **must-hide hidden**: identifiers removed from the safe text, out of those listed for that
  document. **leaked** names the ones still there. Each should be 100%.
- **must-keep kept**: legal terms still in the safe text. **dropped** names the lost ones. A
  dropped term means the rewrite lost substance a lawyer would search for.
- **direct ids left**: names, phones, IDs, case numbers and addresses that Agent 1 found but
  Agent 3 left in. This works for your own documents too, which have no must-hide list.
- **fallbacks**: a step whose model output was broken and that fell back to rules
  (`agent1_rules_only`, `agent2_rules_only`, `agent3_in_place`). A few are normal; many mean the
  model is struggling (try a smaller `MAX_CHARS`).""",
    ),
    (
        "code",
        """# 5. Look at one document in full: pick any id from the table above
shown = {r.doc.id: r for r in (results or baseline)}
DOC = next(iter(shown))   # or any id from the table, e.g. "in08_kar_hc_motor_accident"
display(HTML(render_html(shown[DOC])))""",
    ),
    (
        "code",
        """# The documents that did worst: most must-hide leaks, then most dropped terms
def badness(r):
    c = r.checks()
    return (not r.ok, len(c["leaked"]) + len(c["direct_left"]), len(c["dropped"]))

SHOW = 10   # how many to show; set to len(results) to see every document
for r in sorted(results or baseline, key=badness, reverse=True)[:SHOW]:
    display(HTML(render_html(r) + "<hr>"))""",
    ),
    (
        "markdown",
        """## Your own documents

Three ways in; use any of them and run the next cell.

1. **Upload a folder as a Kaggle dataset** (best for several files): on the right, *Add Input*
   -> *Upload* -> drop your files -> name the dataset -> *Create*. It appears under
   `/kaggle/input/<dataset-name>/`. Put that path in `MY_INPUTS`.
2. **Upload single files into the notebook**: run `!mkdir -p /kaggle/working/my_inputs`, then
   use the file browser on the right (*Output* -> `/kaggle/working/my_inputs` -> upload), or
   `!wget -P /kaggle/working/my_inputs <url>` for a public link.
3. **Paste text** into `PASTED` below.

Supported: `.txt`, `.md`, `.pdf` (with a text layer; scanned PDFs need OCR first), `.docx`, and
`.json` with a `text` field. A `.json` may also give `title`, `must_hide` and `must_keep` lists to
get the same spot checks as the public documents (see `examples/legal_inputs/`). Folders are
searched recursively.

With `BACKEND = "gemini"` your text goes to Google; keep private client documents on
`"ollama"`.""",
    ),
    (
        "code",
        """MY_INPUTS = [
    "examples/legal_inputs",            # two fictional samples in the repo; replace or remove
    # "/kaggle/input/my-legal-docs",    # a Kaggle dataset you uploaded
    # "/kaggle/working/my_inputs",      # files uploaded into the notebook
]
PASTED = \"\"\"
\"\"\"   # paste a document between the quotes, or leave it empty

mine = load_inputs(*MY_INPUTS)
if PASTED.strip():
    mine.append(from_text(PASTED, id="pasted"))
print(f"{len(mine)} document(s):", [d.id for d in mine])

own = await run_agents(mine, backend=BACKEND, model=MODEL, max_chars=MAX_CHARS,
                       out_dir=f"{OUT}/my_inputs/{BACKEND}")
display(pd.DataFrame(summary_rows(own)).set_index("document"))
for r in own:
    display(HTML(render_html(r) + "<hr>"))""",
    ),
    (
        "markdown",
        """## Save the results

`REPORT.md` has the totals, the table and every safe text; each document's full output (spans,
facts, routing, safe text, timings) is in its JSON file. Download `agent_runs.zip` from the
*Output* panel, or click **Save Version** to keep everything with the notebook.""",
    ),
    (
        "code",
        """everything = (results or baseline) + (own if "own" in dir() else [])
report = write_report(everything, f"{OUT}/REPORT.md")
display(Markdown(open(report, encoding="utf-8").read()[:4000]))
!cd /kaggle/working && zip -qr agent_runs.zip agent_runs && ls -lh agent_runs.zip""",
    ),
]


def notebook(cells: list[tuple[str, str]] = CELLS) -> dict:
    out = []
    for kind, source in cells:
        lines = source.split("\n")
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks" / "legal_agents_kaggle.ipynb"
    path.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
