"""Write notebooks/eval_health_kaggle.ipynb and notebooks/eval_legal_kaggle.ipynb.

    python tools/eval_suite/make_notebooks.py

The two notebooks run the same steps (tools/eval_suite/run.py); they differ only in set-up:
each runs its own package from one clone of main, pinned to the freeze tag.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = "eval-freeze-2"  # the clean test's code: change nothing after it has been run
REPO_URL = "https://github.com/raksha-sendhil/3rd_sem_el.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}  # fmt: skip


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}  # fmt: skip


INTRO = """
# Evaluation, {title} domain (Kaggle)

Runs every metric of `docs/evaluation_metrics.md` for the **{title}** domain: 5 single-question
metrics, 2 multi-question metrics, latency, and the end-to-end check through OCR. Each is
measured for **no privacy** (standard RAG) and **ours** (Agents 1-3 + the session monitor).
Step-by-step guide: `docs/evaluation_metrics.md`, section "Running it".

**Before running** (right sidebar):
* **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
* **Add-ons -> Secrets**: {secrets} Tick each as attached to this notebook.

**This is the clean test.** The code is frozen at the tag `eval-freeze-2`, the real data is
unseen and the people are new (`data/eval_clean/clean_people.json` on main). Run it once and
report what comes out.

**To let it run on its own** (recommended): set `SCALE` in cell 1, click **Save Version
-> Save & Run All (Commit) -> Save** (top right) and close Kaggle. Every cell runs in the
background; when it finishes, the version's **Output** tab holds `eval_results_{domain}_<SCALE>.zip`.

| Documents and questions per test | high | 5h | 2h | quick |
|---|---:|---:|---:|---:|
| Real documents indexed | 300 | 150 | 100 | 12 |
| Real documents held out | 150 | 100 | 60 | 6 |
| Generated records of new people, up to | 300 | 150 | 100 | 15 (seen) |
| Target Info questions | 200 | 100 | 60 | 5 |
| Repeat / ROUGE Prompt questions | 200 | 100 | 60 | 5 |
| Membership, members + non-members | 200 + 150 | 100 + 100 | 60 + 60 | 5 + 5 |
| Utility questions | 200 | 100 | 60 | 5 |
| Multi-question sessions (8 rounds) | 120 | 60 | 30 | 3 |
| Scans, end to end (as many as exist) | 50 | 25 | 20 | 2 |
| Gemini requests, about | 2,400 | 1,300 | 900 | 150 |
| Time, about (estimated) | 8 h | 4.5 h | 2.5 h | 10-20 min |

`SCALE = "quick"` only checks the set-up, on data development has seen: run it first. Kaggle
stops a session at 12 hours. If the build stops on the Gemini quota, add keys from other projects
and run again: every step skips what is already on disk in the same session.
"""


HEALTH_SETUP = """
# 1 Setup (run at the start of every session)
SCALE = "high"   # "high" (~8 h) | "5h" | "2h" | "quick" (set-up check on seen data, 10-20 min)
MODEL = "llama3:8b"                       # answers, both pipelines (as the dashboard)
GEMINI_MODEL = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"   # Agents 1-3 (fallback order)
OCR_MODEL = "qwen3-vl:8b-instruct"        # reads the scans (end-to-end check)
FREEZE = "eval-freeze-2"   # the frozen code: do not change it for the clean test

import os, subprocess, sys
from pathlib import Path

W = Path("/kaggle/working")
REPO = W / "3rd_sem_el"
os.environ["SANITISE_WORKDIR"] = "/tmp/sanitise"      # models stay out of /kaggle/working
os.environ["OLLAMA_NUM_PARALLEL"] = "4"   # answers are asked 2 questions x 2 pipelines at once
os.environ["EVAL_CONCURRENCY"] = "2"
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", FREEZE, "{repo_url}", str(REPO)], check=True)
os.chdir(REPO)
!git log --oneline -1
# the new people are data, so they may reach main after the code was frozen
PEOPLE = "data/eval_clean/clean_people.json"
PEOPLE_FILE = W / "clean_people.json"
if not PEOPLE_FILE.exists() and subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", "main"]).returncode == 0:
    found = subprocess.run(["git", "show", f"FETCH_HEAD:{PEOPLE}"], capture_output=True)
    if found.returncode == 0:
        PEOPLE_FILE.write_bytes(found.stdout)
!pip install -q -r requirements.txt
!pip install -q -e ".[retrieval]" --no-deps
!pip install -q "sentence-transformers>=3.0" "chromadb>=0.5" "google-genai>=1.0" matplotlib
sys.path.insert(0, str(REPO / "tools"))
import colab_cache
colab_cache.install_ollama_and_model(MODEL)
colab_cache.install_ollama_and_model(OCR_MODEL)
colab_cache.start_ollama()

from kaggle_secrets import UserSecretsClient
secrets, keys = UserSecretsClient(), []
for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]:
    try:
        keys += [k.strip() for k in secrets.get_secret(name).split(",") if k.strip()]
    except Exception:
        pass
keys = list(dict.fromkeys(keys))
if not keys:
    raise RuntimeError("Add the Kaggle secret GEMINI_API_KEY (Add-ons -> Secrets), tick it as "
                       "attached to this notebook, and run this cell again.")
key = ",".join(keys)
WORK = W / "eval" / ("health" if SCALE == "full" else f"health_{SCALE}")
os.environ.update({
    "SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL": MODEL, "SANITISE_LLM_TIMEOUT_S": "600",
    "SANITISE_RETRIEVAL_MODE": "hybrid", "SANITISE_OCR_MODEL": OCR_MODEL,
    "SANITISE_AGENTS_BACKEND": "gemini", "SANITISE_AGENTS_MODEL": GEMINI_MODEL,
    "SANITISE_GEMINI_API_KEY": key, "GEMINI_API_KEYS": key,
    "SANITISE_MONITOR_DB_PATH": str(WORK / "stores" / "monitor.sqlite"),
})
RUN = f"python tools/eval_suite/run.py {{step}} --domain health --work {WORK} --scale {SCALE} --people-file {PEOPLE_FILE}"
def run(step, extra=""):
    colab_cache.start_ollama()  # back up if it stopped (e.g. a cell was interrupted)
    cmd = RUN.format(step=step) + " " + extra
    print("$", cmd, flush=True)
    code = subprocess.run(cmd, shell=True, check=False).returncode
    if code and step in ("data", "build"):  # stop here: later numbers would be wrong
        raise RuntimeError(f"{step} failed: read the last lines above")
print(f"ready: answers {MODEL} | Agents 1-3 {GEMINI_MODEL} ({len(key.split(','))} key(s)) | "
      f"OCR {OCR_MODEL} | results in {WORK}")
"""

def legal_setup() -> str:
    """Health's set-up, running main's legal package (src/sanitise_legal) with its own profile."""
    setup = HEALTH_SETUP
    for old, new in (
        ('WORK = W / "eval" / ("health" if SCALE == "full" else f"health_{SCALE}")',
         'WORK = W / "eval" / ("legal" if SCALE == "full" else f"legal_{SCALE}")'),
        ("--domain health --work", "--domain legal --work"),
        ('"SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL"',
         '"SANITISE_DOMAIN": "legal", "SANITISE_GEMINI_RPM": str(10 * len(keys)),\n'
         '    "SANITISE_GEMINI_CACHE_DIR": str(W / "gemini_cache"),\n'
         '    "SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL"'),
    ):  # fmt: skip
        assert old in setup, old
        setup = setup.replace(old, new, 1)
    return setup


STEPS = {
    "data": """
# 2 Data: real public text downloaded now, the generated corpus copied, every question written
run("data")
""",
    "ocr": """
# 3 End to end, part 1: the 50 real scans read by OCR (main's step 0), cut into parts{repo_note}
run("e2e-prep"{repo_arg})
""",
    "label": """
# 4 End to end, part 2: Gemini marks each scan's private values and writes 2 questions after them
# and 2 about the content (saved in e2e/labels.json: open it and check a few, see the guide)
run("e2e-label")
""",
    "questions": """
# 4b Utility questions: harmless, about the content only, the same for both pipelines
run("utility-questions")
""",
    "build": """
# 5 Build: Agents 1-3 over the generated corpus, the real corpus and both end-to-end sets
# (long; re-run after a time limit: documents already in a store are skipped)
run("build")
""",
    "single": """
# 6 Single-question attacks: Target Info, Repeat / ROUGE Prompt, membership inference, utility
run("single")
""",
    "multi": """
# 7 Multi-question attacks: 180 sessions of 8 rounds, no privacy vs ours (with the monitor)
run("multi")
""",
    "e2e": """
# 8 End to end, part 3: the scans' questions on the gold-text store (A) and the OCR store (B)
run("e2e")
""",
    "report": """
# 9 Report: every metric, REPORT.md and the figures
run("report")
from IPython.display import Image, Markdown, display
display(Markdown((WORK / "results" / "REPORT.md").read_text()))
for fig in sorted((WORK / "results" / "figures").glob("*.png")):
    display(Image(str(fig)))
""",
    "pack": """
# 10 Pack the results to download (Output tab, or Save Version) and commit to the repo
import shutil, zipfile
zip_path = W / f"eval_results_{domain}{{'' if SCALE == 'full' else '_' + SCALE}}.zip"
keep = ["results", "answers", "data/targeted.json", "data/sessions.json", "data/repeat.json",
        "data/membership.json", "data/utility.json", "e2e/labels.json", "e2e/manifest.json"]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for item in keep:
        p = WORK / item
        for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []):
            if f.is_file():
                z.write(f, f"{domain}/" + str(f.relative_to(WORK)))
print(f"{{zip_path}}: {{zip_path.stat().st_size / 1e6:.1f}} MB")
print("Download it from the Output tab (or Save Version), then follow 'Saving the results' in the guide.")
""",
}


def build(domain: str) -> dict:
    title = "health" if domain == "health" else "legal"
    secrets = ("`GEMINI_API_KEY` (several keys, comma-separated, are used in turn when one runs "
               "out of daily requests)." if domain == "health" else
               "`GEMINI_API_KEY`, optionally `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_9` (used for "
               "Agents 1-3, the utility questions and labelling the scans).")  # fmt: skip
    intro = INTRO.format(title=title, domain=domain, secrets=secrets,
                         docs="1,000" if domain == "legal" else "900",
                         build="~4-6 h (Gemini rate limit)")  # fmt: skip
    if domain == "health":
        setup = HEALTH_SETUP
    else:
        setup = legal_setup()
    setup = setup.replace("{repo_url}", REPO_URL)
    cells = [md(intro), code(setup)]
    for name, src in STEPS.items():
        if name == "ocr":
            src = src.format(
                repo_note="",
                repo_arg="",
            )  # fmt: skip
        elif name == "pack":
            src = src.format(domain=domain)
        cells.append(code(src))
    return notebook(cells)


RERUN_INTRO = """
# Evaluation, health: rerun the utility questions on a finished run (Kaggle, ~1 hour)

The first utility questions asked about a record's *wording* (ChatDoctor questions quoted the
patient; scan questions named hospitals and section headings), which the rewrite changes, so
ours often answered "The records do not say" with the right record in hand. This notebook asks
new questions about the *content* (`tools/eval_suite/utility_questions.py`) on the stores your
5h run already built, and writes a new report. Nothing is rebuilt; every other number is the
same as in the 5h run.

**Before running:**
1. Session options: **GPU T4 x2**, Internet **On**; secret **GEMINI_API_KEY** attached.
2. Right sidebar → **Input** → **+ Add Input** → **Your Work** → pick the evaluation notebook
   whose 5h version finished → **Add**. Its output (the stores, answers and questions) appears
   under `/kaggle/input/`.
3. **Save Version → Save & Run All (Commit)**, then close Kaggle. When it finishes, the
   version's **Output** tab holds `eval_results_health_5h.zip` (the new full report).

| Cell | What it does | Time |
|---|---|---|
| 1 Setup | clone main, install, Ollama + llama3:8b (no vision model needed) | ~12 min |
| 2 Restore | copy the 5h run from the attached input | ~1 min |
| 3 New questions | Gemini rewrites 100 ChatDoctor and 50 scan questions about content | ~12 min |
| 4 Answer | the new utility questions, both pipelines, on the saved stores | ~15 min |
| 5 End to end | the scans' new questions on both scan stores | ~10 min |
| 6 Report + pack | the full report (all metrics) and the zip | ~3 min |
"""

RERUN_RESTORE = """
# 2 Restore the finished 5h run from the attached input (its stores, answers and questions)
import shutil
found = [p.parent for p in Path("/kaggle/input").rglob("stores") if p.parent.name == "health_5h"]
if not found:
    raise RuntimeError("No finished 5h run under /kaggle/input: add the evaluation notebook's "
                       "output as an input (right sidebar -> Input -> + Add Input -> Your Work).")
if not (WORK / "stores" / "real.sqlite").exists():
    shutil.copytree(found[0], WORK, dirs_exist_ok=True)
print("restored", found[0], "->", WORK, "| stores:", sorted(p.name for p in (WORK / "stores").iterdir()))
"""

RERUN_STEPS = [
    """
# 3 New utility questions about content (Gemini): ChatDoctor and the scans
run("utility-questions")
""",
    """
# 4 Answer the new ChatDoctor utility questions, both pipelines, on the saved real store
run("single", "--only utility")
""",
    """
# 5 Answer the scans' new utility questions on both scan stores (gold text and OCR)
run("e2e")
""",
]


def build_rerun() -> dict:
    setup = HEALTH_SETUP.replace("{repo_url}", REPO_URL)
    setup = setup.replace('SCALE = "5h"', 'SCALE = "5h"  # the run being redone; keep "5h"')
    setup = setup.replace("colab_cache.install_ollama_and_model(OCR_MODEL)\n", "")
    cells = [md(RERUN_INTRO), code(setup), code(RERUN_RESTORE)]
    cells += [code(c) for c in RERUN_STEPS]
    cells.append(code(STEPS["report"]))
    cells.append(code(STEPS["pack"].format(domain="health")))
    return notebook(cells)


FINANCE_INTRO = """
# Evaluation, finance domain (Kaggle): one source per run

Finance has four sources (docs/finance_plan.md), one run each: set `SOURCE` in cell 1. Runs on
different Kaggle accounts can go in parallel.

| SOURCE | Owner | Data (fetched into `data/finance/<source>/`) | Metrics |
|---|---|---|---|
| `bank` | A | CFPB complaints (real) + AgamiAI statements (population) | single-question + **multi-question** |
| `kyc` | B | Gretel documents (labelled) + NIST tax forms (scans) | single-question + end to end |
| `corporate` | C | Enron emails + TAT-DQA pages (real) + Kleister reports (scans) | single-question + end to end |
| `cheques` | C | IDRBT handwritten cheques (real scans) | end to end (handwriting OCR) |

**Before running:** GPU **T4 x2**, Internet **On**, secret **GEMINI_API_KEY** attached. For
`cheques`, also add the private Kaggle dataset with the IDRBT zip and the typed `labels.csv`
as an input and set `CHEQUES_ZIP`. Then **Save Version -> Save & Run All (Commit)** and close Kaggle; the version's
**Output** holds `eval_results_finance_<source>_<scale>.zip`.

Steps a source does not have are skipped (`cheques` has only scans; only `bank` has attack
sessions). `SCALE = "quick"` checks the set-up in ~10 minutes; `"high"` is the real run and takes
all the documents a source has, up to 300 + 150 held out and 50 scans (`"5h"`, `"2h"`: fewer).
"""


def build_finance() -> dict:
    setup = HEALTH_SETUP.replace("{repo_url}", REPO_URL)
    for old, new in (
        ('SCALE = "high"', 'SOURCE = "kyc"   # bank | kyc | corporate | cheques (your source)\n'
                         'CHEQUES_ZIP = ""  # cheques only: path of the IDRBT zip under /kaggle/input\n'
                         'SCALE = "high"'),
        ('WORK = W / "eval" / ("health" if SCALE == "full" else f"health_{SCALE}")',
         'WORK = W / "eval" / f"finance_{SOURCE}_{SCALE}"'),
        ("--domain health --work", "--domain finance --source {SOURCE} --work"),
    ):  # fmt: skip
        assert old in setup, old
        setup = setup.replace(old, new, 1)
    fetch = """
# 2 Data: the source's folder is in the repo (each owner commits theirs); cheques are fetched
# here from your private Kaggle dataset, since the images are never committed. The code stays
# frozen; the data is main's newest, so each owner's data goes to main.
subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", "main"], check=True)
subprocess.run(["git", "checkout", "FETCH_HEAD", "--", "data/finance/" + SOURCE], check=True)
if SOURCE == "cheques":
    import shutil
    # labels with the account numbers and dates stay in the owner's private Kaggle dataset
    for labels in sorted(Path("/kaggle/input").rglob("labels.csv"))[:1]:
        shutil.copy(labels, REPO / "data/finance/cheques/labels.csv")
    !python tools/finance/cheques.py --zip "{CHEQUES_ZIP}"
run("data")
"""
    cells = [md(FINANCE_INTRO), code(setup), code(fetch)]
    for step, note in (
        ("utility-questions", "2b Utility questions asked about several documents are rewritten after their own (Gemini)"),
        ("e2e-prep", "3 End to end: the source's scans through OCR (labels from the dataset's fields)"),
        ("build", "4 Build: the finance agents over the source's documents (resumes)"),
        ("single", "5 Single-question attacks and utility"),
        ("multi", "6 Multi-question attacks (bank only; skipped otherwise)"),
        ("e2e", "7 End to end: the scans' questions"),
    ):  # fmt: skip
        cells.append(code(f'# {note}\nrun("{step}")\n'))
    cells.append(code(STEPS["report"]))
    cells.append(code(STEPS["pack"].format(domain="finance").replace(
        'f"eval_results_finance', 'f"eval_results_finance_{SOURCE}').replace(
        'f"finance/" +', 'f"finance_{SOURCE}/" +')))  # fmt: skip
    return notebook(cells)


def main() -> None:
    for domain in ("health", "legal"):
        path = ROOT / "notebooks" / f"eval_{domain}_kaggle.ipynb"
        path.write_text(
            json.dumps(build(domain), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("wrote", path.relative_to(ROOT))
    path = ROOT / "notebooks" / "eval_finance_kaggle.ipynb"
    path.write_text(
        json.dumps(build_finance(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
