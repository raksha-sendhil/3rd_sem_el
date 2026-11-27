"""Write notebooks/eval_finance_colab.ipynb: one finance evaluation split across machines.

The work folder (data, stores, answers) lives on Google Drive as one zip. Each session restores it,
runs its stage and saves it back after every step and every 10 minutes, so a session that Colab
stops loses at most 10 minutes; the next session (any account with the Drive folder) resumes.

    STAGE = "prep"       GPU, about 20 min: data, scans through OCR
    (build elsewhere)    a laptop or a CPU runtime: utility questions and the build (resumable)
    STAGE = "questions"  GPU, about 2 h: attacks, scan questions, report, results zip

    python tools/eval_suite/make_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

FREEZE = "eval-freeze-28"

SETUP = '''# 1 Setup (run first in every session; Runtime > Change runtime type > T4 GPU)
STAGE = "prep"        # "prep": data + OCR (about 20 min) | "questions": attacks + report (about 2 h)
SOURCE = "bank"       # bank | kyc | corporate | cheques
SCALE = "high"
DRIVE_DIR = "/content/drive/MyDrive/eval_finance"   # the shared folder that holds the work zip
FREEZE = "%s"   # the frozen code: do not change it
MODEL = "llama3:8b"                  # answers, both pipelines
OCR_MODEL = "qwen3-vl:8b-instruct"   # reads the scans

import os, shutil, sqlite3, subprocess, sys, threading, time, zipfile
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
REPO = Path("/content/3rd_sem_el")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", FREEZE,
                    "https://github.com/raksha-sendhil/3rd_sem_el.git", str(REPO)], check=True)
os.chdir(REPO)
!pip install -q -r requirements.txt
!pip install -q -e ".[retrieval]" --no-deps
!pip install -q "sentence-transformers>=3.0" "chromadb>=0.5" matplotlib
sys.path.insert(0, str(REPO / "tools"))
import colab_cache
colab_cache.install_ollama_and_model(OCR_MODEL if STAGE == "prep" else MODEL)
colab_cache.start_ollama()

NAME = f"finance_{SOURCE}_{SCALE}"
WORK = Path("/content/work") / NAME
SAVED = Path(DRIVE_DIR) / f"work_{NAME}.zip"
os.environ.update({
    "SANITISE_WORKDIR": "/content", "OLLAMA_NUM_PARALLEL": "4", "EVAL_CONCURRENCY": "2",
    "SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL": MODEL, "SANITISE_LLM_TIMEOUT_S": "600",
    "SANITISE_RETRIEVAL_MODE": "hybrid", "SANITISE_OCR_MODEL": OCR_MODEL,
    "SANITISE_MONITOR_DB_PATH": str(WORK / "stores" / "monitor.sqlite"),
})

def restore():
    if SAVED.exists() and not WORK.exists():
        with zipfile.ZipFile(SAVED) as z:
            z.extractall(WORK.parent)
        print(f"restored {SAVED} ({SAVED.stat().st_size / 1e6:.1f} MB)")
    WORK.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
def save():
    """The work folder to Drive as one zip; SQLite files through the backup API, so a store being
    written is copied consistently. Written to a temporary name, then renamed."""
    with _lock:
        Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)
        tmp = SAVED.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(WORK.rglob("*")):
                if not f.is_file() or "index_" in str(f.relative_to(WORK)):
                    continue  # the search index is rebuilt from the store
                arc = f"{NAME}/" + str(f.relative_to(WORK))
                if f.suffix == ".sqlite":
                    copy = Path("/tmp") / ("bk_" + f.name)
                    src, dst = sqlite3.connect(str(f)), sqlite3.connect(str(copy))
                    src.backup(dst); src.close(); dst.close()
                    z.write(copy, arc); copy.unlink()
                elif not f.name.endswith(("-wal", "-shm", "-journal")):
                    z.write(f, arc)
        os.replace(tmp, SAVED)
        print(f"  saved to Drive ({SAVED.stat().st_size / 1e6:.1f} MB)", flush=True)

def keep_saving(every_s=600):
    while True:
        time.sleep(every_s)
        try:
            save()
        except Exception as exc:
            print(f"  (save failed, tried again later: {exc})", flush=True)

restore()
threading.Thread(target=keep_saving, daemon=True).start()
RUN = f"python tools/eval_suite/run.py {{step}} --domain finance --source {SOURCE} --work {WORK} --scale {SCALE}"
def run(step):
    colab_cache.start_ollama()
    cmd = RUN.format(step=step)
    print("$", cmd, flush=True)
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    code = subprocess.run(["bash", "-o", "pipefail", "-c",
                           "(" + cmd + ") 2>&1 | tee -a '" + str(WORK / "results" / "run.log") + "'"],
                          check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    save()
    if code:
        print(f"!!! {step} failed (see above); its finished work is saved; run this cell again to resume")
    return code
print(f"stage {STAGE}: {NAME} on {FREEZE}; work folder {WORK}")
''' % FREEZE

PREP = '''# 2 Prep (STAGE = "prep"): data, then the scans through OCR. Saved to Drive after each step.
assert STAGE == "prep", "set STAGE = 'prep' in cell 1"
if run("data") == 0:
    run("e2e-prep")
print("Prep done. Next: the build on the laptop (or a CPU runtime), from the zip in", DRIVE_DIR)
'''

QUESTIONS = '''# 3 Questions (STAGE = "questions"): attacks, scan questions, report. Resumes where it stopped.
assert STAGE == "questions", "set STAGE = 'questions' in cell 1"
from huggingface_hub import snapshot_download
for attempt in range(6):  # the search model, before any question needs it
    try:
        snapshot_download("BAAI/bge-m3"); break
    except Exception as exc:
        print(f"embedding model download failed ({exc}); again in {20 * 2 ** attempt} s")
        time.sleep(20 * 2 ** attempt)
for step in ["single", "multi", "e2e", "report"]:
    run(step)
zip_path = Path(DRIVE_DIR) / f"eval_results_{NAME}.zip"
keep = ["results", "answers", "data/targeted.json", "data/sessions.json", "data/repeat.json",
        "data/membership.json", "data/utility.json", "pii/labels.json", "e2e/labels.json",
        "e2e/manifest.json", "stores/build.json"]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for item in keep:
        p = WORK / item
        for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []):
            if f.is_file():
                z.write(f, f"{NAME}/" + str(f.relative_to(WORK)))
print(f"results: {zip_path}")
from IPython.display import Markdown, display
try:
    display(Markdown((WORK / "results" / "REPORT.md").read_text()))
except Exception as exc:
    print(f"no report to show ({exc})")
'''

GUIDE = f"""# Finance evaluation across sessions (Colab + laptop)

The work folder is kept on Google Drive (`DRIVE_DIR`) as one zip and saved after every step and
every 10 minutes, so a stopped session loses at most 10 minutes. Code: tag `{FREEZE}`.

1. **Prep** (Colab, T4 GPU, about 20 min): `STAGE = "prep"`, run cells 1 and 2.
2. **Build** (laptop, no GPU): download the zip from Drive, run the utility-questions and build
   steps (see the guide), zip the folder again and put it back in the Drive folder.
3. **Questions** (Colab, T4 GPU, about 2 h): `STAGE = "questions"`, run cells 1 and 3. The results
   zip appears in the Drive folder.

If Colab stops a session, start a new one (any account that has the Drive folder) and run the same
cells again: everything finished is restored and skipped.
"""

KAGGLE_SETUP = """# 1 Setup. The work folder comes in as a Kaggle dataset: Add Input > your dataset with
# work_finance_bank_high.zip (made on the laptop after the build). No Gemini key is needed.
SOURCE = "bank"
SCALE = "high"
HOURS = 2.5          # questions stop 10 min before this; then the report has its own 20 min
MODEL = "llama3:8b"
FREEZE = "%s"

import glob, os, subprocess, sys, time, zipfile
from pathlib import Path
START = time.time()
W = Path("/kaggle/working")
REPO = W / "3rd_sem_el"
NAME = f"finance_{SOURCE}_{SCALE}"
WORK = W / "eval" / NAME
import shutil
found = glob.glob(f"/kaggle/input/**/work_{NAME}.zip", recursive=True)
unzipped = [p for p in glob.glob(f"/kaggle/input/**/{NAME}", recursive=True) if Path(p, "stores").is_dir()]
if not WORK.exists():
    if found:
        with zipfile.ZipFile(found[0]) as z:
            z.extractall(WORK.parent)
        print("work folder restored from", found[0])
    elif unzipped:  # Kaggle unzips an uploaded zip into a folder
        shutil.copytree(unzipped[0], WORK)
        print("work folder copied from", unzipped[0])
    else:
        raise RuntimeError(f"add the dataset with work_{NAME}.zip as an input (Add Input), then run again")
print("stores:", sorted(p.name for p in (WORK / "stores").glob("*.sqlite")))
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", FREEZE,
                    "https://github.com/raksha-sendhil/3rd_sem_el.git", str(REPO)], check=True)
os.chdir(REPO)
os.environ["SANITISE_WORKDIR"] = "/tmp/sanitise"
!pip install -q -r requirements.txt
!pip install -q -e ".[retrieval]" --no-deps
!pip install -q "sentence-transformers>=3.0" "chromadb>=0.5" matplotlib
sys.path.insert(0, str(REPO / "tools"))
import colab_cache
colab_cache.install_ollama_and_model(MODEL)
colab_cache.start_ollama()
from huggingface_hub import snapshot_download
for attempt in range(6):  # the search model, before any question needs it
    try:
        snapshot_download("BAAI/bge-m3"); break
    except Exception as exc:
        print(f"embedding model download failed ({exc}); again in {20 * 2 ** attempt} s")
        time.sleep(20 * 2 ** attempt)
os.environ.update({
    "OLLAMA_NUM_PARALLEL": "4", "EVAL_CONCURRENCY": "2",
    "SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL": MODEL, "SANITISE_LLM_TIMEOUT_S": "600",
    "SANITISE_RETRIEVAL_MODE": "hybrid",
    "SANITISE_MONITOR_DB_PATH": str(WORK / "stores" / "monitor.sqlite"),
})
DEADLINE = time.time() + HOURS * 3600   # counted from here, after the set-up
os.environ["EVAL_DEADLINE"] = str(DEADLINE)
print(f"set-up took {(time.time() - START) / 60:.0f} min; questions stop by "
      f"{time.strftime('%%H:%%M', time.gmtime(DEADLINE - 600))} UTC, report by "
      f"{time.strftime('%%H:%%M', time.gmtime(DEADLINE + 1200))} UTC")
RUN = f"python tools/eval_suite/run.py {{step}} --domain finance --source {SOURCE} --work {WORK} --scale {SCALE}"
def run(step, limit_s):
    colab_cache.start_ollama()
    cmd = RUN.format(step=step)
    print("$", cmd, f"(at most {limit_s / 60:.0f} min)", flush=True)
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    try:  # a hard stop: the step is ended at its limit whatever it is doing
        subprocess.run(["bash", "-o", "pipefail", "-c",
                        "(" + cmd + ") 2>&1 | tee -a '" + str(WORK / "results" / "run.log") + "'"],
                       check=False, timeout=max(60, limit_s), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except subprocess.TimeoutExpired:
        print(f"!!! {step} stopped at its time limit; what it finished is kept", flush=True)
""" % FREEZE

KAGGLE_QUESTIONS = """# 2 Questions and report. Every step resumes and stops on time; the report always runs.
for step in ["single", "multi", "e2e"]:
    run(step, DEADLINE - time.time())
run("report", 20 * 60)
"""

KAGGLE_PACK = """# 3 Pack the results (Output tab)
zip_path = W / f"eval_results_{NAME}.zip"
keep = ["results", "answers", "data/targeted.json", "data/sessions.json", "data/repeat.json",
        "data/membership.json", "data/utility.json", "pii/labels.json", "e2e/labels.json",
        "e2e/manifest.json", "stores/build.json"]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for item in keep:
        p = WORK / item
        for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []):
            if f.is_file():
                z.write(f, f"{NAME}/" + str(f.relative_to(WORK)))
print(f"{zip_path}: {zip_path.stat().st_size / 1e6:.1f} MB; whole session {(time.time() - START) / 3600:.2f} h")
"""


def cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def main() -> None:
    nb = {
        "cells": [cell("markdown", GUIDE), cell("code", SETUP), cell("code", PREP), cell("code", QUESTIONS)],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    out = Path(__file__).resolve().parents[2] / "notebooks" / "eval_finance_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    intro = ("# Finance questions on Kaggle (after the build)\n\nAdd the work zip as an input, set the "
             "accelerator to GPU T4, then Save Version > Save & Run All. Questions stop by `HOURS` after "
             "the set-up; the report gets 20 minutes after that.")
    kg = {
        "cells": [cell("markdown", intro), cell("code", KAGGLE_SETUP),
                  cell("code", KAGGLE_QUESTIONS), cell("code", KAGGLE_PACK)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    out2 = out.parent / "eval_finance_questions_kaggle.ipynb"
    out2.write_text(json.dumps(kg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
