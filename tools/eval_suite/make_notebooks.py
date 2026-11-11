"""Write notebooks/eval_health_kaggle.ipynb and notebooks/eval_legal_kaggle.ipynb.

    python tools/eval_suite/make_notebooks.py

The two notebooks run the same steps (tools/eval_suite/run.py); they differ only in set-up:
health runs main's healthcare pipeline, legal runs the legal branch's pipeline, with this suite
and main's OCR taken from a clone of main.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
* **Session options**: Accelerator **GPU T4 x2**, Internet **On**, Persistence **Files only**
  (so `/kaggle/working` survives between sessions and every step resumes).
* **Add-ons -> Secrets**: {secrets} Tick each as attached to this notebook.

| Cell | What it does | Time, full scale |
|---|---|---|
| 1 Setup | clone, install, Ollama and the models, keys | ~15 min (every session) |
| 2 Data | download the real data, copy the generated corpus, write the questions | ~5 min |
| 3 OCR | read the 50 real scans (end-to-end check) | ~30-60 min |
| 4 Labels | Gemini marks private values and writes questions for the 50 scans | ~10 min |
| 5 Build | Agents 1-3 over every document (~{docs} documents) | {build} |
| 6 Single | the single-question attacks and utility questions (~1,500 answers per pipeline) | ~2-3 h |
| 7 Multi | 180 attack sessions x 8 rounds | ~2-3 h |
| 8 End to end | the 50 scans' questions on both stores | ~15 min |
| 9 Report | every number, REPORT.md and the figures | ~5 min |
| 10 Pack | `eval_results_{domain}.zip` to download and commit | ~1 min |

The whole run does not fit in one 12-hour session. Run cell 1 at the start of every session,
then carry on from the first unfinished cell: each one skips what is already on disk.
First run with `SCALE = "quick"` (a few items per step, ~30 min) to check everything works, then
set `SCALE = "full"` and run again (quick results are kept apart).
"""


HEALTH_SETUP = """
# 1 Setup (run at the start of every session)
SCALE = "quick"   # "quick" = a few items per step to check the set-up; then "full"
MODEL = "llama3:8b"                       # answers, both pipelines (as the dashboard)
GEMINI_MODEL = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"   # Agents 1-3 (fallback order)
OCR_MODEL = "qwen3-vl:8b-instruct"        # reads the scans (end-to-end check)
BRANCH = "main"

import os, subprocess, sys
from pathlib import Path

W = Path("/kaggle/working")
REPO = W / "3rd_sem_el"
os.environ["SANITISE_WORKDIR"] = "/tmp/sanitise"      # models stay out of /kaggle/working
if REPO.exists():
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, "{repo_url}", str(REPO)], check=True)
os.chdir(REPO)
!git log --oneline -1
!pip install -q -r requirements.txt
!pip install -q -e ".[retrieval]" --no-deps
!pip install -q "sentence-transformers>=3.0" "chromadb>=0.5" matplotlib
sys.path.insert(0, str(REPO / "tools"))
import colab_cache
colab_cache.install_ollama_and_model(MODEL)
colab_cache.install_ollama_and_model(OCR_MODEL)
colab_cache.start_ollama()

from kaggle_secrets import UserSecretsClient
try:
    key = UserSecretsClient().get_secret("GEMINI_API_KEY")
except Exception as exc:
    raise RuntimeError("Add the Kaggle secret GEMINI_API_KEY (Add-ons -> Secrets), tick it as "
                       "attached to this notebook, and run this cell again.") from exc
WORK = W / "eval" / ("health" if SCALE == "full" else "health_quick")
os.environ.update({
    "SANITISE_LLM_BACKEND": "ollama", "SANITISE_MODEL": MODEL, "SANITISE_LLM_TIMEOUT_S": "600",
    "SANITISE_RETRIEVAL_MODE": "hybrid", "SANITISE_OCR_MODEL": OCR_MODEL,
    "SANITISE_AGENTS_BACKEND": "gemini", "SANITISE_AGENTS_MODEL": GEMINI_MODEL,
    "SANITISE_GEMINI_API_KEY": key, "GEMINI_API_KEYS": key,
    "SANITISE_MONITOR_DB_PATH": str(WORK / "stores" / "monitor.sqlite"),
})
RUN = f"python tools/eval_suite/run.py {{step}} --domain health --work {WORK} --scale {SCALE}"
def run(step, extra=""):
    cmd = RUN.format(step=step) + " " + extra
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)
print(f"ready: answers {MODEL} | Agents 1-3 {GEMINI_MODEL} ({len(key.split(','))} key(s)) | "
      f"OCR {OCR_MODEL} | results in {WORK}")
"""

LEGAL_SETUP = """
# 1 Setup (run at the start of every session)
SCALE = "quick"   # "quick" = a few items per step to check the set-up; then "full"
LEGAL_BRANCH = "law-pipeline-final"   # the branch with the legal agents
ANSWERS = "ollama"   # "ollama": the legal profile's Ollama model answers (fits the time);
                     # "gemini": the profile's Gemini answer model (needs many keys and days)
AGENT1_ADAPTER = ""  # optional: path of a fine-tuned Agent 1 LoRA adapter (else the base model)
OCR_MODEL = "qwen3-vl:8b-instruct"    # reads the scans (end-to-end check; main's OCR)

import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

W = Path("/kaggle/working")
MAIN = W / "main"      # this evaluation suite and the OCR (step 0) come from main
LEGAL = W / "legal"    # the legal pipeline (Agents 1-3, monitor, scorer) comes from the branch
for path, branch in ((MAIN, "main"), (LEGAL, LEGAL_BRANCH)):
    if path.exists():
        subprocess.run(["git", "-C", str(path), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", branch, "{repo_url}", str(path)], check=True)
!git -C /kaggle/working/main log --oneline -1
!git -C /kaggle/working/legal log --oneline -1
os.chdir(LEGAL)
!pip install -q -r /kaggle/working/main/requirements.txt
!pip install -q -U transformers accelerate peft bitsandbytes google-genai presidio-analyzer sentence-transformers chromadb rank_bm25 rapidfuzz rouge-score matplotlib
!pip install -q -e /kaggle/working/legal
!python -m spacy download en_core_web_lg -q

# Ollama on both GPUs (on GPU 0 only when a fine-tuned Agent 1 takes GPU 1)
!apt-get update -qq && apt-get install -y -qq zstd
!which ollama || (curl -fsSL https://ollama.com/install.sh | sh)
def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2); return True
    except Exception:
        return False
if not ollama_up():
    env = {{**os.environ, **({{"CUDA_VISIBLE_DEVICES": "0"}} if AGENT1_ADAPTER else {{}})}}
    subprocess.Popen(["ollama", "serve"], stdout=open(W / "ollama.log", "a"),
                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
    for _ in range(30):
        if ollama_up(): break
        time.sleep(2)
profile = json.load(open(LEGAL / "config/domains.json"))
subprocess.run(["ollama", "pull", profile["legal"]["model"]], check=True)
subprocess.run(["ollama", "pull", OCR_MODEL], check=True)

# keys from the Kaggle secrets, as the legal notebook reads them
from kaggle_secrets import UserSecretsClient
secrets, keys = UserSecretsClient(), []
for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{{i}}" for i in range(2, 10)]:
    try:
        keys.append(secrets.get_secret(name))
    except Exception:
        pass
if not keys:
    raise RuntimeError("Add the Kaggle secret GEMINI_API_KEY (Add-ons -> Secrets) and attach it.")

# the legal profile: answers on Ollama unless ANSWERS = "gemini"; optional fine-tuned Agent 1
if ANSWERS == "ollama":
    profile["legal"]["answer_backend"] = "same"
if AGENT1_ADAPTER:
    profile["legal"].update(agent1_backend="transformers", agent1_adapter=AGENT1_ADAPTER)
    os.environ["SANITISE_AGENT1_DEVICE"] = "cuda:1"
json.dump(profile, open(LEGAL / "config/domains.json", "w"), indent=2)
print(json.dumps(profile["legal"], indent=1))

WORK = W / "eval" / ("legal" if SCALE == "full" else "legal_quick")
os.environ.update({
    "SANITISE_DOMAIN": "legal", "SANITISE_LLM_BACKEND": "ollama", "SANITISE_LLM_TIMEOUT_S": "300",
    "SANITISE_RETRIEVAL_MODE": "hybrid", "SANITISE_OCR_MODEL": OCR_MODEL,
    "GEMINI_API_KEYS": ",".join(keys), "SANITISE_GEMINI_CACHE_DIR": str(W / "gemini_cache"),
    "SANITISE_GEMINI_RPM": "10", "SANITISE_GEMINI_KEY_COOLDOWN_S": "60",
})
SUITE = MAIN / "tools/eval_suite/run.py"
def run(step, extra="", repo=LEGAL):
    cmd = (f"python {{SUITE}} {{step}} --domain legal --work {{WORK}} --scale {{SCALE}} "
           f"--repo {{repo}} --main-repo {{MAIN}} {{extra}}")
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)
print(f"ready: legal pipeline {{LEGAL_BRANCH}} | {{len(keys)}} Gemini key(s) | OCR {{OCR_MODEL}} | "
      f"results in {{WORK}}")
"""

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
zip_path = W / f"eval_results_{domain}{{'' if SCALE == 'full' else '_quick'}}.zip"
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
               "labelling the scans, and by the legal profile wherever it calls Gemini).")  # fmt: skip
    intro = INTRO.format(title=title, domain=domain, secrets=secrets,
                         docs="1,000" if domain == "legal" else "900",
                         build="~4-6 h (Gemini rate limit)" if domain == "health"
                         else "~5-7 h (Gemma on Ollama)")  # fmt: skip
    if domain == "health":
        setup = HEALTH_SETUP
    else:  # written with doubled braces; plain Python once undoubled
        setup = LEGAL_SETUP.replace("{{", "{").replace("}}", "}")
    setup = setup.replace("{repo_url}", REPO_URL)
    cells = [md(intro), code(setup)]
    for name, src in STEPS.items():
        if name == "ocr":
            src = src.format(
                repo_note="" if domain == "health" else "\n# (main's code reads the scans; the legal agents take the parts in cell 5)",
                repo_arg="" if domain == "health" else ", repo=MAIN",
            )  # fmt: skip
        elif name == "pack":
            src = src.format(domain=domain)
        cells.append(code(src))
    return notebook(cells)


def main() -> None:
    for domain in ("health", "legal"):
        path = ROOT / "notebooks" / f"eval_{domain}_kaggle.ipynb"
        path.write_text(
            json.dumps(build(domain), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
