"""Fast setup for the advisor notebooks (notebooks/demo.ipynb, notebooks/scores.ipynb).

Standard library only: it runs before anything is installed.

Installs only what Agent 1 needs at run time, and does the two slow parts at the same
time: the Python packages (pip) on one thread, Ollama + the model download on another.
Everything is idempotent, so running it again in the same Colab session returns at once.

Optional ``cache``: a folder (e.g. on Google Drive, ~9 GB free needed) where the
installed packages, Ollama and the model are saved as archives after the first run and
restored from on later runs instead of being downloaded.

    import colab_cache
    colab_cache.setup(Path("/content/3rd_sem_el"), model="llama3:8b")
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# What the pipeline imports at run time (tests, CLI and API extras are not needed).
RUNTIME_PACKAGES = {
    "pydantic",
    "pydantic-settings",
    "httpx",
    "presidio-analyzer",
    "spacy",
    "tldextract",
    "en-core-web-lg",
}
CONTENT = Path("/content")
PYLIB = CONTENT / "pylib"
MODELS = CONTENT / "ollama_models"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_PARTS = ("usr/local/bin/ollama", "usr/local/lib/ollama")


def _say(msg: str) -> None:
    print(msg, flush=True)


def _package_name(requirement: str) -> str:
    name = requirement.split("@")[0].split("==")[0].split("[")[0].split(";")[0]
    return name.strip().lower().replace("_", "-")


def runtime_requirements(requirements_txt: Path) -> list[str]:
    """The pinned lines of requirements.txt that the pipeline needs at run time."""
    lines = [ln.strip() for ln in requirements_txt.read_text(encoding="utf-8").splitlines()]
    return [
        ln
        for ln in lines
        if ln and not ln.startswith("#") and _package_name(ln) in RUNTIME_PACKAGES
    ]


# ------------------------------------------------------------------ Python packages


def install_python(repo: Path) -> None:
    if not PYLIB.exists():
        _say("installing Python packages ...")
        reqs = runtime_requirements(repo / "requirements.txt")
        tmp = PYLIB.with_name("pylib.partial")
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--target", str(tmp), *reqs],
            check=True,
        )
        tmp.rename(PYLIB)
        _say("  Python packages ready")
    for path in (str(PYLIB), str(repo / "src")):  # repo code first, then its packages
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    importlib.invalidate_caches()


# ------------------------------------------------------------------ Ollama + model


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=2)
        return True
    except Exception:
        return False


def start_ollama() -> None:
    """Start the Ollama server if it is not running. Safe to call any time."""
    if ollama_up():
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OLLAMA_MODELS": str(MODELS), "OLLAMA_KEEP_ALIVE": "1h"}
    with open(CONTENT / "ollama.log", "a") as log:
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        if ollama_up():
            return
        time.sleep(1)
    raise RuntimeError((CONTENT / "ollama.log").read_text()[-2000:])


def model_present(model: str) -> bool:
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    wanted = model if ":" in model else f"{model}:latest"
    return any(line.split()[:1] == [wanted] for line in listing.splitlines())


def install_ollama_and_model(model: str) -> None:
    if shutil.which("ollama") is None:
        _say("installing Ollama ...")
        subprocess.run("apt-get -qq install -y zstd pciutils > /dev/null", shell=True, check=True)
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1", shell=True, check=True
        )
    start_ollama()
    if not model_present(model):
        _say(f"downloading {model} (4.7 GB) ...")
        pull = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
        if pull.returncode != 0:
            raise RuntimeError(f"ollama pull {model} failed:\n{pull.stderr[-1500:]}")
    _say(f"  {model} ready")


# ------------------------------------------------------------------ optional archive cache


def _restore(cache: Path, archives: dict[str, Path]) -> None:
    jobs = []
    if not PYLIB.exists() and archives["python"].exists():
        jobs.append(["tar", "-xf", str(archives["python"]), "-C", str(CONTENT)])
    if "ollama" in archives and shutil.which("ollama") is None and archives["ollama"].exists():
        jobs.append(["tar", "-xf", str(archives["ollama"]), "-C", "/"])
    if "model" in archives and archives["model"].exists() and not MODELS.exists():
        MODELS.mkdir(parents=True)
        jobs.append(["tar", "-xf", str(archives["model"]), "-C", str(MODELS)])
    if jobs:
        _say(f"restoring {len(jobs)} archive(s) from {cache} ...")
    for proc in [subprocess.Popen(cmd) for cmd in jobs]:
        proc.wait()  # a damaged archive just means the install step below does the work


def _save(archives: dict[str, Path]) -> None:
    sources = {
        "python": (str(CONTENT), ["pylib"]),
        "ollama": ("/", [p for p in OLLAMA_PARTS if (Path("/") / p).exists()]),
        "model": (str(MODELS), ["."]),
    }
    for key, target in archives.items():
        if target.exists():
            continue
        root, parts = sources[key]
        local = CONTENT / f"{key}.tar"
        try:
            subprocess.run(["tar", "-cf", str(local), "-C", root, *parts], check=True)
            _say(f"saving {target.name} ({local.stat().st_size / 1e9:.1f} GB) ...")
            shutil.copyfile(local, target.with_name(target.name + ".partial"))
            os.replace(target.with_name(target.name + ".partial"), target)
        except (OSError, subprocess.CalledProcessError) as exc:
            _say(f"  could not save {target.name}: {exc}")
        finally:
            local.unlink(missing_ok=True)


# ------------------------------------------------------------------ entry point


def setup(repo: Path, model: str | None = "llama3:8b", cache: Path | None = None) -> None:
    """Make the pipeline importable and, if ``model`` is given, servable by Ollama."""
    archives: dict[str, Path] = {}
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        tag = f"py{sys.version_info.major}{sys.version_info.minor}"
        archives["python"] = cache / f"pylib-{tag}.tar"
        if model:
            archives["ollama"] = cache / "ollama-bin.tar"
            archives["model"] = cache / f"model-{model.replace(':', '_').replace('/', '_')}.tar"
        _restore(cache, archives)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(install_python, repo)]
        if model:
            jobs.append(pool.submit(install_ollama_and_model, model))
        for job in jobs:
            job.result()  # re-raises the first failure

    if archives:
        _save(archives)
