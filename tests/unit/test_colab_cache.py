"""tools/colab_cache.py: the parts that do not need Colab."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import colab_cache  # noqa: E402


def test_runtime_requirements_keep_only_what_the_pipeline_imports() -> None:
    reqs = colab_cache.runtime_requirements(REPO / "requirements.txt")
    names = {colab_cache._package_name(r) for r in reqs}
    assert names == colab_cache.RUNTIME_PACKAGES
    assert all("==" in r or " @ " in r for r in reqs)  # still pinned


def test_install_python_skips_pip_when_present_and_puts_repo_code_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pylib = tmp_path / "pylib"
    pylib.mkdir()
    monkeypatch.setattr(colab_cache, "PYLIB", pylib)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def no_pip(*args, **kwargs):
        raise AssertionError("pip must not run when the packages are already installed")

    monkeypatch.setattr(colab_cache.subprocess, "run", no_pip)
    colab_cache.install_python(tmp_path / "repo")
    assert sys.path[:2] == [str(tmp_path / "repo" / "src"), str(pylib)]


def test_setup_without_a_model_never_touches_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(colab_cache, "install_python", lambda repo: calls.append("python"))
    monkeypatch.setattr(colab_cache, "install_ollama_and_model", lambda model: calls.append(model))
    colab_cache.setup(tmp_path, model=None)
    assert calls == ["python"]
    colab_cache.setup(tmp_path, model="llama3:8b")
    assert sorted(calls[1:]) == ["llama3:8b", "python"]
