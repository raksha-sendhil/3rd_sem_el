"""The check/preflight commands, the in-place rewrite's grammar, and source hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sanitise_legal.agents.rewriter.pipeline import _fit
from sanitise_legal.rules.roles import _LAWYER

REPO = Path(__file__).resolve().parents[2]


def test_no_control_characters_in_the_source() -> None:
    """A lost escape once left a backspace in a regex (roles._LAWYER), which never matched."""
    bad = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    offenders = [
        str(p.relative_to(REPO))
        for folder in ("src", "tests", "tools", "eval")
        for p in (REPO / folder).rglob("*.py")
        if bad.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders


@pytest.mark.parametrize(
    ("before", "replacement", "fitted"),
    [
        ("listed before the ", "a magistrate's court", "listed before "),
        ("dispute with M/s ", "a trading business", "dispute with "),
        ("meeting us on ", "February 2025", "meeting us in "),
        ("Dear Mr. ", "[CLIENT_1]", "Dear Mr. "),
        ("an idea ", "the 2020s", "an idea "),
    ],
)
def test_generalised_phrases_fit_the_sentence(before: str, replacement: str, fitted: str) -> None:
    assert _fit(before, replacement) == fitted


def test_lawyer_titles_block_a_party_role() -> None:
    assert _LAWYER.search("Regards,\nAdv. ") and _LAWYER.search("before Justice ")
    assert not _LAWYER.search("Our client ")


@pytest.mark.presidio
def test_check_command_prints_every_agent(capsys) -> None:
    from sanitise_legal.legal.__main__ import main

    main(
        [
            "check",
            "--mock",
            "--data",
            str(REPO / "data" / "legal"),
            "--doc",
            "c001_advice_email",
            "--question",
            "What did we advise?",
        ]
    )
    out = capsys.readouterr().out
    for heading in ("AGENT 1", "AGENT 2", "AGENT 3: routing", "AGENT 3: safe text", "ANSWER"):
        assert heading in out
    assert "no residual leaks" in out and "Bose" not in out.split("safe text")[1]


def test_preflight_reports_missing_keys(monkeypatch, capsys) -> None:
    from sanitise_legal.legal import __main__ as cli
    from sanitise_legal.llm.gemini import KEY_NAMES

    for name in (*KEY_NAMES, "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)

    async def unreachable(_settings):
        return {"reachable": False, "model_available": False, "detail": "offline"}

    monkeypatch.setattr(cli, "_ollama_health", unreachable)
    monkeypatch.chdir(REPO)
    assert cli._preflight(REPO / "data" / "legal") is False
    out = capsys.readouterr().out
    assert "[FAIL] 0 key(s)" in out and "legal: agents gemma4:e4b" in out
