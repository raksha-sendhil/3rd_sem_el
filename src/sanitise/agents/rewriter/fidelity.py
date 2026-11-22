"""Is the rewrite faithful?  OWNER: Person C.  Deterministic checks on Agent 3's text.

The prompt asks for third person, attribution and every fact. A model sometimes ignores that,
so code checks it:

* ``problems``: what is wrong (first person, a speaker never named, content facts missing, or
  ten words in a row copied from a fact: a Repeat Prompt). Agent 3 is asked once more.
* ``repair``: if it is still wrong, code fixes it. A rewrite in the wrong voice or without its
  speaker is replaced by ``render`` (the facts under who said them); a missing content fact is
  added with its speaker.

Only safe values are used, so nothing here can bring back a removed detail.
"""

from __future__ import annotations

import re

from sanitise.schemas import FactTuple

Safe = list[tuple[FactTuple, str]]

CONTENT = {
    "has_condition", "has_symptom", "took_medication", "had_procedure", "assessment", "plan",
    "test_result", "allergy", "vital_sign", "progress",
}  # fmt: skip
_TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")
# the pronoun "I", not the numeral ("type I", "stage I", "I.V.")
_PRONOUN_I = (
    r"(?<!ype )(?<!tage )(?<!lass )(?<!rade )(?<!hase )(?<!actor )(?<!olar )"
    r"(?<![\w./&-])I(?![\w./&-])"
)
_FIRST_PERSON = re.compile(rf"{_PRONOUN_I}|(?<![\w-])(?:my|My|myself)(?![\w-])")
_PERSON_SWAPS = [
    (r"\bI am\b|\bI['’]m\b", "they are"),
    (r"\bI was\b", "they were"),
    (r"\bI['’]ve\b", "they have"),
    (_PRONOUN_I, "they"),
    (r"\b(?:my|My|your|Your)\b", "their"),
    (r"\b(?:myself|yourself)\b", "themselves"),
    (r"\b(?:you|You) are\b", "they are"),
    (r"\bme\b", "them"),
    (r"\b(?:you|You)\b", "they"),
]


def third_person(value: str) -> str:
    for pattern, new in _PERSON_SWAPS:
        value = re.sub(pattern, new, value)
    return value


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _TOKEN.sub(" ", third_person(text)).lower())
    return {w[:5] for w in words if len(w) > 3 or w.isdigit()}


def missing(text: str, safe: Safe) -> Safe:
    """Content facts with fewer than half of their words in ``text``."""
    have = _stems(text)
    out = []
    for f, value in safe:
        if f.relation in CONTENT and (want := _stems(value)) and 2 * len(want & have) < len(want):
            out.append((f, value))
    return out


def first_person(text: str) -> bool:
    return bool(_FIRST_PERSON.search(_QUOTED.sub(" ", text)))


def unnamed_speakers(text: str, safe: Safe) -> list[str]:
    low = text.lower()
    who = {f.said_by for f, _ in safe if f.said_by and f.relation in CONTENT}
    return sorted(w for w in who if not re.search(rf"\b{re.escape(w)}", low))


COPY_WORDS = 10  # this many words in a row count as verbatim copying (Zeng et al. 2024)


def copied(text: str, safe: Safe) -> list[str]:
    """Values copied into ``text`` with ten or more words in a row."""
    said = f" {' '.join(re.findall(r'[a-z0-9]+', text.lower()))} "
    out = []
    for _, value in safe:
        words = re.findall(r"[a-z0-9]+", third_person(value).lower())
        runs = (words[i : i + COPY_WORDS] for i in range(len(words) - COPY_WORDS + 1))
        if any(f" {' '.join(run)} " in said for run in runs):
            out.append(value)
    return out


def problems(text: str, safe: Safe) -> list[str]:
    out = []
    if first_person(text):
        out.append('it uses "I" or "my": write everything in the third person')
    for who in unnamed_speakers(text, safe):
        out.append(f"it does not say what the {who} said: attribute those facts to the {who}")
    out += [f'it leaves out this fact: "{value}"' for _, value in missing(text, safe)]
    out += [
        f'it copies this word for word, so reword it and keep the terms and numbers: "{value}"'
        for value in copied(text, safe)
    ]
    return out


def _sentence(f: FactTuple, value: str, label: str) -> str:
    value = third_person(value).strip(" .")
    if f.said_by:
        verb = "reported"
        if f.said_by != "patient" and f.relation in ("plan", "assessment"):
            verb = "advised" if f.relation == "plan" else "gave this assessment"
        return f"The {f.said_by} {verb}: {value}."
    return f"{label[:1].upper()}{label[1:]}, {f.relation.replace('_', ' ')}: {value}."


def render(safe: Safe, labels: dict[str, str]) -> str:
    """The facts as plain third-person sentences, grouped by who said them, with no model."""
    groups: dict[str | None, list[str]] = {}
    for f, value in safe:
        if f.relation == "identifier" and value == labels[f.subject]:
            continue
        lines = groups.setdefault(f.said_by, [])
        line = _sentence(f, value, labels[f.subject])
        if line not in lines:
            lines.append(line)
    return " ".join(line for lines in groups.values() for line in lines)


def repair(text: str, safe: Safe, labels: dict[str, str]) -> str:
    if first_person(text) or unnamed_speakers(text, safe):
        return render(safe, labels)
    extra = [_sentence(f, value, labels[f.subject]) for f, value in missing(text, safe)]
    return " ".join([text.rstrip(), *dict.fromkeys(extra)]) if extra else text
