"""Who said each fact.  OWNER: Person C.  Deterministic: the model is not asked.

A record with labelled turns ("Patient: ... Doctor: ...") has two speakers, and which of them
said a thing is content: "the doctor advised an MRI" is not "the patient had an MRI".
``attribute`` gives each fact the speaker of the turn that contains its words, as ``said_by``.
``uncovered`` lists the sentences of a reply that no fact covers, so Agent 2 is asked again.
"""

from __future__ import annotations

import re

from sanitise_finance.schemas import FactTuple

_ROLE = {
    "patient": "patient", "doctor": "doctor", "dr": "doctor", "physician": "doctor",
    "customer": "customer", "agent": "agent", "representative": "agent", "bank": "bank",
    "client": "client", "lawyer": "lawyer", "advocate": "lawyer",
    "caller": "customer", "advisor": "advisor",
}  # fmt: skip
_LABEL = re.compile(rf"^[ \t]*({'|'.join(_ROLE)})\.?[ \t]*:", re.IGNORECASE | re.MULTILINE)
# on a tie, these relations are the reply's
_REPLY_RELATIONS = {"assessment", "plan"}
_REPLIER = {"doctor", "agent", "bank", "lawyer", "advisor"}
_NOT_SAID = {"identifier", "contact", "relation_to"}


def turns(text: str) -> list[tuple[str, str]]:
    """(speaker, words) per turn. Empty unless two different speakers are labelled, and empty
    for a form, whose "Patient: Asha Rao" / "Doctor: R. Mehta" lines are fields, not speech."""
    marks = list(_LABEL.finditer(text))
    if len({_ROLE[m.group(1).lower()] for m in marks}) < 2:
        return []
    out = []
    for i, m in enumerate(marks):
        body = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        out.append((_ROLE[m.group(1).lower()], body))
    return out if sum(len(body.split()) >= 6 for _, body in out) >= 2 else []


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def attribute(text: str, facts: list[FactTuple]) -> list[FactTuple]:
    said = [(who, f" {' '.join(_words(body))} ") for who, body in turns(text)]
    if not said:
        return facts
    out = []
    for f in facts:
        words = _words(f.value)
        if f.relation in _NOT_SAID or not words:
            out.append(f)
            continue
        content = [w for w in words if len(w) > 2] or words
        phrase = f" {' '.join(words)} "
        reply = f.relation in _REPLY_RELATIONS
        scored = [
            (
                phrase in body,
                sum(f" {w} " in body for w in content),
                (who in _REPLIER) == reply,
                who,
            )
            for who, body in said
        ]
        _, share, _, speaker = max(scored, key=lambda s: s[:3])
        out.append(f.model_copy(update={"said_by": speaker}) if share else f)
    return out


_PLEASANTRY = re.compile(
    r"welcome|thank|regards|take care|best wishes|hope (?:this|i|that|you|it)|wish(?:ing)? you"
    r"|feel free|i (?:can )?understand your|(?:gone|went|read) (?:through )?your"
    r"|read your (?:question|query)|let me (?:know|try)|write back|further (?:quer|question)"
    r"|^(?:hi|hello|dear)\b.{0,30}$",
    re.IGNORECASE,
)


def uncovered(text: str, values: list[str]) -> list[str]:
    """Sentences of the replier's turns with fewer than 40% of their content words in any fact."""
    have = {w[:5] for v in values for w in _words(v)}
    out = []
    for who, body in turns(text):
        if who not in _REPLIER:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", body.strip()):
            words = {w[:5] for w in _words(sentence) if len(w) > 3}
            if len(words) < 4 or _PLEASANTRY.search(sentence):
                continue
            if len(words & have) < 0.4 * len(words):
                out.append(sentence.strip())
    return out
