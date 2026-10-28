"""Agent 3 - output grounding check.  OWNER: Person C (Platform and Integration).

Person A's leak check looks for the details Agent 1 removed. This check covers the other way a
rewrite can identify someone: the model writing a name, number or address that is not in the
facts it was given at all (made up, or remembered from training). It needs no list of what was
removed, so it works on any document.

``ungrounded(safe_text, facts_text)`` reports every specific word in the rewrite that the
facts do not contain:

* a word with a digit in it ("1987", "B-12", "40s"), unless the same word is in the facts;
* an email address, unless it is in the facts;
* a capitalised word that is not the first word of a sentence, unless the facts contain it
  (ignoring case) or it is a form of address ("Doctor", "Sir"). Capitalised words next to it
  are reported with it, so "Helen Park" is one span even at the start of a sentence.

Placeholder tokens are never reported. A capitalised word on its own at the start of a sentence
is not checked: that is where ordinary words are capitalised, and a name there almost always
has a surname, or a later mention, that is.

``mask(safe_text, spans)`` replaces each reported span with ``[REDACTED]``.
"""

from __future__ import annotations

import re

from sanitise.schemas import PIICategory, SensitiveSpan

C = PIICategory
REDACTED = "[REDACTED]"
TOKEN = re.compile(r"\[[A-Z]+(?:_[A-Z]+)*(?:_\d+)?\]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|\n\s*|[\"'(]\s*)$")
ADDRESS_WORDS = {
    "i", "dr", "doctor", "doc", "mr", "mrs", "ms", "sir", "madam", "dear", "hello", "hi",
    "thanks", "thank", "regards", "ok",
}  # fmt: skip
# never part of a name, so never masked with one ("In Pune" masks only "Pune")
FUNCTION_WORDS = {
    "a", "an", "the", "my", "our", "your", "his", "her", "their", "its", "we", "you", "he",
    "she", "they", "it", "this", "that", "these", "those", "in", "on", "at", "of", "for", "to",
    "from", "with", "by", "and", "but", "or", "so", "since", "about", "after", "before", "also",
    "as", "if", "when", "please", "recently", "now", "then", "there", "here",
}  # fmt: skip


def _words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(TOKEN.sub(" ", text))}


def ungrounded(safe_text: str, facts_text: str) -> list[SensitiveSpan]:
    allowed = _words(facts_text) | ADDRESS_WORDS
    emails = {e.lower() for e in EMAIL.findall(facts_text)}
    blocked = [(m.start(), m.end()) for m in TOKEN.finditer(safe_text)]
    hits: list[tuple[int, int, PIICategory]] = []

    for m in EMAIL.finditer(safe_text):
        if m.group().lower() not in emails:
            hits.append((m.start(), m.end(), C.EMAIL))
            blocked.append((m.start(), m.end()))

    words = [m for m in WORD.finditer(safe_text) if not any(a <= m.start() < b for a, b in blocked)]

    def capital(i: int) -> bool:  # a capitalised word the facts do not explain
        word = words[i].group()
        return word[0].isupper() and word.lower() not in allowed | FUNCTION_WORDS

    def adjacent(i: int, j: int) -> bool:  # words i < j separated by one space only
        return safe_text[words[i].end() : words[j].start()] == " "

    i = 0
    while i < len(words):
        m = words[i]
        word = m.group()
        if any(c.isdigit() for c in word):
            if word.lower() not in allowed:
                hits.append((m.start(), m.end(), C.ID_NUMBER))
            i += 1
            continue
        at_start = _SENTENCE_START.search(safe_text[: m.start()]) is not None
        if not capital(i) or at_start:
            i += 1
            continue
        first = last = i
        while first > 0 and capital(first - 1) and adjacent(first - 1, first):
            first -= 1
        while last + 1 < len(words) and capital(last + 1) and adjacent(last, last + 1):
            last += 1
        hits.append((words[first].start(), words[last].end(), C.NAME))
        i = last + 1

    hits.sort()
    out: list[SensitiveSpan] = []
    for start, end, category in hits:
        if out and start < (out[-1].end or 0):
            continue
        out.append(
            SensitiveSpan(
                text=safe_text[start:end],
                category=category,
                start=start,
                end=end,
                method="rule",
                rationale="grounding check: not in the facts given to the model",
            )
        )
    return out


def mask(safe_text: str, spans: list[SensitiveSpan]) -> str:
    out = safe_text
    for s in sorted(spans, key=lambda s: s.start or 0, reverse=True):
        assert s.start is not None and s.end is not None
        out = out[: s.start] + REDACTED + out[s.end :]
    return out
