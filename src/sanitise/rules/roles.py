"""Party roles for NAME spans: client, opposing party or witness.

The legal build spec replaces names with ``[CLIENT_1]``, ``[PARTY_2]``, ``[WITNESS_1]``
rather than one ``[PERSON_n]`` series, so the answering model can still tell who is who.
The role is read from the words just before a name ("our client Rohan Mehta",
"Respondent: Sunita Kale", "... v. Sunita Kale") and recorded in the span's rationale as
``role=<role>``; routing copies it into the decision and ``allocate_placeholders`` picks
the token prefix from it.

A role found for one mention applies to every mention of the same name, and to a lone
first name or surname that belongs to it, so "Rohan" and "Mr. Mehta" share the token of
"Rohan Mehta".
"""

import re

from sanitise.schemas import PIICategory, SensitiveSpan

ROLES = ("client", "party", "witness")
_ROLE_TAG = re.compile(r"\brole=(client|party|witness)\b")

# What the cue words mean. The firm's own client sits on either side of a case, so only
# words that name our side ("client", the writer of an enquiry, the addressee of an advice
# email) map to client. Sign-offs are not cues: advice emails are signed by the lawyer.
_CUES: list[tuple[str, re.Pattern]] = [
    ("witness", re.compile(r"\b(?:witness(?:es)?|deponent|eye-?witness)\b", re.I)),
    (
        "party",
        re.compile(
            r"\b(?:respondent|defendant|accused|opposing\s+party|opposite\s+party|counterparty"
            r"|other\s+side|against|dispute\s+with|v\.|vs\.?|versus|drawer|landlord|builder)\b",
            re.I,
        ),
    ),
    (
        "client",
        re.compile(
            r"\b(?:client|petitioner|complainant|plaintiff|appellant|applicant|dear"
            r"|my\s+name\s+is|this\s+is|i\s+am)\b",
            re.I,
        ),
    ),
]
_WINDOW = 40  # characters before the name that may hold its cue
_SENTENCE_BREAK = re.compile(r"[.;\n]")
# a full stop after these is an abbreviation, not the end of a sentence
_ABBREVIATIONS = re.compile(r"\b(?:mr|mrs|ms|dr|v|vs|adv|no|smt|shri|sri|m/s|st|ltd|pvt)$", re.I)
_HONORIFICS = {"mr", "mrs", "ms", "dr", "shri", "sri", "smt", "kumari", "adv", "advocate"}


def role_of(span_or_text: SensitiveSpan | str | None) -> str | None:
    """The ``role=`` tag in a span's rationale (or in any string, such as a decision reason)."""
    text = span_or_text.rationale if isinstance(span_or_text, SensitiveSpan) else span_or_text
    match = _ROLE_TAG.search(text or "")
    return match.group(1) if match else None


def name_words(text: str) -> list[str]:
    """Lower-case words of a name, without honorifics: "Mr. Rohan Mehta" -> [rohan, mehta]."""
    words = re.findall(r"[a-z][a-z'’-]*", text.lower())
    return [w for w in words if w not in _HONORIFICS]


_LAWYER = re.compile(r"(?:adv|advocate|counsel|justice|judge)\.?\s*$", re.I)


def _cue(text: str, start: int) -> str | None:
    before = text[max(0, start - _WINDOW) : start]
    if _LAWYER.search(before):  # "Adv. Priya Nair" is a lawyer, not a party
        return None
    breaks = [
        m
        for m in _SENTENCE_BREAK.finditer(before)
        if not (m.group() == "." and _ABBREVIATIONS.search(before, 0, m.start()))
    ]
    if breaks:
        before = before[breaks[-1].end() :]
    best: tuple[int, str] | None = None
    for role, pattern in _CUES:
        for m in pattern.finditer(before):
            # the cue nearest to the name wins
            if best is None or m.end() > best[0]:
                best = (m.end(), role)
    return best[1] if best else None


def tag_roles(text: str, spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
    """Add ``role=<role>`` to the rationale of every NAME span whose role can be found."""
    names = [s for s in spans if s.category is PIICategory.NAME and s.start is not None]
    found: dict[tuple[str, ...], str] = {}
    for s in names:
        role = role_of(s) or _cue(text, s.start)
        key = tuple(name_words(s.text))
        if role and key and key not in found:
            found[key] = role
    if not found:
        return spans

    def role_for(span: SensitiveSpan) -> str | None:
        key = tuple(name_words(span.text))
        if key in found:
            return found[key]
        if len(key) == 1:  # a lone first name or surname of a tagged full name
            owners = {r for full, r in found.items() if len(full) > 1 and key[0] in full}
            if len(owners) == 1:
                return owners.pop()
        return None

    out = []
    for s in spans:
        role = role_for(s) if s in names and not role_of(s) else None
        if role:
            s = s.model_copy(update={"rationale": f"{s.rationale} role={role}"})
        out.append(s)
    return out
