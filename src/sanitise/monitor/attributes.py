"""What a protected answer can reveal about a person, and how to say it less precisely.

The protected pipeline answers from sanitised text, so the only personal details an answer can
carry are the ones Agent 3 left in, kept or generalised: "vertigo", "in their 30s", "a town in
the Nilgiris district", "a healthcare worker". Names, phones and IDs are placeholders and carry
nothing. Each such detail is an ``Attribute``:

* ``subject_id``: whose detail it is, ``<doc_id>:<subject>`` (Agent 2's subject, e.g.
  ``writer``); one record per person, as the sanitised store keeps records unlinkable;
* ``type``: age, date, location, occupation, condition or physical (``ATTRIBUTE_TYPES``);
  symptoms and relations are left out, they describe almost everyone;
* ``ladder``: the form shown in the sanitised text first, then coarser forms from Agent 3's
  generalisation ladder (``ontology.generalise``), e.g. "a town in the Nilgiris district" ->
  "a town in Tamil Nadu". The ladder's last word, the generic phrase ("a medical condition"),
  is not on it: it reveals nothing, so it never counts as a disclosure.

``mentioned(text, phrase)`` finds a form in an answer the way a model tends to repeat it: the
whole phrase, or its core ("Nilgiris" for "a town in the Nilgiris district", "thirties" for
"in their 30s"), whole words only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sanitise.agents.rewriter.ontology import GENERIC, generalise
from sanitise.schemas import Action, PIICategory, SanitisedRecord

C = PIICategory

TYPE_OF: dict[PIICategory, str] = {
    C.AGE: "age",
    C.DATE: "date",
    C.LOCATION: "location",
    C.ADDRESS: "location",
    C.OCCUPATION: "occupation",
    C.MEDICAL_CONDITION: "condition",
    C.PHYSICAL: "physical",
}
ATTRIBUTE_TYPES = tuple(dict.fromkeys(TYPE_OF.values()))
_GENERIC = {phrase.lower() for phrase in GENERIC.values()}
_DECADES = {"2": "twenties", "3": "thirties", "4": "forties", "5": "fifties", "6": "sixties",
            "7": "seventies", "8": "eighties", "9": "nineties"}  # fmt: skip
_LEAD = re.compile(
    r"^(?:in (?:their|his|her) )?(?:a|an|the)?\s*"
    r"(?:(?:town|city|village|place|area|hospital|clinic|school|university|state|region)\s+"
    r"(?:in|near|of)\s+(?:the\s+)?)?",
    re.I,
)
# words around a core mention that belong to it: "in the Nilgiris district", "in her thirties"
_AROUND_BEFORE = r"(?:(?:in\s+(?:their|his|her)\s+)|(?:the\s+))?"
_AROUND_AFTER = r"(?:\s+district)?"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .,;:")


def is_generic(phrase: str) -> bool:
    return norm(phrase) in _GENERIC


@dataclass(frozen=True)
class Attribute:
    subject_id: str
    type: str
    ladder: tuple[str, ...]  # [0] = the form in the sanitised text; never empty
    generic: str  # the phrase that says nothing ("a working professional"): the last resort

    @property
    def shown(self) -> str:
        return self.ladder[0]

    def form(self, level: int) -> str:
        """The form at ``level`` steps up; one past the ladder is the generic phrase."""
        return self.ladder[level] if level < len(self.ladder) else self.generic

    def reveals(self, level: int) -> bool:
        return level < len(self.ladder)


def _ladder(span: str, category: PIICategory, action: Action, replacement: str | None) -> list[str]:
    """The shown form, then each coarser form the ontology gives, without the generic one."""
    shown = span if action is Action.KEEP else (replacement or "")
    if not shown or is_generic(shown):
        return []
    forms = [shown]
    start = 1
    if action is Action.GENERALISE:  # continue from the level the rewrite used
        for level in (1, 2, 3):
            if norm(generalise(span, category, level=level)) == norm(shown):
                start = level + 1
                break
        else:
            start = 2
    for level in range(start, 4):
        form = generalise(span, category, level=level)
        if is_generic(form) or norm(form) in {norm(f) for f in forms}:
            continue
        forms.append(form)
    return forms


def record_attributes(record: SanitisedRecord) -> list[Attribute]:
    """Every attribute the record's sanitised text can reveal, with the person it is about."""
    owner = {norm(f.value): f.subject for f in record.extractor.facts}
    default = record.subjects[0].id if record.subjects else "main"
    out: list[Attribute] = []
    seen: set[tuple[str, str, str]] = set()
    for d in record.rewriter.decisions:
        kind = TYPE_OF.get(d.category)
        if kind is None or d.action is Action.PLACEHOLDER or d.start is None:
            continue
        ladder = _ladder(d.span_text, d.category, d.action, d.replacement)
        if not ladder:
            continue
        span = norm(d.span_text)
        subject = owner.get(span) or next(
            (s for v, s in owner.items() if v and (v in span or span in v)), default
        )
        key = (subject, kind, norm(ladder[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Attribute(f"{record.doc_id}:{subject}", kind, tuple(ladder), GENERIC[d.category])
        )
    return out


def _variants(phrase: str) -> list[str]:
    """The phrase and its core, longest first, as written: what a model is likely to repeat."""
    full = re.sub(r"\s+", " ", phrase).strip(" .,;:")
    core = re.sub(r"\s+district$", "", _LEAD.sub("", full), flags=re.I).strip()
    out = [full]
    if core and core.lower() != full.lower() and len(core) >= 4:
        out.append(core)
    decade = re.fullmatch(r"(\d)0s", core)
    if decade and decade.group(1) in _DECADES:
        out.append(_DECADES[decade.group(1)])
    return out


def _pattern(variant: str, *, around: bool = False) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(w) for w in variant.split())
    if around:
        words = _AROUND_BEFORE + words + _AROUND_AFTER
    return re.compile(rf"(?<![\w]){words}(?![\w])", re.I)


def mentioned(text: str, phrase: str) -> bool:
    return any(_pattern(v).search(text) for v in _variants(phrase))


def replace_form(text: str, old: str, new: str) -> str:
    """``text`` with every mention of ``old`` said as ``new`` instead. A mention of only the
    core ("the Nilgiris district", "her thirties") is replaced, with the words around it that
    belong to it, by the whole new phrase ("a town in Tamil Nadu", "a young adult")."""
    for i, variant in enumerate(_variants(old)):
        text = _pattern(variant, around=i > 0).sub(new, text)
    return text
