"""Clinical records (case sheets, clinic notes, discharge summaries).  OWNER: Person C.

A clinical record is written by a clinician about a patient, so ``subjects()``'s "writer" (the
author of a message) is the wrong frame for it. ``is_case_file`` recognises one by its labelled
sections (C/O, H/O, Diagnosis, Plan ...). Then:

* ``patient_id``: the patient is the person named in a patient field ("Patient: ...",
  "Name: ..."); with no such field, the record is about the writer's subject ("45 y/o M presents
  with ...") and the writer is the patient.
* ``relabel``: the labels the model sees: "the patient", "the clinician who wrote the record",
  and the patient's relatives ("Guardian: ... (mother)" is the patient's mother, not the
  clinician's).
* ``rename``: the ids in Agent 2's output: the patient becomes ``patient`` and a relative
  ``patient.<kin>``, so Agent 3 and the dashboard name them by role.

Ids stay as ``subjects()`` made them while Agent 2 runs (identity facts, subject resolution);
only the output is renamed.
"""

from __future__ import annotations

import re

from sanitise.schemas import FactTuple, PIICategory, SensitiveSpan, Subject

_SECTION = re.compile(
    r"^[ \t]*(c/o|h/o|o/e|k/c/o|chief complaints?|presenting complaints?|complaints?"
    r"|history of present(?:ing)? illness|hpi|past (?:medical )?history|pmh|family history"
    r"|examination|exam|vitals|vital signs|investigations?|inv|labs?|lab results"
    r"|impression|imp|diagnosis|provisional diagnosis|final diagnosis|assessment|plan|advice"
    r"|rx|treatment|medications?|allerg(?:y|ies)|course in (?:the )?hospital|hospital course"
    r"|follow[- ]up|mrn|uhid|ip no|op no|dob|age/sex|date of (?:visit|admission|discharge)"
    r"|admitted|discharged|consultant|referred by|patient(?: name)?)\b[^:\n]{0,15}:",
    re.IGNORECASE | re.MULTILINE,
)
_PATIENT_FIELD = re.compile(
    r"(?:patient(?:'s)?(?:\s+name)?|pt\.?(?:\s+name)?|name(?:\s+of\s+(?:the\s+)?patient)?)"
    r"\s*[:\-]\s*(?:(?:mr|mrs|ms|miss|master|baby|smt|shri|sri|kum|dr)\.?\s+)?$",
    re.IGNORECASE,
)


def is_case_file(text: str) -> bool:
    """Three or more different labelled clinical sections."""
    return len({m.group(1).lower() for m in _SECTION.finditer(text)}) >= 3


def patient_id(text: str, spans: list[SensitiveSpan], subjects: list[Subject]) -> str:
    for span in sorted(
        (s for s in spans if s.category is PIICategory.NAME and s.start is not None),
        key=lambda s: s.start or 0,
    ):
        assert span.start is not None
        line = text[text.rfind("\n", 0, span.start) + 1 : span.start]
        if _PATIENT_FIELD.search(line):
            for s in subjects:
                if span.text in s.mentions:
                    return s.id
    return "writer"


def relabel(subjects: list[Subject], patient: str) -> list[Subject]:
    out = []
    for s in subjects:
        if s.id == patient:
            label = "the patient"
        elif s.id == "writer":
            label = "the clinician who wrote the record"
        elif s.id.startswith("writer."):
            label = "the patient's " + s.id.removeprefix("writer.").replace("_", " ")
        else:
            label = s.label
        out.append(s.model_copy(update={"label": label}))
    return out


def own_identity(facts: list[FactTuple], patient: str | None) -> list[FactTuple]:
    """In a clinical record an ID number or contact detail with no one named before it (the
    record's header) is the patient's, not the clinician's."""
    if patient is None or patient == "writer":
        return facts
    return [
        f.model_copy(update={"subject": patient}) if f.subject == "writer" else f for f in facts
    ]


def rename(subject: str, patient: str) -> str:
    if subject == patient:
        return "patient"
    if subject.startswith("writer."):
        return "patient." + subject.removeprefix("writer.")
    return subject
