"""Agent 3 - generalisation ladders.  OWNER: Person A (Deterministic Logic).

``generalise(text, category, *, level=1)`` returns a less specific phrase that still reads
naturally in a sentence. ``level`` is how many steps up the ladder to climb: one by default,
two for a member of a QUASI_COMBO (docs/phase3_plan.md section 9).

=================  ========================================  ===============================
Category           One step                                  Two steps
=================  ========================================  ===============================
AGE                ``34 year old`` -> ``in their 30s``        ``a young adult``
                   (``a baby``, ``a child``, ``a teenager``,  (child / teenager / young adult
                   ``aged 90 or over`` at the ends)           / middle-aged / older adult)
DATE               ``21 January 2024`` -> ``January 2024``    ``2024``  (day -> month -> year
                   ``2015`` -> ``the 2010s``                  -> decade)
LOCATION           ``Coonoor`` -> ``a town in the Nilgiris    ``a town in Tamil Nadu``
                   district`` (places.json)
                   named hospital / school / island -> ``a hospital in Delhi``, ``a university``
OCCUPATION         ``tea taster`` -> ``a food industry        ``a working professional``
                   worker`` (occupations.json, else a
                   keyword group)
MEDICAL_CONDITION  ``amyloidosis`` -> ``a rare protein        ``a medical condition``
                   disorder`` (conditions.json, else a
                   keyword group)
PHYSICAL           ``6 ft 3 inches`` -> ``between 6 ft and    ``tall``
                   6 ft 4 in``; ``220 pounds`` ->
                   ``between 220 and 240 pounds``
SYMPTOM            ``left-sided abdominal pain`` ->           ``a symptom``
                   ``abdominal pain`` (qualifiers dropped)
RELATION           ``my son`` -> ``a family member``          same
=================  ========================================  ===============================

Person B's tables in ``data/ontology/`` (format in plan 6.3) are read when they exist and win
over everything else; without them the keyword groups and generic phrases below are used, so
the pipeline works before the tables land. Whatever the ladder produces is checked with the
leak check's matcher: a phrase that would still contain the original detail climbs another
step, and if nothing is safe the generic phrase for the category is used.

``is_rare(condition)`` tells ``route`` which conditions to generalise rather than keep:
``"rare": true`` in conditions.json, or, for a condition the table does not list, a short
built-in list of well-known rare diseases.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sanitise.agents.rewriter.leakcheck import contains
from sanitise.schemas import PIICategory

C = PIICategory
log = logging.getLogger(__name__)

ONTOLOGY_DIR = Path(__file__).resolve().parents[4] / "data" / "ontology"

GENERIC: dict[PIICategory, str] = {
    C.AGE: "an adult",
    C.DATE: "an earlier date",
    C.LOCATION: "a town in the region",
    C.OCCUPATION: "a working professional",
    C.PHYSICAL: "of typical build",
    C.MEDICAL_CONDITION: "a medical condition",
    C.SYMPTOM: "a symptom",
    C.RELATION: "a family member",
    C.QUASI_COMBO: "a combination of personal details",
}


@lru_cache(maxsize=8)
def table(name: str) -> dict[str, Any]:
    """``data/ontology/<name>.json`` with lower-case keys, or ``{}`` if it does not exist yet."""
    path = ONTOLOGY_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable ontology table %s: %s", path, exc)
        return {}
    return {_key(k): v for k, v in data.items()} if isinstance(data, dict) else {}


def _key(text: str) -> str:
    return " ".join(re.sub(r"[^\w'\u2019 -]", " ", text.lower()).split())


# ---------------------------------------------------------------- AGE

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}  # fmt: skip
_DECADE_WORDS = {
    "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60,
    "seventies": 70, "eighties": 80, "nineties": 90,
}  # fmt: skip


def _age_years(text: str) -> float | None:
    lower = text.lower()
    if m := re.search(r"\b(\d0)'?s\b", lower):
        return float(m.group(1)) + 5
    for word, decade in _DECADE_WORDS.items():
        if word in lower:
            return decade + 5.0
    m = re.search(r"\d+(?:\.\d+)?", lower)
    if m:
        n = float(m.group())
    else:
        words = re.findall(r"[a-z]+", lower)
        values = [_NUMBER_WORDS[w] for w in words if w in _NUMBER_WORDS]
        if not values:
            return None
        n = float(sum(values))
    if re.search(r"\b(?:months?|mths?|mos?)\b", lower):
        return n / 12
    if re.search(r"\b(?:weeks?|wks?|days?)\b", lower):
        return 0.0
    return n


def _age(text: str, level: int) -> str:
    years = _age_years(text)
    if years is None:
        return GENERIC[C.AGE]
    if years >= 90:
        return "aged 90 or over"  # HIPAA Safe Harbor groups every age from 90 up
    if level == 1:
        if years < 2:
            return "a baby"
        if years < 13:
            return "a child"
        if years < 18:
            return "a teenager"
        if years < 20:
            return "in their late teens"
        return f"in their {int(years) // 10 * 10}s"
    if years < 13:
        return "a child"
    if years < 18:
        return "a teenager"
    if years < 40:
        return "a young adult"
    if years < 65:
        return "middle-aged"
    return "an older adult"


# ---------------------------------------------------------------- DATE

_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]  # fmt: skip


def _date(text: str, level: int) -> str:
    lower = text.lower()
    month = next(
        (i for i, name in enumerate(_MONTHS) if re.search(rf"\b{name[:3].lower()}", lower)), None
    )
    year_m = re.search(r"\b((?:19|20)\d{2})\b", lower)
    numeric = re.search(r"\b\d{1,2}[/.-]\d{1,2}[/.-](\d{2,4})\b", lower)
    year = int(year_m.group(1)) if year_m else None
    if year is None and numeric:
        y = numeric.group(1)
        year = int(y) if len(y) == 4 else 2000 + int(y)
    has_day = bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", re.sub(r"\b\d{4}\b", "", lower)))
    # precision of the original: 0 day, 1 month, 2 year, 3 decade
    if year is None:
        if month is not None and level == 1:
            return _MONTHS[month]
        return GENERIC[C.DATE]
    precision = 0 if (has_day and (month is not None or numeric)) else 1 if month is not None else 2
    target = min(precision + level, 3)
    if target == 1 and month is not None:
        return f"{_MONTHS[month]} {year}"
    if target <= 2:
        return str(year)
    return f"the {year // 10 * 10}s"


# ---------------------------------------------------------------- LOCATION

_FACILITY_KINDS: list[tuple[str, str]] = [
    (r"hospital|clinic|medcity|medical|health|infirmary|hospice|nursing home|aiims", "a hospital"),
    (r"university|college|institute|iit\b|iim\b|nit\b|academy|polytechnic", "a university"),
    (r"school|kindergarten|nursery", "a school"),
    (r"\bisland\b|\bisle\b", "an island"),
    (r"\bdesk\b|\boffice\b|\bplant\b|\brefinery\b", "a company site"),
]


def _place(key: str) -> dict[str, Any] | None:
    entry = table("places").get(key)
    return entry if isinstance(entry, dict) else None


def _place_phrase(name: str, entry: dict[str, Any], level: int, kind: str | None = None) -> str:
    kind = kind or f"a {entry.get('kind', 'town')}"
    district, state = entry.get("district"), entry.get("state")
    if level == 1 and district and _key(district) != _key(name):
        return f"{kind} in the {district} district"
    if state:
        return f"{kind} in {state}"
    return f"{kind} in the region"


def _location(text: str, level: int) -> str:
    key = _key(text)
    parts = [p.strip() for p in re.split(r",|\bin\b", text) if p.strip()]
    entry = _place(key)
    if entry:
        return _place_phrase(text, entry, level)
    for pattern, kind in _FACILITY_KINDS:
        if re.search(pattern, key):
            # "Fortis Hospital, Mulund" / "AIIMS Delhi": say where, one level up
            words = [w for p in parts for w in (p, *p.split())]
            for word in reversed(words):
                place = _place(_key(word))
                if place:
                    return _place_phrase(word, place, level + 1, kind)
            return kind
    for part in parts[1:] + [p for p in key.split() if len(p) > 3]:
        place = _place(_key(part))
        if place:
            return _place_phrase(part, place, level)
    return GENERIC[C.LOCATION]


# ---------------------------------------------------------------- OCCUPATION

_OCCUPATION_GROUPS: list[tuple[str, str]] = [
    (r"student|undergraduate|pupil|schoolboy|schoolgirl|graduate student|trainee", "a student"),
    (
        r"surgeon|doctor|physician|nurse|radiographer|pharmacist|dentist|therapist|paramedic"
        r"|midwife|psychiatrist|psychologist|neurologist|cardiologist|pediatrician|optometrist",
        "a healthcare worker",
    ),
    (
        r"teacher|librarian|lecturer|professor|principal|tutor|headmaster|headmistress",
        "an education worker",
    ),
    (
        r"player|athlete|fencer|swimmer|runner|footballer|cricketer|gymnast|boxer|wrestler"
        r"|coach|cyclist|sportsman|sportswoman",
        "a sportsperson",
    ),
    (r"taster|chef|cook|baker|waiter|waitress|butcher|brewer|barista", "a food industry worker"),
    (r"lighthouse|sailor|fisherman|seaman|mariner|deckhand|harbour|harbor", "a maritime worker"),
    (r"driver|pilot|conductor|courier|chauffeur", "a transport worker"),
    (
        r"engineer|developer|programmer|technician|scientist|researcher|analyst",
        "a technical professional",
    ),
    (
        r"electrician|plumber|carpenter|welder|mechanic|builder|labou?rer|mason|painter",
        "a tradesperson",
    ),
    (r"police|soldier|army|navy|firefighter|guard|officer", "a public service worker"),
    (r"farmer|planter|gardener|shepherd|grower", "an agricultural worker"),
    (r"cashier|shopkeeper|salesman|saleswoman|salesperson|retail|vendor", "a retail worker"),
    (
        r"scheduler|trader|manager|clerk|accountant|secretary|assistant|director|president"
        r"|executive|consultant|administrator|banker|broker|auditor|receptionist",
        "an office worker",
    ),
    (r"homemaker|housewife|househusband", "a homemaker"),
    (
        r"artist|musician|singer|dancer|actor|actress|writer|journalist|photographer|designer",
        "a creative professional",
    ),
]


def _occupation(text: str, level: int) -> str:
    key = _key(text)
    retired = re.match(r"(retired|former|ex-?)\s+", key)
    core = key[retired.end() :] if retired else key
    group = table("occupations").get(core)
    if not isinstance(group, str):
        group = next((g for pattern, g in _OCCUPATION_GROUPS if re.search(pattern, core)), None)
    if level >= 2 or group is None:
        group = "a student" if group == "a student" else GENERIC[C.OCCUPATION]
    if retired:
        return re.sub(r"^an? ", "a retired ", group)
    return group


# ---------------------------------------------------------------- MEDICAL_CONDITION

_CONDITION_GROUPS: list[tuple[str, str]] = [
    (
        r"cancer|myeloma|lymphoma|leuka?emia|carcinoma|sarcoma|melanoma|tumou?r|malignan",
        "a type of cancer",
    ),
    (r"amyloidosis", "a rare protein disorder"),
    (r"vertigo|tinnitus|labyrinth|meniere|hearing loss", "a balance or hearing problem"),
    (r"stroke|aneurysm|cerebrovascular", "a cerebrovascular condition"),
    (
        r"arrhythmia|tachycardia|fibrillation|coronary|heart|cardiac|angina|myocard",
        "a heart condition",
    ),
    (r"diabet|thyroid|metabolic|insulin|hormon", "a metabolic condition"),
    (r"asthma|copd|pneumonia|bronch|lung|pulmonary|tuberculosis", "a lung condition"),
    (r"dermatitis|eczema|psoriasis|acne|rash|skin|urticaria", "a skin condition"),
    (r"ligament|tendon|fracture|tear|sprain|cartilage|meniscus|dislocat", "a joint injury"),
    (r"nodule|cyst|polyp|lump|mass\b|lymph node", "a growth"),
    (
        r"neuropathy|sclerosis|epilep|parkinson|dementia|alzheimer|migraine",
        "a neurological condition",
    ),
    (r"leukocytosis|anemia|anaemia|thrombo|platelet|blood", "a blood condition"),
    (r"depress|anxiety|bipolar|schizo|ocd|ptsd", "a mental health condition"),
    (r"infection|hepatitis|hiv|virus|viral|bacterial", "an infection"),
]

# Used only for conditions conditions.json does not list.
_RARE_FALLBACK = re.compile(
    r"amyloidosis|myeloma|huntington|amyotrophic|\bals\b|cystic fibrosis|sarcoidosis|scleroderma"
    r"|myasthenia|wilson'?s disease|gaucher|fabry|pompe|guillain|syringomyelia|acromegaly"
    r"|addison|pheochromocytoma|mesothelioma|ha?emophilia|marfan|ehlers|porphyria|progeria"
    r"|narcolepsy|kawasaki|takayasu|behcet|moyamoya"
)


def is_rare(condition: str) -> bool:
    entry = table("conditions").get(_key(condition))
    if isinstance(entry, dict):
        return bool(entry.get("rare"))
    return bool(_RARE_FALLBACK.search(_key(condition)))


def _condition(text: str, level: int) -> str:
    entry = table("conditions").get(_key(text))
    if level == 1 and isinstance(entry, dict) and entry.get("category"):
        return str(entry["category"])
    if level == 1:
        group = next(
            (g for pattern, g in _CONDITION_GROUPS if re.search(pattern, _key(text))), None
        )
        if group:
            return group
    return GENERIC[C.MEDICAL_CONDITION]


# ---------------------------------------------------------------- PHYSICAL


def _feet(inches: float) -> str:
    feet, rest = divmod(round(inches), 12)
    return f"{feet} ft" + (f" {rest} in" if rest else "")


def _physical(text: str, level: int) -> str:
    lower = text.lower()
    if m := re.search(r"bmi\D{0,10}(\d{2}(?:\.\d)?)", lower):
        bmi = float(m.group(1))
        if level == 1:
            low = int(bmi) // 5 * 5
            return f"a BMI between {low} and {low + 5}"
        return (
            "a BMI in the obese range" if bmi >= 30 else "an overweight BMI" if bmi >= 25
            else "an underweight BMI" if bmi < 18.5 else "a healthy BMI"
        )  # fmt: skip
    inches = None
    if m := re.search(
        r"(\d)\s*(?:ft|feet|foot|'|\u2019)\.?\s*(?:and\s+)?(\d{1,2}(?:\.\d)?)?", lower
    ):
        inches = int(m.group(1)) * 12 + float(m.group(2) or 0)
    elif m := re.search(r"(\d{3}(?:\.\d)?)\s*(?:cm|cms|centimet)", lower):
        inches = float(m.group(1)) / 2.54
    elif m := re.search(r"\b([12]\.\d{1,2})\s*(?:m|meters?|metres?)\b", lower):
        inches = float(m.group(1)) * 100 / 2.54
    if inches is not None:
        if level == 1:
            if "cm" in lower or "met" in lower:
                low = int(inches * 2.54) // 10 * 10
                return f"between {low} and {low + 10} cm"
            low = int(inches) // 4 * 4
            if round(inches) == low:  # "5 ft 4 in" must not be an end of its own band
                low -= 2
            return f"between {_feet(low)} and {_feet(low + 4)}"
        return "tall" if inches >= 72 else "short" if inches <= 63 else "of average height"
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(lbs?|pounds?|kgs?|kilograms?|kilos?|stone)", lower):
        value, unit = float(m.group(1)), m.group(2)
        kg = (
            value * 0.4536
            if unit.startswith(("lb", "pound"))
            else value * 6.35
            if unit == "stone"
            else value
        )
        if level == 1:
            word = (
                "pounds"
                if unit.startswith(("lb", "pound"))
                else "stone"
                if unit == "stone"
                else "kg"
            )
            width = 20 if word == "pounds" else 2 if word == "stone" else 10
            if kg < 10:  # a baby's weight
                width = 1
            low = int(value) // width * width
            return f"between {low} and {low + width} {word}"
        if kg < 10:
            return "a baby's weight"
        return (
            "heavily built" if kg >= 100 else "of slight build" if kg <= 50 else "of average build"
        )
    return GENERIC[C.PHYSICAL]


# ---------------------------------------------------------------- SYMPTOM

_SYMPTOM_QUALIFIERS = re.compile(
    r"\b(?:left|right)(?:-|\s)?sided\b|\b(?:left|right)\b|\bbilateral\b|\bmild\b|\bsevere\b"
    r"|\bpersistent\b|\bchronic\b|\bacute\b|\bintermittent\b|\bsharp\b|\bdull\b|\bsudden\b"
    r"|\bconstant\b|\brecurrent\b|\bon\s+(?:my|his|her|their|both)\b|\bboth\b",
    re.IGNORECASE,
)


def _symptom(text: str, level: int) -> str:
    if level >= 2:
        return GENERIC[C.SYMPTOM]
    stripped = " ".join(_SYMPTOM_QUALIFIERS.sub(" ", text).split()).strip(" ,-")
    return stripped if stripped and _key(stripped) != _key(text) else GENERIC[C.SYMPTOM]


# ---------------------------------------------------------------- entry point

_LADDERS = {
    C.AGE: _age,
    C.DATE: _date,
    C.LOCATION: _location,
    C.OCCUPATION: _occupation,
    C.MEDICAL_CONDITION: _condition,
    C.PHYSICAL: _physical,
    C.SYMPTOM: _symptom,
}


def generalise(text: str, category: PIICategory, *, level: int = 1) -> str:
    ladder = _LADDERS.get(category)
    if ladder is None:
        return GENERIC.get(category, "a personal detail")
    for step in range(max(1, level), 4):
        phrase = ladder(text, step)
        if phrase and not contains(phrase, text, category):
            return phrase
    return GENERIC.get(category, "a personal detail")
