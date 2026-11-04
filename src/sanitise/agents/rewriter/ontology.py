"""Agent 3 - generalisation ladders.  OWNER: Person A (Deterministic Logic).

``generalise(text, category)`` returns a less specific phrase that still reads naturally in a
sentence and never contains the original text (the leak check would flag it). Each category
climbs a ladder; ``steps=2`` climbs two rungs, which routing uses for every member of a
QUASI_COMBO (legal build spec: "one step by default, two inside a combination").

==============  =======================================  ==================================
category        one step                                 two steps
==============  =======================================  ==================================
AGE             ``52-year-old`` -> ``in their 50s``      ``middle-aged``
DATE            ``12 March 2024`` -> ``March 2024``      ``2024``
LOCATION        ``Pune`` -> ``a city in Maharashtra``    ``a city in western India``
OCCUPATION      ``cardiologist`` -> ``a doctor``         ``a professional``
ORGANIZATION    ``Mehta Builders Pvt Ltd`` ->            ``a company``
                ``a real-estate developer``
COURT           ``Bombay High Court`` -> ``a High Court`` ``a court``
MONEY           ``Rs 4,25,00,000`` ->                    ``a large sum``
                ``between Rs 1 crore and Rs 10 crore``
LEGAL_ISSUE     (rare only) ``maritime salvage claim``   same
                -> its category in legal_issues.json
==============  =======================================  ==================================

Person B's tables in ``data/ontology/`` (format in docs/phase3_plan.md 6.3, plus
``legal_issues.json`` in the conditions format and an optional ``region`` on places) take
precedence; the built-in rules below cover what the tables do not.

``is_rare(issue)`` tells ``route`` which legal issues (or medical conditions) to generalise
rather than keep.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sanitise.schemas import PIICategory

C = PIICategory

ONTOLOGY_DIR = Path(__file__).resolve().parents[4] / "data" / "ontology"

_GENERIC: dict[PIICategory, str] = {
    C.LOCATION: "a place in India",
    C.OCCUPATION: "a working professional",
    C.DATE: "a recent date",
    C.PHYSICAL: "an average build",
    C.MEDICAL_CONDITION: "a medical condition",
    C.LEGAL_ISSUE: "a specialised legal dispute",
    C.ORGANIZATION: "a private company",
    C.COURT: "a court",
    C.MONEY: "a sum of money",
    C.QUASI_COMBO: "a combination of personal details",
}


@lru_cache(maxsize=8)
def table(name: str) -> dict[str, Any]:
    """``data/ontology/<name>.json`` with lower-case keys, or ``{}`` if it does not exist yet."""
    path = ONTOLOGY_DIR / f"{name}.json"
    if not path.is_file():
        return {}
    return {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def is_rare(condition: str) -> bool:
    """True for a legal issue (or medical condition) marked ``"rare": true`` in B's tables."""
    key = _key(condition)
    for name in ("legal_issues", "conditions"):
        entry = table(name).get(key)
        if isinstance(entry, dict) and entry.get("rare"):
            return True
    return False


# ---------------------------------------------------------------- AGE


def _age(text: str, steps: int) -> str:
    match = re.search(r"\d+", text)
    if not match:
        return "an adult"
    age = int(match.group())
    if steps >= 2:
        if age < 18:
            return "a minor"
        if age < 35:
            return "a young adult"
        return "middle-aged" if age < 60 else "a senior citizen"
    decade = age // 10 * 10
    return f"in their {decade}s" if decade >= 20 else "a young person"


# ---------------------------------------------------------------- DATE

_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_MONTH_RE = re.compile(r"\b(" + "|".join(m[:3] for m in _MONTHS) + r")[a-z]*\.?", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.-](\d{1,2})[/.-]((?:19|20)\d{2})\b")


def _part_of_decade(year: int) -> str:
    third = ("early", "mid", "late")[min(year % 10 // 4, 2)]
    return f"the {third} {year // 10 * 10}s"


def _date(text: str, steps: int) -> str:
    year_match = _YEAR_RE.search(text)
    numeric = _NUMERIC_DATE.search(text)
    month = None
    if numeric and 1 <= int(numeric.group(1)) <= 12:
        month = _MONTHS[int(numeric.group(1)) - 1].capitalize()
    elif m := _MONTH_RE.search(text):
        month = next(x for x in _MONTHS if x.startswith(m.group(1).lower())).capitalize()
    if not year_match:
        return "earlier that year" if month else _GENERIC[C.DATE]
    year = int(year_match.group())
    has_day = bool(numeric) or bool(
        re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b(?!\d)", _YEAR_RE.sub("", text))
    )
    # the rungs: day month year -> month year -> year -> part of a decade -> decade
    rungs = []
    if has_day and month:
        rungs.append(f"{month} {year}")
    if month:
        rungs.append(str(year))
    rungs += [_part_of_decade(year), f"the {year // 10 * 10}s"]
    return rungs[min(steps, len(rungs)) - 1]


# ---------------------------------------------------------------- LOCATION

# Fallback when places.json has no entry: city -> state for the cities the corpus uses.
_CITY_STATE = {
    "mumbai": "Maharashtra",
    "pune": "Maharashtra",
    "nagpur": "Maharashtra",
    "nashik": "Maharashtra",
    "thane": "Maharashtra",
    "aurangabad": "Maharashtra",
    "kolhapur": "Maharashtra",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "gurgaon": "Haryana",
    "gurugram": "Haryana",
    "noida": "Uttar Pradesh",
    "lucknow": "Uttar Pradesh",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "mysuru": "Karnataka",
    "mysore": "Karnataka",
    "mangaluru": "Karnataka",
    "chennai": "Tamil Nadu",
    "coimbatore": "Tamil Nadu",
    "madurai": "Tamil Nadu",
    "coonoor": "Tamil Nadu",
    "hyderabad": "Telangana",
    "kolkata": "West Bengal",
    "ahmedabad": "Gujarat",
    "surat": "Gujarat",
    "vadodara": "Gujarat",
    "jaipur": "Rajasthan",
    "kochi": "Kerala",
    "thiruvananthapuram": "Kerala",
    "bhopal": "Madhya Pradesh",
    "indore": "Madhya Pradesh",
    "chandigarh": "Chandigarh",
    "patna": "Bihar",
    "bhubaneswar": "Odisha",
    "panaji": "Goa",
}
_STATE_REGION = {
    "maharashtra": "western India",
    "gujarat": "western India",
    "goa": "western India",
    "rajasthan": "northern India",
    "delhi": "northern India",
    "haryana": "northern India",
    "punjab": "northern India",
    "chandigarh": "northern India",
    "uttar pradesh": "northern India",
    "karnataka": "southern India",
    "tamil nadu": "southern India",
    "kerala": "southern India",
    "telangana": "southern India",
    "andhra pradesh": "southern India",
    "west bengal": "eastern India",
    "bihar": "eastern India",
    "odisha": "eastern India",
    "madhya pradesh": "central India",
}


def _location(text: str, steps: int) -> str:
    key = _key(text)
    place = table("places").get(key)
    if isinstance(place, dict):
        kind, state = place.get("kind", "town"), place.get("state")
    elif key in _CITY_STATE:
        kind, state = "city", _CITY_STATE[key]
    elif key in _STATE_REGION:  # a state: one step is its region
        region = _STATE_REGION[key]
        return f"a state in {region}" if steps == 1 else "a state in India"
    else:
        return _GENERIC[C.LOCATION]
    if not state:
        return f"a {kind} in India"
    region = (place or {}).get("region") if isinstance(place, dict) else None
    region = region or _STATE_REGION.get(state.lower(), "India")
    return f"a {kind} in {state}" if steps == 1 else f"a {kind} in {region}"


# ---------------------------------------------------------------- OCCUPATION

_OCCUPATION_GROUP = {
    "cardiologist": "a doctor",
    "surgeon": "a doctor",
    "physician": "a doctor",
    "dentist": "a doctor",
    "paediatrician": "a doctor",
    "pediatrician": "a doctor",
    "chartered accountant": "a finance professional",
    "accountant": "a finance professional",
    "bank manager": "a finance professional",
    "software engineer": "an engineer",
    "civil engineer": "an engineer",
    "engineer": "an engineer",
    "professor": "an educator",
    "teacher": "an educator",
    "lecturer": "an educator",
    "shop owner": "a small business owner",
    "shopkeeper": "a small business owner",
    "trader": "a small business owner",
    "builder": "a construction professional",
    "contractor": "a construction professional",
    "government employee": "a public servant",
    "retired government officer": "a retired public servant",
}


def _occupation(text: str, steps: int) -> str:
    key = _key(text)
    entry = table("occupations").get(key)
    if isinstance(entry, dict):
        group, sector = entry.get("group"), entry.get("sector")
    else:
        group, sector = entry or _OCCUPATION_GROUP.get(key), None
    if steps >= 2 or not group:
        return sector or (_GENERIC[C.OCCUPATION] if not group else "a professional")
    return str(group)


# ---------------------------------------------------------------- ORGANIZATION

_SECTORS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"builders?|developers?|construction|infra|realty|estates?|housing|homes", re.I),
        "a real-estate developer",
    ),
    (
        re.compile(r"bank|finance|financial|capital|credit|investments?|nbfc", re.I),
        "a financial institution",
    ),
    (re.compile(r"insurance|assurance", re.I), "an insurance company"),
    (
        re.compile(r"pharma|pharmaceuticals?|healthcare|hospitals?|clinic|diagnostics", re.I),
        "a healthcare company",
    ),
    (re.compile(r"motors?|auto(?:mobiles?)?", re.I), "an automobile company"),
    (
        re.compile(r"tech|technologies|infotech|software|systems|solutions|digital", re.I),
        "a technology company",
    ),
    (re.compile(r"textiles?|garments?|fabrics?", re.I), "a textile company"),
    (re.compile(r"foods?|agro|dairy", re.I), "a food company"),
    (re.compile(r"logistics|transport|shipping|freight|movers", re.I), "a logistics company"),
    (
        re.compile(r"traders?|trading|exports?|imports?|enterprises|industries", re.I),
        "a trading business",
    ),
    (re.compile(r"co-?operative|society", re.I), "a co-operative society"),
    (re.compile(r"associates|& co\b|llp|chambers|legal", re.I), "a professional firm"),
]


def _organization(text: str, steps: int) -> str:
    if steps >= 2:
        return "a company"
    entry = table("organizations").get(_key(text))
    if entry:
        return str(entry)
    for pattern, sector in _SECTORS:
        if pattern.search(text):
            return sector
    return _GENERIC[C.ORGANIZATION]


# ---------------------------------------------------------------- COURT

_COURT_TYPES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"supreme\s+court", re.I), "the apex court"),
    (re.compile(r"high\s+court", re.I), "a High Court"),
    (re.compile(r"consumer|cdrc|ncdrc|scdrc|dcdrc", re.I), "a consumer commission"),
    (re.compile(r"nclt|nclat|drt|drat|itat|\bcat\b|rera|tribunal", re.I), "a tribunal"),
    (re.compile(r"family\s+court", re.I), "a family court"),
    (re.compile(r"sessions", re.I), "a sessions court"),
    (re.compile(r"magistrate", re.I), "a magistrate's court"),
    (re.compile(r"small\s+causes", re.I), "a small causes court"),
    (re.compile(r"commercial\s+court", re.I), "a commercial court"),
    (re.compile(r"labour|industrial", re.I), "a labour court"),
    (re.compile(r"district", re.I), "a district court"),
    (re.compile(r"civil", re.I), "a civil court"),
]


def court_type(text: str) -> str:
    """``Pune District Court`` -> ``a district court``. Also used by the session monitor."""
    for pattern, kind in _COURT_TYPES:
        if pattern.search(text):
            return kind
    return _GENERIC[C.COURT]


# ---------------------------------------------------------------- MONEY

_LAKH, _CRORE = 100_000, 10_000_000
_UNITS = {
    "thousand": 1_000,
    "k": 1_000,
    "lakh": _LAKH,
    "lakhs": _LAKH,
    "lac": _LAKH,
    "lacs": _LAKH,
    "crore": _CRORE,
    "crores": _CRORE,
    "cr": _CRORE,
    "million": 1_000_000,
    "mn": 1_000_000,
    "billion": 1_000_000_000,
}
_AMOUNT = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|k|lakhs?|lacs?|crores?|cr|million|mn|billion)?\b", re.I
)
_BANDS: list[tuple[float, str]] = [
    (_LAKH, "under ₹1 lakh"),
    (10 * _LAKH, "between ₹1 lakh and ₹10 lakh"),
    (_CRORE, "between ₹10 lakh and ₹1 crore"),
    (10 * _CRORE, "between ₹1 crore and ₹10 crore"),
    (100 * _CRORE, "between ₹10 crore and ₹100 crore"),
]


def rupees(text: str) -> float | None:
    """``Rs 4,25,00,000`` -> 42500000.0, ``1.2 crore`` -> 12000000.0; None if no amount."""
    match = _AMOUNT.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "").lower()
    return value * _UNITS.get(unit, 1)


def _money(text: str, steps: int) -> str:
    amount = rupees(text)
    if amount is None:
        return _GENERIC[C.MONEY]
    if steps >= 2:
        if amount < 10 * _LAKH:
            return "a modest sum"
        return "a substantial sum" if amount < _CRORE else "a large sum"
    for limit, band in _BANDS:
        if amount < limit:
            return band
    return "over ₹100 crore"


# ---------------------------------------------------------------- dispatch


def _from_table(name: str, text: str, category: PIICategory) -> str:
    entry = table(name).get(_key(text))
    if isinstance(entry, dict):
        return str(entry.get("category", _GENERIC[category]))
    return _GENERIC[category]


def generalise(text: str, category: PIICategory, *, steps: int = 1) -> str:
    """One rung up the ladder for ``category`` (``steps=2`` for QUASI_COMBO members)."""
    steps = max(1, steps)
    if category is C.AGE:
        return _age(text, steps)
    if category is C.DATE:
        return _date(text, steps)
    if category is C.LOCATION:
        return _location(text, steps)
    if category is C.OCCUPATION:
        return _occupation(text, steps)
    if category is C.ORGANIZATION:
        return _organization(text, steps)
    if category is C.COURT:
        return court_type(text) if steps == 1 else "a court"
    if category is C.MONEY:
        return _money(text, steps)
    if category is C.LEGAL_ISSUE:
        return _from_table("legal_issues", text, category)
    if category is C.MEDICAL_CONDITION:
        return _from_table("conditions", text, category)
    return _GENERIC.get(category, "a personal detail")
