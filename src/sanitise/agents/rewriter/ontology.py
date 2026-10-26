"""Agent 3 - generalisation ladders.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER: reads Person B's tables in ``data/ontology/`` when they exist (format in
docs/phase3_plan.md 6.3) and otherwise falls back to one generic phrase per category. Replace
the body; keep the signature.

``generalise(text, category)`` returns a less specific phrase that still reads naturally in a
sentence. The real version (Person A, item 5) climbs real ladders:

* AGE        ``34 year old`` -> ``in their 30s``          (age band)
* LOCATION   ``Coonoor`` -> ``a town in Tamil Nadu``       (places.json: town -> state)
* OCCUPATION ``tea taster`` -> ``a food industry worker``  (occupations.json)
* MEDICAL_CONDITION (rare) ``amyloidosis`` -> ``a rare protein disorder`` (conditions.json)
* DATE       ``21 January 2024`` -> ``January 2024``
* PHYSICAL   ``6 ft 3 inches`` -> ``tall``

``is_rare(condition)`` tells ``route`` which conditions to generalise rather than keep.
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
    C.LOCATION: "a town in the region",
    C.OCCUPATION: "a working professional",
    C.DATE: "a recent date",
    C.PHYSICAL: "an average build",
    C.MEDICAL_CONDITION: "a medical condition",
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
    entry = table("conditions").get(_key(condition))
    return bool(isinstance(entry, dict) and entry.get("rare"))


def generalise(text: str, category: PIICategory) -> str:
    key = _key(text)
    if category is C.AGE:
        match = re.search(r"\d+", text)
        if match:
            decade = int(match.group()) // 10 * 10
            return f"in their {decade}s" if decade >= 20 else "a young person"
        return "an adult"
    if category is C.LOCATION and key in table("places"):
        place = table("places")[key]
        return f"a {place.get('kind', 'town')} in {place.get('state', 'the region')}"
    if category is C.OCCUPATION and key in table("occupations"):
        return str(table("occupations")[key])
    if category is C.MEDICAL_CONDITION and key in table("conditions"):
        return str(table("conditions")[key].get("category", _GENERIC[category]))
    return _GENERIC.get(category, "a personal detail")
