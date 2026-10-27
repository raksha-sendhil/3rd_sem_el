"""Person A: generalisation ladders and the rare-condition check."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from sanitise.agents.rewriter import ontology
from sanitise.agents.rewriter.leakcheck import contains
from sanitise.agents.rewriter.ontology import GENERIC, generalise, is_rare
from sanitise.schemas import PIICategory

C = PIICategory


@pytest.fixture
def tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Person B's tables, in the format of plan 6.3."""
    (tmp_path / "places.json").write_text(
        json.dumps(
            {
                "Coonoor": {"district": "Nilgiris", "state": "Tamil Nadu", "kind": "town"},
                "pune": {"district": "Pune", "state": "Maharashtra", "kind": "city"},
                "delhi": {"district": "New Delhi", "state": "Delhi", "kind": "city"},
            }
        )
    )
    (tmp_path / "occupations.json").write_text(json.dumps({"tea taster": "a food industry worker"}))
    (tmp_path / "conditions.json").write_text(
        json.dumps(
            {
                "amyloidosis": {"category": "a rare protein disorder", "rare": True},
                "type 2 diabetes": {"category": "a metabolic condition", "rare": False},
                "vertigo": {"category": "a balance problem", "rare": False},
            }
        )
    )
    monkeypatch.setattr(ontology, "ONTOLOGY_DIR", tmp_path)
    ontology.table.cache_clear()
    yield tmp_path
    ontology.table.cache_clear()


@pytest.fixture
def no_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(ontology, "ONTOLOGY_DIR", tmp_path / "missing")
    ontology.table.cache_clear()
    yield
    ontology.table.cache_clear()


@pytest.mark.parametrize(
    "text, one, two",
    [
        ("34 year old", "in their 30s", "a young adult"),
        ("6 years old", "a child", "a child"),
        ("8 months old", "a baby", "a child"),
        ("17 years old", "a teenager", "a teenager"),
        ("19", "in their late teens", "a young adult"),
        ("58 years old", "in their 50s", "middle-aged"),
        ("71 years old", "in their 70s", "an older adult"),
        ("92 years old", "aged 90 or over", "aged 90 or over"),
        ("thirty-two years old", "in their 30s", "a young adult"),
        ("her late forties", "in their 40s", "middle-aged"),
        ("an adult", "an adult", "an adult"),
    ],
)
def test_age_ladder(text: str, one: str, two: str) -> None:
    assert generalise(text, C.AGE) == one
    assert generalise(text, C.AGE, level=2) == two


@pytest.mark.parametrize(
    "text, one, two",
    [
        ("21 January 2024", "January 2024", "2024"),
        ("Tuesday, October 9, 2001", "October 2001", "2001"),
        ("September 2023", "2023", "the 2020s"),
        ("2015", "the 2010s", "the 2010s"),
        ("05/06/2022", "2022", "2022"),  # numeric: the month is ambiguous, so the year
        ("last month", GENERIC[C.DATE], GENERIC[C.DATE]),
    ],
)
def test_date_ladder(text: str, one: str, two: str) -> None:
    assert generalise(text, C.DATE) == one
    assert generalise(text, C.DATE, level=2) == two


def test_location_from_places_table(tables: Path) -> None:
    assert generalise("Coonoor", C.LOCATION) == "a town in the Nilgiris district"
    assert generalise("Coonoor", C.LOCATION, level=2) == "a town in Tamil Nadu"
    # district named after the city would repeat it: go straight to the state
    assert generalise("Pune", C.LOCATION) == "a city in Maharashtra"
    assert generalise("AIIMS Delhi", C.LOCATION) == "a hospital in Delhi"


def test_location_kinds_without_tables(no_tables: None) -> None:
    assert generalise("St. Thomas Hospital", C.LOCATION) == "a hospital"
    assert generalise("Greenfield Primary School", C.LOCATION) == "a school"
    assert generalise("IIT Madras", C.LOCATION) == "a university"
    assert generalise("Kavaratti island", C.LOCATION) == "an island"
    assert generalise("Coonoor", C.LOCATION) == GENERIC[C.LOCATION]


def test_occupation(tables: Path) -> None:
    assert generalise("tea taster", C.OCCUPATION) == "a food industry worker"  # table
    assert generalise("pediatric neurosurgeon", C.OCCUPATION) == "a healthcare worker"
    assert generalise("school librarian", C.OCCUPATION) == "an education worker"
    assert generalise("retired army officer", C.OCCUPATION) == "a retired public service worker"
    assert generalise("undergraduate student", C.OCCUPATION, level=2) == "a student"
    assert generalise("tea taster", C.OCCUPATION, level=2) == GENERIC[C.OCCUPATION]
    assert generalise("quantum knitter", C.OCCUPATION) == GENERIC[C.OCCUPATION]


def test_conditions_and_rarity(tables: Path) -> None:
    assert is_rare("Amyloidosis") and not is_rare("type 2 diabetes")
    assert generalise("amyloidosis", C.MEDICAL_CONDITION) == "a rare protein disorder"
    assert generalise("vertigo", C.MEDICAL_CONDITION) == "a balance problem"  # table wins
    assert generalise("multiple myeloma", C.MEDICAL_CONDITION) == "a type of cancer"
    assert generalise("amyloidosis", C.MEDICAL_CONDITION, level=2) == GENERIC[C.MEDICAL_CONDITION]


def test_rare_fallback_only_for_conditions_not_in_table(tables: Path) -> None:
    assert is_rare("multiple myeloma")  # not in the table: built-in list
    (tables / "conditions.json").write_text(
        json.dumps({"multiple myeloma": {"category": "a blood cancer", "rare": False}})
    )
    ontology.table.cache_clear()
    assert not is_rare("multiple myeloma")  # the table decides


@pytest.mark.parametrize(
    "text, one, two",
    [
        ("6 ft 3 inches", "between 6 ft and 6 ft 4 in", "tall"),
        ("5'4\"", "between 5 ft 2 in and 5 ft 6 in", "of average height"),
        ("180 cm", "between 180 and 190 cm", "of average height"),
        ("6 ft 1 in", "between 6 ft and 6 ft 4 in", "tall"),
        ("approximately 220 pounds", "between 220 and 240 pounds", "of average build"),
        ("78 kg", "between 70 and 80 kg", "of average build"),
        ("2.4 kg", "between 2 and 3 kg", "a baby's weight"),
        ("BMI of 31", "a BMI between 30 and 35", "a BMI in the obese range"),
    ],
)
def test_physical_ladder(text: str, one: str, two: str) -> None:
    assert generalise(text, C.PHYSICAL) == one
    assert generalise(text, C.PHYSICAL, level=2) == two


def test_symptom_drops_qualifiers() -> None:
    assert generalise("left-sided abdominal pain", C.SYMPTOM) == "abdominal pain"
    assert generalise("bilateral pelvic pain", C.SYMPTOM) == "pelvic pain"
    assert generalise("fever", C.SYMPTOM) == GENERIC[C.SYMPTOM]


def test_categories_without_ladder() -> None:
    assert generalise("my son", C.RELATION) == "a family member"
    assert generalise("x + y", C.QUASI_COMBO) == GENERIC[C.QUASI_COMBO]
    assert generalise("Priya Raman", C.NAME) == "a personal detail"


@pytest.mark.parametrize(
    "text, category",
    [
        ("32 years old", C.AGE),
        ("2015", C.DATE),
        ("21 January 2024", C.DATE),
        ("Coonoor", C.LOCATION),
        ("Fortis Hospital, Mulund", C.LOCATION),
        ("tea taster", C.OCCUPATION),
        ("approximately 220 pounds", C.PHYSICAL),
        ("6 ft 3 inches", C.PHYSICAL),
        ("amyloidosis", C.MEDICAL_CONDITION),
    ],
)
@pytest.mark.parametrize("level", [1, 2])
def test_generalisation_never_repeats_the_original(text: str, category: C, level: int) -> None:
    assert not contains(generalise(text, category, level=level), text, category)


def test_unreadable_table_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "places.json").write_text("{ not json")
    monkeypatch.setattr(ontology, "ONTOLOGY_DIR", tmp_path)
    ontology.table.cache_clear()
    try:
        assert generalise("Coonoor", C.LOCATION) == GENERIC[C.LOCATION]
    finally:
        ontology.table.cache_clear()
