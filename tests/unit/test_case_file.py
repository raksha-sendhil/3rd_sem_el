"""Clinical records: recognised by their sections, about the patient, written as a case summary."""

from sanitise.agents.extractor import casefile
from sanitise.agents.rewriter import prompt as rewrite_prompt
from sanitise.schemas import (
    Action,
    FactTuple,
    Importance,
    PIICategory,
    RoutingDecision,
    SensitiveSpan,
    Subject,
)

NOTE = (
    "CLINIC NOTE\n"
    "Patient: Mrs. Leela Menon    Age/Sex: 61/F\n"
    "C/O: knee pain x 2 months, better with rest.\n"
    "O/E: BP 142/90. Wt 70 kg.\n"
    "Imp: osteoarthritis. Plan: X-ray, review in 2 weeks. Son: Arun, Ph 90000 11111.\n"
)
CHAT = "Hi doctor, my son Arun has knee pain for 2 months. Plan to visit next week."


def _span(text: str, needle: str, category: PIICategory) -> SensitiveSpan:
    start = NOTE.index(needle)
    return SensitiveSpan(
        text=needle, category=category, start=start, end=start + len(needle), method="rule"
    )


def test_a_record_with_clinical_sections_is_a_case_file_and_a_chat_is_not():
    assert casefile.is_case_file(NOTE)
    assert not casefile.is_case_file(CHAT)


def test_the_patient_is_the_person_in_the_patient_field_even_with_a_title():
    subjects = [
        Subject(id="writer", label="the writer"),
        Subject(id="person_1", label="Leela Menon", mentions=["Leela Menon"]),
        Subject(id="writer.son", label="the writer's son", mentions=["Son", "Arun"]),
    ]
    spans = [
        _span(NOTE, "Leela Menon", PIICategory.NAME),
        _span(NOTE, "Arun", PIICategory.NAME),
    ]
    patient = casefile.patient_id(NOTE, spans, subjects)
    assert patient == "person_1"
    labels = {s.id: s.label for s in casefile.relabel(subjects, patient)}
    assert labels == {
        "writer": "the clinician who wrote the record",
        "person_1": "the patient",
        "writer.son": "the patient's son",
    }
    assert [casefile.rename(s.id, patient) for s in subjects] == [
        "writer",
        "patient",
        "patient.son",
    ]


def test_without_a_patient_field_the_writer_is_the_patient():
    assert casefile.patient_id("C/O: cough\nO/E: clear\nPlan: rest", [], []) == "writer"
    assert casefile.rename("writer", "writer") == "patient"


def test_a_header_id_belongs_to_the_patient_not_the_clinician():
    fact = FactTuple(
        subject="writer", relation="identifier", value="MRN-1", importance=Importance.LOW
    )
    assert casefile.own_identity([fact], "person_1")[0].subject == "person_1"
    assert casefile.own_identity([fact], None)[0].subject == "writer"


def test_case_file_facts_get_the_case_summary_prompt_and_role_labels():
    facts = [
        FactTuple(subject="patient", relation="has_symptom", value="knee pain", importance="high"),
        FactTuple(subject="patient.son", relation="contact", value="[PHONE_1]", importance="low"),
    ]
    assert rewrite_prompt.is_case_file(facts)
    assert not rewrite_prompt.is_case_file([facts[0].model_copy(update={"subject": "writer"})])
    draft = rewrite_prompt.draft(facts, [])
    assert "the patient - has symptom: knee pain." in draft
    assert "the patient's son - contact: [PHONE_1]." in draft


def test_a_short_removed_detail_is_replaced_only_as_a_whole_word():
    m = RoutingDecision(
        span_text="M",
        category=PIICategory.OCCUPATION,
        action=Action.GENERALISE,
        replacement="a working professional",
        reason="test",
    )
    assert rewrite_prompt.safe_value("on antimalarials", [m]) == "on antimalarials"
    age = m.model_copy(
        update={"span_text": "9", "category": PIICategory.AGE, "replacement": "a child"}
    )
    assert rewrite_prompt.safe_value("Hb 9.8 g/dL, age 9", [age]) == "Hb 9.8 g/dL, age a child"


def test_a_date_made_vague_to_its_year_reads_in_not_on():
    d = RoutingDecision(
        span_text="23/10/2026",
        category=PIICategory.DATE,
        action=Action.GENERALISE,
        replacement="2026",
        reason="test",
    )
    assert rewrite_prompt.safe_value("Review on 23/10/2026 if worse", [d]) == (
        "Review in 2026 if worse"
    )
