"""Integration tests: the real Presidio + spaCy pipeline. Loads en_core_web_lg once."""

import pytest

spacy = pytest.importorskip("spacy")
if not spacy.util.is_package("en_core_web_lg"):
    pytest.skip("en_core_web_lg not installed", allow_module_level=True)

import itertools

from sanitise.rules import get_analyzer, rule_detect
from sanitise.schemas import PIICategory as C

pytestmark = pytest.mark.presidio

CASE_07 = (
    "My son has been experiencing left-sided abdominal pain and bilateral pelvic pain for "
    "approximately 10 weeks. Tests showed enlarged lymph nodes and mild leukocytosis. He is 32 "
    "years old, 6 ft 3 inches tall, and weighs approximately 220 pounds."
)

ENRON = (
    "John, please call me at 713/853-1234 ext. 5512 or Sally Beck's cell 713-555-0199. Send the "
    "contract no. ENA-20391 to 1400 Smith St., Houston, TX 77002. Jeff Skilling will review on "
    "Monday, October 2, 2000. Email jeff.skilling@enron.com."
)

CHAT = (
    "Hi doctor, I am a 45 year old female, a retired school teacher from Kerala. My 5 year old "
    "daughter has fever since yesterday. I'm 28, 5'4\" and 60 kg. BMI of 31. My husband is a nurse "
    "by profession. MRN: 00451234. Thanks Dr. Patel."
)


@pytest.fixture(scope="module", autouse=True)
def _warm():
    get_analyzer()


def found(text: str) -> set[tuple[str, C]]:
    return {(s.text, s.category) for s in rule_detect(text)}


@pytest.mark.parametrize("text", [CASE_07, ENRON, CHAT])
def test_invariants(text):
    spans = rule_detect(text)
    assert spans
    for s in spans:
        assert s.method == "rule"
        assert s.rationale
        assert 0.0 <= s.confidence <= 1.0
        assert text[s.start : s.end] == s.text
        assert s.text == s.text.strip()
    for a, b in itertools.pairwise(spans):
        assert a.end <= b.start, f"overlap: {a} / {b}"


def test_spec_demo_case_07():
    got = found(CASE_07)
    assert ("My son", C.RELATION) in got
    assert ("32 years old", C.AGE) in got
    assert ("6 ft 3 inches", C.PHYSICAL) in got
    assert ("approximately 220 pounds", C.PHYSICAL) in got
    # "approximately 10 weeks" is a duration - left for the LLM pass, not a DATE
    assert not any(cat is C.DATE for _, cat in got)


def test_enron_record():
    got = found(ENRON)
    assert {
        ("John", C.NAME),
        ("Sally Beck", C.NAME),
        ("Jeff Skilling", C.NAME),
        ("713-555-0199", C.PHONE),
        ("ENA-20391", C.ID_NUMBER),
        ("1400 Smith St., Houston, TX 77002", C.ADDRESS),
        ("Monday, October 2, 2000", C.DATE),
        ("jeff.skilling@enron.com", C.EMAIL),
    } <= got
    assert any(t.startswith("713/853-1234") and c is C.PHONE for t, c in got)


def test_chatdoctor_record():
    got = found(CHAT)
    assert {
        ("45 year old", C.AGE),
        ("retired school teacher", C.OCCUPATION),
        ("Kerala", C.LOCATION),
        ("5 year old", C.AGE),
        ("daughter", C.RELATION),
        ("28", C.AGE),
        ("5'4\"", C.PHYSICAL),
        ("60 kg", C.PHYSICAL),
        ("BMI of 31", C.PHYSICAL),
        ("My husband", C.RELATION),
        ("nurse", C.OCCUPATION),
        ("00451234", C.ID_NUMBER),
        ("Patel", C.NAME),
    } <= got
    assert not any(t.lower() in {"doctor", "yesterday"} for t, _ in got)


@pytest.mark.parametrize(
    "text, span, cat",
    [
        ("She is 67 yo and diabetic.", "67 yo", C.AGE),
        ("Patient aged 54 presented with cough.", "aged 54", C.AGE),
        ("A thirty-two years old man.", "thirty-two years old", C.AGE),
        ("My mom is in her late forties.", "her late forties", C.AGE),
        ("my 8 month old baby has a rash", "8 month old", C.AGE),
        ("He is 180 cm tall.", "180 cm", C.PHYSICAL),
        ("I weigh about 75 kgs.", "about 75 kgs", C.PHYSICAL),
        ("My mother-in-law has asthma.", "My mother-in-law", C.RELATION),
        ("her younger brother smokes", "her younger brother", C.RELATION),
        (
            "I work as a night shift security guard at a mall.",
            "night shift security guard",
            C.OCCUPATION,
        ),
        ("He was a professional footballer for years.", "professional footballer", C.OCCUPATION),
        ("Occupation: welder", "welder", C.OCCUPATION),
        ("Mail it to P.O. Box 1188, please.", "P.O. Box 1188", C.ADDRESS),
        ("zip code 90210", "90210", C.ADDRESS),
        ("Policy number: AB-778812", "AB-778812", C.ID_NUMBER),
        ("SSN 078-05-1120 on file", "078-05-1120", C.ID_NUMBER),
        ("Born on 12/04/1987 in Leeds.", "12/04/1987", C.DATE),
        ("Contact me at k.lay@enron.com today", "k.lay@enron.com", C.EMAIL),
    ],
)
def test_recognizer_hits(text, span, cat):
    assert (span, cat) in found(text)


@pytest.mark.parametrize(
    "text",
    [
        "I have had a headache for 2 days and it gets worse at night.",
        "Take 500 mg twice daily for three weeks.",
        "The lesion is 3 feet away from nothing; my pain is 7 out of 10.",
        "Hello doctor, thanks for your answer.",
    ],
)
def test_no_false_positives_on_plain_clinical_text(text):
    assert rule_detect(text) == []


def test_empty_and_whitespace():
    assert rule_detect("") == []
    assert rule_detect("   \n") == []


def test_deterministic():
    assert rule_detect(CHAT) == rule_detect(CHAT)


@pytest.mark.parametrize(
    "text, span, cat",
    [
        ("Seen at Manipal Hospital last week.", "Manipal Hospital", C.LOCATION),
        ("My father is 67 and diabetic.", "67", C.AGE),
        ("She passed away from cancer at 55.", "55", C.AGE),
        ("Tell Grandma happy 80th birthday!", "80th birthday", C.AGE),
        ("Hi Mom, can Dad call me?", "Mom", C.RELATION),
        ("Hi Mom, can Dad call me?", "Dad", C.RELATION),
        ("I am a 26 yr old female nurse.", "nurse", C.OCCUPATION),
        ("I am 58, retired army officer, from Pune.", "retired army officer", C.OCCUPATION),
        ("Thanks,\nGreg Whalley\nPresident, Enron Wholesale Services", "President", C.OCCUPATION),
        ("His birth weight was 2.4 kg.", "2.4 kg", C.PHYSICAL),
        ("Call Tammie at x3-5761 today.", "x3-5761", C.PHONE),
        ("CNIC 42101-1234567-1.", "42101-1234567-1", C.ID_NUMBER),
    ],
)
def test_round2_recognizer_hits(text, span, cat):
    assert (span, cat) in found(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hi, doctor, I have a cough.",
        "Hello, Doctor. My throat hurts.",
        "I think I have Peyronie's disease.",
        "The pain started this summer around 6pm.",
    ],
)
def test_round2_no_false_positives(text):
    assert rule_detect(text) == []


@pytest.mark.parametrize(
    "text, span",
    [
        (
            "I live at 22/7 Palm Grove Residency, Hosur Road, Bengaluru 560068 now.",
            "22/7 Palm Grove Residency, Hosur Road, Bengaluru 560068",
        ),
        (
            "Post the forms to Flat 4C, Lake View Towers, Kakkanad 682030.",
            "Flat 4C, Lake View Towers, Kakkanad 682030",
        ),
    ],
)
def test_indian_address_with_pin_code(text, span):
    assert (span, C.ADDRESS) in found(text)


def test_core_layer_is_presidio_only():
    got = {c for _, c in found_layers(CASE_07, ("core",))}
    assert not got & {C.AGE, C.PHYSICAL, C.RELATION}


def test_spec_layer_adds_demo_categories():
    got = found_layers(CASE_07, ("core", "spec"))
    assert {
        ("My son", C.RELATION),
        ("32 years old", C.AGE),
        ("6 ft 3 inches", C.PHYSICAL),
        ("approximately 220 pounds", C.PHYSICAL),
    } <= got


def test_email_layer_can_be_switched_off():
    text = "Call Tammie at x3-5761.\n\nGreg Whalley\nPresident, Enron Wholesale Services"
    without = found_layers(text, ("core", "spec", "general", "healthcare"))
    assert ("x3-5761", C.PHONE) not in without
    assert ("President", C.OCCUPATION) not in without
    assert {("x3-5761", C.PHONE), ("President", C.OCCUPATION)} <= found(text)


def test_unknown_layer_rejected():
    with pytest.raises(ValueError):
        rule_detect("text", layers=("core", "finance"))


def found_layers(text, layers):
    return {(s.text, s.category) for s in rule_detect(text, layers=layers)}


@pytest.mark.parametrize(
    "text, span, cat",
    [
        # ages written in words
        ("He is eleven, and is under the care of a doctor.", "eleven", C.AGE),
        ("She is nine and has asthma.", "nine", C.AGE),
        ("My son is twenty-one.", "twenty-one", C.AGE),
        ("He was aged twelve when it started.", "twelve", C.AGE),
        # names in phrasings spaCy misses for uncommon names
        ("Please call me at the earliest, and my name is Nila. Thanks.", "Nila", C.NAME),
        ("Her name is Sunaina Rao and she is unwell.", "Sunaina Rao", C.NAME),
        ("Hello, this is Kavya. My father is ill.", "Kavya", C.NAME),
        ("You can call me Ishani.", "Ishani", C.NAME),
        # a 10-digit mobile number that passes the UK NHS checksum is still a phone
        ("phone - 9881023483.", "9881023483", C.PHONE),
        ("Her NHS number is 943 476 5919.", "943 476 5919", C.ID_NUMBER),
    ],
)
def test_ages_in_words_names_and_nhs_context(text, span, cat):
    assert (span, cat) in found(text)


@pytest.mark.parametrize(
    "text",
    [
        "She is one of the best doctors.",
        "He is two days late for the test.",
        "I was prescribed a drug named Crocin.",
        "A condition called Bell's palsy.",
        "This is Springleaf Hospital.",
        "The name is not important.",
    ],
)
def test_no_age_or_name_from_lookalike_phrases(text):
    assert not {cat for _, cat in found(text)} & {C.AGE, C.NAME}


@pytest.mark.parametrize(
    "text, span",
    [
        ("Please send it today.\n\nThanks,\nNora Quint", "Nora Quint"),
        ("See you then.\nBest regards,\nOmar", "Omar"),
    ],
)
def test_name_after_a_sign_off(text, span):
    assert (span, C.NAME) in found(text)


@pytest.mark.parametrize(
    "text, span",
    [
        ("Please reply when you can. Thank you, Meher", "Meher"),
        ("Could you call me back?\nThanks, Joel Prakash.", "Joel Prakash"),
    ],
)
def test_name_after_a_sign_off_on_the_same_line(text, span):
    assert (span, C.NAME) in found(text)


@pytest.mark.parametrize(
    "text",
    ["Please reply soon. Thank you, Doctor.", "Thanks, Dr. Rao will see you.", "Thanks again."],
)
def test_no_name_from_a_sign_off_to_a_doctor(text):
    assert not any(t in {"Doctor", "Dr", "Again"} and c is C.NAME for t, c in found(text))
