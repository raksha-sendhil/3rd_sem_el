"""The legal recognizer layer (legal build spec): one regex at a time, then the whole pass.

The regex tests need Presidio's classes but not spaCy, so they run in the fast suite. The
``presidio``-marked tests at the bottom run the real analyzer over whole legal documents.
"""

import itertools

import pytest

pytest.importorskip("presidio_analyzer")

from sanitise.rules.recognizers import LAYERS
from sanitise.schemas import PIICategory as C

_LEGAL = {name: (entity, rules) for name, entity, rules in LAYERS["legal"]}


def hits(recognizer: str, text: str) -> list[str]:
    _entity, rules = _LEGAL[recognizer]
    out = []
    for rule in rules:
        for m in rule.regex.finditer(text):
            start, end = m.span("value") if "value" in rule.regex.groupindex else m.span()
            out.append(text[start:end])
    return out


@pytest.mark.parametrize(
    ("recognizer", "text", "expected"),
    [
        ("LegalIdRecognizer", "PAN ABCPM1234K on file", "ABCPM1234K"),
        ("LegalIdRecognizer", "Aadhaar 2345 6789 0123.", "2345 6789 0123"),
        ("LegalIdRecognizer", "Aadhaar no. 234567890123", "234567890123"),
        ("LegalIdRecognizer", "GSTIN 27ABCPM1234K1Z5", "27ABCPM1234K1Z5"),
        ("LegalIdRecognizer", "passport no. K1234567", "K1234567"),
        ("FinancialRecognizer", "IFSC HDFC0001234", "HDFC0001234"),
        ("FinancialRecognizer", "account no. 1234 5678 9012", "1234 5678 9012"),
        ("FinancialRecognizer", "card 4111 1111 1111 1111 used", "4111 1111 1111 1111"),
        ("CaseNumberRecognizer", "filed as CS(COMM) 412/2024 before", "CS(COMM) 412/2024"),
        ("CaseNumberRecognizer", "in O.S. No. 412/2024,", "O.S. No. 412/2024"),
        ("CaseNumberRecognizer", "W.P.(C) 1234 of 2023 is", "W.P.(C) 1234 of 2023"),
        ("CaseNumberRecognizer", "FIR No. 233/2025 was", "FIR No. 233/2025"),
        ("CaseNumberRecognizer", "complaint CC/118/2023 before", "CC/118/2023"),
        ("CaseNumberRecognizer", "CNR MHPU010012342024.", "MHPU010012342024"),
        ("CaseNumberRecognizer", "Bail Application No. 77/2025", "Bail Application No. 77/2025"),
        ("CourtRecognizer", "before the Bombay High Court on", "Bombay High Court"),
        ("CourtRecognizer", "In Pune District Court the", "Pune District Court"),
        ("CourtRecognizer", "the High Court of Delhi held", "High Court of Delhi"),
        ("CourtRecognizer", "petition before NCLT Mumbai", "NCLT Mumbai"),
        ("MoneyRecognizer", "paid Rs 4,25,00,000 for", "Rs 4,25,00,000"),
        ("MoneyRecognizer", "a cheque of ₹1.2 crore bounced", "₹1.2 crore"),
        ("MoneyRecognizer", "claimed 25 lakh rupees", "25 lakh rupees"),
        ("MoneyRecognizer", "INR 50,000/- only", "INR 50,000/-"),
        ("IndianPhoneRecognizer", "call +91 98450 12345 today", "+91 98450 12345"),
        ("IndianPhoneRecognizer", "mobile 09845012345.", "09845012345"),
        (
            "IndianAddressRecognizer",
            "lives at Flat 4B, Koregaon Park, Pune 411001. The",
            "Flat 4B, Koregaon Park, Pune 411001",
        ),
        ("IndianAddressRecognizer", "PIN code: 411 001", "411 001"),
        ("CompanyRecognizer", "sued Mehta Builders Pvt Ltd for", "Mehta Builders Pvt Ltd"),
        ("CompanyRecognizer", "against M/s Kale Developers. GSTIN", "Kale Developers"),
        ("MatterCodenameRecognizer", "under Project Falcon the", "Project Falcon"),
        ("LegalNameRecognizer", "Client: Rohan Mehta, a", "Rohan Mehta"),
        ("LegalNameRecognizer", "Respondent - Smt. Sunita Kale", "Sunita Kale"),
        ("LegalOccupationRecognizer", "is a cardiologist in", "cardiologist"),
    ],
)
def test_legal_rule_finds(recognizer, text, expected):
    assert expected in hits(recognizer, text)


@pytest.mark.parametrize(
    ("recognizer", "text"),
    [
        ("LegalIdRecognizer", "a 16-digit card 4111 1111 1111 1111"),  # not an Aadhaar
        ("CourtRecognizer", "the court adjourned; the district court will hear it"),
        ("MatterCodenameRecognizer", "the Project Manager signed"),
        ("CaseNumberRecognizer", "Section 138 of the NI Act 1881"),
        ("MoneyRecognizer", "in 2024 the client"),
        ("LegalNameRecognizer", "the client The"),
    ],
)
def test_legal_rule_ignores(recognizer, text):
    assert hits(recognizer, text) == []


# ------------------------------------------------------------------ whole documents

INTAKE = (
    "Intake note. Client: Rohan Mehta, a 52-year-old cardiologist, Flat 4B, Koregaon Park, "
    "Pune 411001. Phone +91 98450 12345, email r.mehta@example.in, PAN ABCPM1234K. The client "
    "says Mehta Builders Pvt Ltd took Rs 4,25,00,000 for a flat in 2021 and never delivered. "
    "Filed as CS(COMM) 412/2024 before the Bombay High Court on 12 March 2024. Opposing party: "
    "Sunita Kale. Witness: Anil Joshi. Code name Project Falcon."
)
EMAIL = (
    "Dear Mr. Mehta, regarding O.S. No. 412/2024 in the Pune District Court and FIR No. "
    "233/2025, please collect the bank statements for account no. 123456789012 (IFSC "
    "HDFC0001234) and your Aadhaar 2345 6789 0123. The cheque of ₹1.2 crore was dishonoured. "
    "Regards, Adv. Priya Nair"
)
CAUSE_TITLE = "Complaint CC/118/2023. Rohan Mehta v. M/s Kale Developers. Amount 25 lakh rupees."


@pytest.fixture(scope="module")
def detect():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip("en_core_web_lg not installed")
    from sanitise.rules import rule_detect

    return rule_detect


@pytest.mark.presidio
@pytest.mark.parametrize("text", [INTAKE, EMAIL, CAUSE_TITLE])
def test_offsets_exact_and_no_overlap(detect, text):
    spans = detect(text)
    for s in spans:
        assert text[s.start : s.end] == s.text
    for a, b in itertools.pairwise(spans):
        assert a.end <= b.start


@pytest.mark.presidio
def test_intake_note(detect):
    got = {(s.text, s.category) for s in detect(INTAKE)}
    assert {
        ("Rohan Mehta", C.NAME),
        ("52-year-old", C.AGE),
        ("cardiologist", C.OCCUPATION),
        ("Flat 4B, Koregaon Park, Pune 411001", C.ADDRESS),
        ("+91 98450 12345", C.PHONE),
        ("r.mehta@example.in", C.EMAIL),
        ("ABCPM1234K", C.ID_NUMBER),
        ("Mehta Builders Pvt Ltd", C.ORGANIZATION),
        ("Rs 4,25,00,000", C.MONEY),
        ("CS(COMM) 412/2024", C.CASE_NUMBER),
        ("Bombay High Court", C.COURT),
        ("12 March 2024", C.DATE),
        ("Sunita Kale", C.NAME),
        ("Anil Joshi", C.NAME),
        ("Project Falcon", C.MATTER_CODENAME),
    } <= got


@pytest.mark.presidio
def test_roles(detect):
    from sanitise.rules.roles import role_of

    roles = {s.text: role_of(s) for s in detect(INTAKE) + detect(EMAIL)}
    assert roles["Rohan Mehta"] == "client"
    assert roles["Sunita Kale"] == "party"
    assert roles["Anil Joshi"] == "witness"
    assert roles["Mehta"] == "client"  # "Dear Mr. Mehta": the addressee of advice
    assert roles["Priya Nair"] is None  # "Adv. Priya Nair" signs as the lawyer


@pytest.mark.presidio
def test_cause_title_is_split(detect):
    got = {(s.text, s.category) for s in detect(CAUSE_TITLE)}
    assert ("Rohan Mehta", C.NAME) in got
    assert ("Kale Developers", C.ORGANIZATION) in got
    assert ("CC/118/2023", C.CASE_NUMBER) in got
    assert not any(" v. " in t for t, _ in got)
