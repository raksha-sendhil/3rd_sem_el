"""Category mapping and the fixed precedence order used to resolve conflicts."""

from sanitise.schemas import PIICategory

C = PIICategory

# Presidio entity type -> project category. Entities not listed here are never
# requested from Presidio (ORGANIZATION is deliberately left out: in the Enron
# corpus it is almost always the employer itself and drowns the signal).
ENTITY_TO_CATEGORY: dict[str, PIICategory] = {
    # Presidio built-ins
    "PERSON": C.NAME,
    "DATE_TIME": C.DATE,
    "PHONE_NUMBER": C.PHONE,
    "EMAIL_ADDRESS": C.EMAIL,
    "LOCATION": C.LOCATION,
    "US_SSN": C.ID_NUMBER,
    "US_ITIN": C.ID_NUMBER,
    "US_PASSPORT": C.ID_NUMBER,
    "US_DRIVER_LICENSE": C.ID_NUMBER,
    "US_BANK_NUMBER": C.ID_NUMBER,
    "UK_NHS": C.ID_NUMBER,  # only with "NHS" written nearby, see engine._to_span
    "MEDICAL_LICENSE": C.ID_NUMBER,
    "CREDIT_CARD": C.ID_NUMBER,
    "IBAN_CODE": C.ID_NUMBER,
    "IP_ADDRESS": C.ID_NUMBER,
    "MAC_ADDRESS": C.ID_NUMBER,
    "NRP": C.OTHER,  # nationality / religion / political group
    "URL": C.OTHER,
    # Custom recognizers (recognizers.py) emit the project category names directly
    "AGE": C.AGE,
    "NAME": C.NAME,
    "PHYSICAL": C.PHYSICAL,
    "RELATION": C.RELATION,
    "OCCUPATION": C.OCCUPATION,
    "ADDRESS": C.ADDRESS,
    "ID_NUMBER": C.ID_NUMBER,
    "PHONE": C.PHONE,
    # legal layer (recognizers.py)
    "FINANCIAL": C.FINANCIAL,
    "CASE_NUMBER": C.CASE_NUMBER,
    # Not "ORGANIZATION": that name would also pull in every spaCy ORG hit ("High Court",
    # "NI Act", "Section 138"). Only company names with a legal-form suffix are rule spans;
    # the rest is the LLM pass's job.
    "COMPANY": C.ORGANIZATION,
    "COURT": C.COURT,
    "MONEY": C.MONEY,
    "MATTER_CODENAME": C.MATTER_CODENAME,
}

REQUESTED_ENTITIES: list[str] = sorted(ENTITY_TO_CATEGORY)

# When two spans of different categories overlap, the one earlier in this list
# wins. Ordered from most specific / most format-constrained to most generic, so
# e.g. "32 years old" is AGE not DATE, and "42 Elm St, Springfield" is ADDRESS
# not LOCATION. Agreed Day 1 - do not reorder without updating the annotation guide.
# Legal categories: case numbers and accounts are format-constrained like IDs (and beat
# the generic labelled-ID rule, so "account no. 1234..." is FINANCIAL); a court
# name beats the place inside it ("Pune District Court" is COURT, not LOCATION); money
# beats DATE so "Rs 2,024" is not a year.
CATEGORY_PRECEDENCE: list[PIICategory] = [
    C.EMAIL,
    C.CASE_NUMBER,
    C.FINANCIAL,
    C.ID_NUMBER,
    C.PHONE,
    C.ADDRESS,
    C.MONEY,
    C.AGE,
    C.PHYSICAL,
    C.DATE,
    C.COURT,
    C.MATTER_CODENAME,
    C.ORGANIZATION,
    C.NAME,
    C.OCCUPATION,
    C.LOCATION,
    C.RELATION,
    C.MEDICAL_CONDITION,
    C.SYMPTOM,
    C.LEGAL_ISSUE,
    C.QUASI_COMBO,
    C.OTHER,
]

_RANK = {cat: i for i, cat in enumerate(CATEGORY_PRECEDENCE)}
assert len(_RANK) == len(PIICategory), "every PIICategory needs a precedence rank"


def precedence(category: PIICategory) -> int:
    """Lower is stronger."""
    return _RANK[category]
