"""Custom regex recognizers, organised in layers on top of Presidio's defaults.

    core        Presidio + spaCy built-ins only (names, dates, emails, phones, ID
                formats, locations). Always on; nothing in this file.
    spec        AGE, PHYSICAL, RELATION - categories Presidio has no recognizer for,
                required by the spec demo (section 6). Not medical knowledge, just
                phrasing patterns.
    general     explicit identifiers Presidio misses in any domain: street addresses,
                labelled ID numbers, occupations.
    healthcare  named hospitals and clinics (ChatDoctor).
    email       Enron-style signatures and phone formats (713/853-1234, x3-5761,
                title lines). Switch off if the fixture set has no emails.
    legal       Indian litigation files (legal build spec): PAN, Aadhaar, GSTIN, bank
                accounts, case numbers, courts, rupee amounts, +91 phones, PIN-code
                addresses, company names, matter code names, party names.

A rule may define a named group ``value``; when present only that group becomes
the span (e.g. ``MRN: 00451234`` yields ``00451234``), otherwise the whole match.
"""

import re
from dataclasses import dataclass, field

from presidio_analyzer import EntityRecognizer, RecognizerResult


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    score: float
    flags: int = re.IGNORECASE
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile(self.pattern, self.flags))


class RegexRecognizer(EntityRecognizer):
    def __init__(self, name: str, entity: str, rules: list[Rule]):
        self.rules = rules
        super().__init__(supported_entities=[entity], name=name, supported_language="en")

    def load(self) -> None:
        pass

    def analyze(self, text, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        entity = self.supported_entities[0]
        results = []
        for rule in self.rules:
            has_value = "value" in rule.regex.groupindex
            for m in rule.regex.finditer(text):
                start, end = m.span("value") if has_value else m.span()
                if start < 0 or start == end:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=entity,
                        start=start,
                        end=end,
                        score=rule.score,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                            "rule": rule.name,
                        },
                    )
                )
        return results


def _alt(words: list[str]) -> str:
    # Longest first so "vice president" wins over "president".
    return "|".join(
        re.escape(w).replace(r"\ ", r"\s+") for w in sorted(words, key=len, reverse=True)
    )


# ---------------------------------------------------------------- AGE

_NUM_WORD = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
)
_AGE_UNIT = r"(?:years?|yrs?|months?|mths?|mos?|weeks?|wks?|days?)"
_DECADE = r"(?:\d0'?s|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)"
_AGE_SUBJECT = (
    r"\b(?:i\s+am|i'm|im|he\s+is|she\s+is|he's|she's|is\s+now|turned"
    r"|(?:father|mother|dad|mom|mum|son|daughter|husband|wife|brother|sister|grandfather"
    r"|grandmother|child|baby|patient)\s+is)"
)
_NOT_A_QUANTITY = (
    r"(?!\s*(?:%|percent|kgs?|lbs?|pounds?|cm|mm|ft|feet|foot|inch(?:es)?|mg|ml|hours?|hrs?"
    r"|days?|weeks?|months?|times|x\b|'|’|\"|\.\d|/\d))"
)

AGE_RULES = [
    Rule(
        "n_units_old",
        rf"\b(?:\d{{1,3}}(?:\.\d)?|{_NUM_WORD})[\s-]*{_AGE_UNIT}[\s-]*(?:old|of\s+age)\b",
        0.9,
    ),
    Rule("n_yo", r"\b\d{1,3}\s*(?:y/o|y\.o\.?|yo|yrs?\.?\s*old)(?![a-z])", 0.85),
    Rule("aged_n", r"\b(?:aged?|age:)\s*(?:of\s+|is\s+)?\d{1,3}\b" + _NOT_A_QUANTITY, 0.8),
    Rule("pronoun_is_n", rf"{_AGE_SUBJECT}\s+(?P<value>\d{{1,3}})\b" + _NOT_A_QUANTITY, 0.6),
    # "He is eleven," / "she is nine and ...": a number word only counts as an age when
    # the clause ends right after it, so "she is one of the ..." is not matched.
    Rule(
        "pronoun_is_word",
        rf"{_AGE_SUBJECT}\s+(?P<value>{_NUM_WORD})(?=\s*(?:[,.;:!?)]|$)|\s+(?:and|but|now|yet)\b)",
        0.6,
    ),
    Rule("aged_word", rf"\baged\s+(?P<value>{_NUM_WORD})\b(?!\s+{_AGE_UNIT})", 0.7),
    Rule("decade", rf"\b(?:my|his|her|their)\s+(?:early\s+|mid[\s-]|late\s+)?{_DECADE}", 0.7),
    Rule(
        "died_at_n",
        r"\b(?:died|passed\s+away|deceased|expired)\b[^.\n]{0,40}?\bat\s+(?:the\s+age\s+of\s+)?(?P<value>\d{1,3})\b"
        + _NOT_A_QUANTITY,
        0.6,
    ),
    Rule("ordinal_birthday", r"\b\d{1,3}(?:st|nd|rd|th)\s+birthday\b", 0.8),
]

# ---------------------------------------------------------------- NAME

# spaCy misses uncommon names when the context is weak ("my name is Nila"). These
# phrasings introduce a person's name whatever the name is. Case-sensitive: the name
# must be capitalised. ("named X" / "called X" are left out: "a drug named Crocin".)
_NAME_WORD = r"[A-Z][a-z]*(?:['’-][A-Z]?[a-z]+)*"
_NAME = rf"{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}}"

NAME_RULES = [
    Rule("name_is", rf"\b(?i:name\s+is|name's)\s+(?P<value>{_NAME})", 0.7, flags=0),
    Rule(
        "greeting_this_is",
        rf"\b(?i:hi|hello|hey|dear\s+\w+)\b[\s,!.-]*(?i:this\s+is)\s+(?P<value>{_NAME})",
        0.7,
        flags=0,
    ),
    Rule("title_name", rf"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+(?P<value>{_NAME})", 0.7, flags=0),
    Rule("call_me", rf"\b(?i:call\s+me)\s+(?P<value>{_NAME})", 0.6, flags=0),
]

# ---------------------------------------------------------------- PHYSICAL

_APPROX = r"(?:(?:approximately|approx\.?|about|around|roughly|nearly|almost|over|under|close\s+to|~)\s*)?"

PHYSICAL_RULES = [
    Rule(
        "height_imperial",
        r"\b[3-7]\s*(?:ft\.?|feet|foot|'|’)(?:\s*(?:and\s+)?\d{1,2}(?:\.\d)?\s*(?:inch(?:es)?|in\b\.?|\"|''|”|’’)?)?"
        r"(?!\s*(?:away|long|wide|deep|high|of|from|below|above)\b)",
        0.8,
    ),
    Rule("height_cm", r"\b1\d{2}(?:\.\d)?\s*(?:cm|cms|centimet(?:er|re)s?)\b", 0.6),
    Rule("height_m", r"\b[12]\.\d{1,2}\s*(?:m|meters?|metres?)\b", 0.5),
    Rule(
        "weight",
        rf"\b{_APPROX}(?:\d{{2,3}}(?:\.\d+)?|\d\.\d+)\s*(?:pounds|lbs?\.?|kgs?|kilograms?|kilos?|stone)(?![a-z])",
        0.8,
    ),
    Rule("bmi", r"\bBMI\s*(?:of|is|was|:|=)?\s*\d{2}(?:\.\d)?\b", 0.7),
]

# ---------------------------------------------------------------- RELATION

_POSSESSIVE = r"(?:my|his|her|our|their)"
_RELATIVE = (
    r"(?:(?:son|daughter|mother|father|brother|sister)[\s-]in[\s-]law|in[\s-]laws?"
    r"|(?:grand|great[\s-]grand|step)?(?:son|daughter|child|children|kids?|mother|father|parents?)"
    r"|boys?|girls?|babys?|babies|husband|wife|spouse|partner|boyfriend|girlfriend|fianc[eé]e?"
    r"|mom|mum|dad|brothers?|sisters?|siblings?|grandma|grandpa|aunt|uncle|niece|nephew|cousin|twins?)"
)
_REL_MODIFIER = r"(?:(?:little|baby|infant|newborn|elder|older|younger|eldest|youngest|only|teenage|adult|late|ex-?)\s+)"

RELATION_RULES = [
    Rule("possessive_relative", rf"\b{_POSSESSIVE}\s+{_REL_MODIFIER}{{0,2}}{_RELATIVE}s?\b", 0.85),
    # "my 5 year old son": the age phrase is an AGE span, so only the noun is RELATION
    Rule(
        "possessive_aged_relative",
        rf"\b{_POSSESSIVE}\s+(?:\d{{1,2}}|{_NUM_WORD})[\s-]*{_AGE_UNIT}[\s-]*old\s+(?P<value>{_RELATIVE}s?)\b",
        0.85,
    ),
    Rule(
        "kin_address",
        r"\b(?:Mom|Mum|Mommy|Mummy|Mama|Dad|Daddy|Papa|Grandma|Grandpa|Granny|Nana|Uncle|Aunt|Auntie)\b",
        0.6,
        flags=0,
    ),
]

# ---------------------------------------------------------------- OCCUPATION

OCCUPATIONS = [
    "accountant",
    "actor",
    "actress",
    "analyst",
    "architect",
    "artist",
    "athlete",
    "attorney",
    "baker",
    "banker",
    "barber",
    "bodybuilder",
    "builder",
    "businessman",
    "businesswoman",
    "butcher",
    "caregiver",
    "carer",
    "carpenter",
    "cashier",
    "ceo",
    "cfo",
    "chef",
    "cleaner",
    "clerk",
    "coach",
    "consultant",
    "contractor",
    "cook",
    "dancer",
    "dentist",
    "designer",
    "developer",
    "director",
    "doctor",
    "driver",
    "electrician",
    "engineer",
    "entrepreneur",
    "executive",
    "farmer",
    "firefighter",
    "fisherman",
    "flight attendant",
    "air hostess",
    "footballer",
    "guard",
    "gymnast",
    "hairdresser",
    "homemaker",
    "housewife",
    "intern",
    "janitor",
    "journalist",
    "judge",
    "labourer",
    "laborer",
    "lawyer",
    "lecturer",
    "librarian",
    "manager",
    "mechanic",
    "miner",
    "model",
    "musician",
    "nanny",
    "nurse",
    "officer",
    "paramedic",
    "pastor",
    "pharmacist",
    "photographer",
    "physician",
    "physiotherapist",
    "pilot",
    "plumber",
    "police officer",
    "policeman",
    "policewoman",
    "president",
    "principal",
    "headmaster",
    "headmistress",
    "administrator",
    "supervisor",
    "inspector",
    "auditor",
    "broker",
    "priest",
    "professor",
    "programmer",
    "psychologist",
    "receptionist",
    "researcher",
    "runner",
    "sailor",
    "salesman",
    "saleswoman",
    "salesperson",
    "scientist",
    "secretary",
    "shopkeeper",
    "singer",
    "soldier",
    "student",
    "surgeon",
    "swimmer",
    "tailor",
    "teacher",
    "technician",
    "therapist",
    "trader",
    "veterinarian",
    "vet",
    "vice president",
    "waiter",
    "waitress",
    "welder",
    "worker",
    "writer",
]
_OCC = rf"(?:{_alt(OCCUPATIONS)})s?"
# "a 26 yr old female nurse": age phrase and gender may sit between article and job
_AGE_GENDER = (
    r"(?:\d{1,3}[\s-]*(?:years?|yrs?|y/o|yo)[\s-]*(?:old\s+)?)?(?:(?:male|female|man|woman)\s+)?"
)
_TITLE_MOD = (
    r"(?:(?:Senior|Sr\.?|Junior|Jr\.?|Executive|Managing|Vice|Assistant|Associate|Chief|Deputy|Head"
    r"|General|Regional|Principal|Lead|Staff)\s+)"
)
_TITLE_HEAD = (
    r"(?:President|Director|Manager|Analyst|Engineer|Counsel|Officer|Specialist|Coordinator"
    r"|Assistant|Trader|Consultant|Secretary|Chairman|Chairwoman|CEO|CFO|COO|CTO|VP|Partner)"
)
_OCC_STOP = r"(?=\s*(?:[.,;:!?)\n]|$|\s(?:at|in|for|with|and|but|since|from|who|which|so|because|on|by|until|till|for)\b))"

OCCUPATION_RULES = [
    Rule(
        "works_as",
        r"\b(?:work(?:s|ed|ing)?|employed|job|career|serv(?:e|es|ed|ing)|trained)\s+as\s+(?:an?\s+|the\s+)?"
        r"(?P<value>[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}?)" + _OCC_STOP,
        0.75,
    ),
    Rule(
        "is_a_occupation",
        r"\b(?:am|is|was|are|were|became|become|becoming|as)\s+(?:an?|the)\s+"
        rf"{_AGE_GENDER}(?P<value>(?:[a-z][a-z-]*\s+){{0,2}}{_OCC})\b",
        0.7,
    ),
    # ", a retired school teacher from ..." - no verb, so require a known occupation head
    Rule(
        "appositive",
        rf"(?:,|\band)\s+an?\s+{_AGE_GENDER}(?P<value>(?:[a-z][a-z-]*\s+){{0,2}}{_OCC})\b",
        0.6,
    ),
    # ", retired army officer," - bare noun phrase between commas
    # needs a modifier so the salutation "Hi, doctor," is not an occupation
    Rule(
        "bare_appositive",
        r",\s+(?!(?:dear|hi|hello|hey|respected|good|my|the)\b)"
        rf"(?P<value>(?:[a-z][a-z-]*\s+){{1,2}}{_OCC})\s*(?=[,.])",
        0.55,
    ),
    Rule(
        "by_profession",
        r"\b(?:(?:an?|the)\s+)?(?P<value>[a-z][a-z-]*(?:\s+[a-z][a-z-]*)?)\s+by\s+profession\b",
        0.75,
    ),
    Rule(
        "occupation_label",
        r"\b(?:profession|occupation)\s*(?:is|:|-)\s*(?:an?\s+)?(?P<value>[a-z][a-z-]*(?:\s+[a-z][a-z-]*)?)",
        0.8,
    ),
]

# ---------------------------------------------------------------- LOCATION (facilities)

FACILITY_RULES = [
    Rule(
        "facility",
        r"\b(?:[A-Z][A-Za-z.'’-]*\s+){1,4}(?:Hospital|Clinic|Medical\s+Cent(?:er|re)|Health\s+Cent(?:er|re)"
        r"|Infirmary|Hospice|Nursing\s+Home|Polyclinic)s?\b",
        0.7,
        flags=0,
    ),
]

# ---------------------------------------------------------------- ADDRESS

_US_STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV"
    "|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY"
)
_STREET_TYPE = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl"
    r"|Terrace|Ter|Parkway|Pkwy|Highway|Hwy|Circle|Cir|Square|Sq|Trail|Trl|Loop|Close|Crescent)"
)

ADDRESS_RULES = [
    Rule(
        "street",
        rf"\b\d{{1,6}}[A-Za-z]?\s+(?:[A-Z][A-Za-z0-9.'-]*\s+){{1,4}}{_STREET_TYPE}\b\.?"
        r"(?:,?\s*(?:Apt|Apartment|Suite|Ste|Unit|Floor|Fl|Room|Rm|#)\.?\s*#?\s*[A-Za-z0-9-]+)?"
        r"(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}\s+\d{5}(?:-\d{4})?\b)?",
        0.85,
        flags=0,
    ),
    Rule(
        "city_state_zip",
        rf"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){{0,2}},\s*(?:{_US_STATES})\s+\d{{5}}(?:-\d{{4}})?\b",
        0.85,
        flags=0,
    ),
    Rule("state_zip", rf"\b(?:{_US_STATES})\s+\d{{5}}(?:-\d{{4}})?\b", 0.7, flags=0),
    Rule("po_box", r"\bP\.?\s?O\.?\s*Box\s+\d+\b", 0.85),
    Rule(
        "zip_labelled",
        r"\b(?:zip(?:\s*code)?|postal\s*code|post\s*code)\s*:?\s*(?P<value>\d{5}(?:-\d{4})?)\b",
        0.8,
    ),
    Rule("uk_postcode", r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", 0.6, flags=0),
]

# ---------------------------------------------------------------- ID_NUMBER

_ID_LABEL = (
    r"(?:MRN|medical\s+record(?:\s+(?:number|no\.?|#))?|(?:patient|case|policy|account|member|employee"
    r"|insurance|claim|reference|invoice|contract|deal|ticket|order|passport|licen[cs]e)\s+(?:id|no\.?|number|#)"
    r"|acct\.?\s*(?:no\.?|#)?|ssn|social\s+security(?:\s+(?:number|no\.?))?|ref\.?\s*(?:no\.?|#)"
    r"|cnic|nhs\s+(?:number|no\.?)|national\s+id(?:\s+(?:number|no\.?))?"
    r"|id\s*(?:no\.?|number|#)?)"
)

ID_RULES = [
    Rule(
        "labelled_id",
        rf"\b{_ID_LABEL}\s*(?:is|:|#|=|-)?\s*(?P<value>(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{{3,}})\b",
        0.9,
    ),
]

# ---------------------------------------------------------------- EMAIL layer

# signature line under a name: "President, Enron Wholesale Services"
SIGNATURE_TITLE_RULES = [
    Rule(
        "signature_title",
        rf"^[ \t]*(?P<value>{_TITLE_MOD}*{_TITLE_HEAD}(?:\s+of\s+[A-Z][A-Za-z]+)?)[ \t]*(?=,|$)",
        0.6,
        flags=re.MULTILINE,
    ),
]

EMAIL_PHONE_RULES = [
    # NANP with slash, common in Enron signatures: 713/853-1234
    Rule("nanp_slash", r"\b\d{3}/\d{3}-\d{4}\b", 0.6),
    Rule("extension", r"\b(?:ext\.?|extension)\s*\d(?:-?\d){2,5}\b", 0.6),
    Rule("x_extension", r"(?<![\w-])x\s?\d-?\d{4}\b", 0.6),
]


# ---------------------------------------------------------------- LEGAL layer (India)
# Legal build spec, "Sensitive categories and routing policy". Formats follow the spec's
# data section: Rs 4,25,000 and Rs 1.2 crore, +91 98450 12345, PAN ABCPM1234K, O.S. No.
# 412/2024, FIR No. 233/2025, CC/118/2023, CNR MHPU010012342024.

# A capitalised word that is not a sentence opener or preposition, so "In Bombay High
# Court" and "The Pune District Court" start at the place name.
_CAP_WORD = (
    r"(?!(?:The|In|At|Before|From|To|Of|And|By|On|For|Our|Their|His|Her|A|An|This|That)\b)"
    r"[A-Z][A-Za-z&'’-]*"
)
_YEAR = r"(?:19|20)\d{2}"

LEGAL_ID_RULES = [
    # PAN: 5 letters, 4 digits, 1 letter; the 4th letter is the holder type (P, C, H, F, ...)
    Rule("pan", r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b", 0.85, flags=0),
    Rule("aadhaar", r"(?<![\d-])[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}(?![\s-]?\d)", 0.8),
    Rule(
        "aadhaar_labelled",
        r"\b(?:aadhaa?r|UID)(?:\s+(?:card\s+)?(?:no\.?|number|#))?\s*(?::|-|is)?\s*(?P<value>[2-9]\d{3}\s?\d{4}\s?\d{4})\b",
        0.9,
    ),
    Rule("gstin", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", 0.9, flags=0),
    Rule(
        "passport_in",
        r"\bpassport(?:\s+(?:no\.?|number|#))?\s*(?::|-|is)?\s*(?P<value>[A-PR-WY][1-9]\d\s?\d{4}[1-9])\b",
        0.9,
    ),
    Rule(
        "voter_id",
        r"\b(?:EPIC|voter\s+id)(?:\s+(?:no\.?|number))?\s*:?\s*(?P<value>[A-Z]{3}\d{7})\b",
        0.85,
    ),
]

FINANCIAL_RULES = [
    Rule("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.85, flags=0),
    Rule(
        "bank_account",
        r"\b(?:a/c|account|acct\.?|bank\s+account|savings\s+account|current\s+account)"
        r"(?:\s+(?:no\.?|number|#))?\s*(?::|-|is)?\s*(?P<value>\d(?:[\s-]?\d){8,17})\b",
        0.9,
    ),
    Rule("card", r"(?<![\d-])\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}(?![\s-]?\d)", 0.85),
    Rule("upi", r"\b[\w.-]{2,}@(?:ok)?(?:axis|hdfcbank|icici|sbi|ybl|paytm|upi|ibl|apl)\b", 0.8),
]

_CASE_TYPE_WORDS = (
    r"(?:Writ\s+Petition|Bail\s+Application|Anticipatory\s+Bail\s+Application|Original\s+Suit"
    r"|Summary\s+Suit|Commercial\s+Suit|Complaint\s+Case|Consumer\s+Complaint|Criminal\s+Complaint"
    r"|Arbitration\s+Petition|Execution\s+Petition|Criminal\s+Appeal|Civil\s+Appeal|First\s+Appeal"
    r"|Second\s+Appeal|Revision\s+Petition|Special\s+Leave\s+Petition|Miscellaneous\s+Application"
    r"|Suit|Case|Appeal|Petition|Complaint|Application|Matter|Crime|C\.?R\.?)"
)
_CASE_ABBR = r"(?:[A-Z]{1,5}\.?(?:\s?[A-Z]{1,4}\.?){0,2}|Crl\.?\s?(?:A|M\.?P|Rev|Appeal|Misc)\.?)"
_CASE_NO = rf"\d{{1,6}}\s*(?:/|\s+of\s+)\s*{_YEAR}"

CASE_NUMBER_RULES = [
    # CS(COMM) 412/2024, O.S. No. 412/2024, W.P.(C) 1234 of 2023, Crl.A. 55/2022
    Rule(
        "abbreviated",
        rf"\b{_CASE_ABBR}\s?(?:\((?:[A-Za-z.]{{1,8}})\))?\s*(?:No\.?\s*|Nos?\.\s*)?{_CASE_NO}\b",
        0.85,
        flags=0,
    ),
    # Original Suit No. 412 of 2024, Bail Application No. 77/2025
    Rule(
        "worded",
        rf"\b{_CASE_TYPE_WORDS}\s*(?:\([A-Za-z.]{{1,8}}\)\s*)?(?:No\.?|Number|#)\s*:?\s*\d{{1,6}}(?:\s*(?:/|\s+of\s+)\s*{_YEAR})?\b",
        0.85,
    ),
    Rule(
        "fir",
        rf"\bF\.?I\.?R\.?\s*(?:No\.?|Number|#)?\s*:?\s*\d{{1,6}}(?:\s*/\s*{_YEAR})?\b",
        0.9,
        flags=0,
    ),
    Rule("slash_form", rf"\b[A-Z]{{1,6}}/\d{{1,6}}/{_YEAR}\b", 0.85, flags=0),  # CC/118/2023
    Rule("cnr", r"\b(?:CNR\s*(?:No\.?|Number)?\s*:?\s*)?(?P<value>[A-Z]{4}\d{12})\b", 0.9, flags=0),
]

_COURT_HEAD = (
    r"(?:High\s+Court|Supreme\s+Court|District\s+(?:and\s+Sessions\s+)?Court|Sessions\s+Court"
    r"|City\s+Civil\s+Court|Civil\s+Court|Family\s+Court|Small\s+Causes\s+Court|Magistrate(?:'s)?\s+Court"
    r"|Commercial\s+Court|Labour\s+Court|Industrial\s+Court|Consumer\s+(?:Disputes\s+Redressal\s+)?(?:Commission|Forum)"
    r"|Tribunal|Court)"
)
_PLACE = r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?"

COURT_RULES = [
    # Bombay High Court, Pune District Court, Delhi State Consumer Disputes Redressal Commission
    Rule(
        "named_court",
        rf"\b(?:{_CAP_WORD}[ \t]+){{1,4}}{_COURT_HEAD}\b(?:[ \t]+(?:at|of)[ \t]+{_PLACE})?",
        0.75,
        flags=0,
    ),
    # High Court of Delhi, District Court at Pune, District Court, Pune
    Rule("court_of_place", rf"\b{_COURT_HEAD}\s*(?:(?:at|of)\s+|,\s*){_PLACE}\b", 0.75, flags=0),
    Rule(
        "tribunal_acronym",
        r"\b(?:NCLT|NCLAT|DRT|DRAT|NCDRC|SCDRC|DCDRC|MahaRERA|RERA|ITAT|CAT)\b(?:[,\s]+(?:at\s+)?(?!Bench\b)[A-Z][a-z]+)?",
        0.7,
        flags=0,
    ),
]

_MONEY_UNIT = r"(?:\s*(?:lakhs?|lacs?|crores?|cr\.?|million|mn|billion|thousand|k)\b)?"
MONEY_RULES = [
    Rule(
        "rupee_symbol",
        rf"(?:₹|\bRs\.?|\bINR)\s?\d[\d,]*(?:\.\d+)?{_MONEY_UNIT}(?:\s*/-)?",
        0.85,
    ),
    Rule(
        "amount_in_words_unit",
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?|crores?)(?:\s+rupees)?\b",
        0.8,
    ),
    Rule("rupees_suffix", r"\b\d[\d,]*(?:\.\d+)?\s+rupees\b", 0.8),
]

INDIAN_PHONE_RULES = [
    Rule("in_mobile", r"(?<![\d+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", 0.75),
    Rule(
        "in_landline", r"(?<![\d+])(?:\+91[\s-]?)?0?[1-9]\d{1,3}[\s-]\d{3,4}[\s-]?\d{4}(?!\d)", 0.5
    ),
]

_ADDR_UNIT = (
    r"(?:Flat|Plot|House|Shop|Office|Unit|Door|Bungalow|Apartment|Apt|Block|Wing|Room|Survey"
    r"|Gat|H\.?\s?No|Sr\.?\s?No)"
)
_ADDR_CHUNK = r"(?:[A-Z][A-Za-z.'’-]*|\d+[A-Za-z]*)(?:[ \t]+(?:[A-Z][A-Za-z.'’-]*|\d+[A-Za-z]*|no\.?|of)){0,4}"
_PIN = r"\d{3}\s?\d{3}"
_IN_LOCALITY = (
    r"(?:Marg|Nagar|Colony|Society|Layout|Chowk|Peth|Wadi|Apartments?|Towers?|Residency|Enclave"
    r"|Heights|CHS|Housing\s+Society|Cross\s+Road|Main\s+Road)"
)

INDIAN_ADDRESS_RULES = [
    # Flat 4B, Koregaon Park, Pune 411001
    Rule(
        "unit_address",
        rf"\b{_ADDR_UNIT}\.?\s*(?:No\.?\s*)?[A-Z]?\d+[A-Z]?(?:[/-]\d+[A-Z]?)?(?:,[ \t]*{_ADDR_CHUNK}){{0,5}}(?:[, \t-]+{_PIN}\b)?",
        0.85,
        flags=0,
    ),
    Rule(
        "city_pin",
        rf"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s*[-–,]?\s*(?P<value>{_PIN})\b(?!\s*(?:/|of\b))",
        0.5,
        flags=0,
    ),
    Rule("pin_labelled", rf"\b(?:PIN|pin\s*code|pincode)\s*:?\s*(?P<value>{_PIN})\b", 0.85),
    Rule("locality", rf"\b(?:{_CAP_WORD}[ \t]+){{1,3}}{_IN_LOCALITY}\b", 0.6, flags=0),
]

_COMPANY_SUFFIX = (
    r"(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|Ltd\.?|Limited|LLP|Inc\.?|Corporation|Corp\.?"
    r"|&\s*(?:Co\.?|Sons|Associates)|Associates|Enterprises|Industries|Traders|Builders|Developers"
    r"|Constructions?|Infra(?:structure)?|Realty|Co-?operative\s+(?:Bank|Society)|Bank|Finance"
    r"|Motors|Textiles|Exports|Pharma(?:ceuticals)?|Logistics|Infotech|Technologies|Solutions)"
)
ORGANIZATION_RULES = [
    Rule(
        "company_suffix",
        rf"\b(?:{_CAP_WORD}[ \t]+){{1,5}}{_COMPANY_SUFFIX}(?![A-Za-z])\.?",
        0.8,
        flags=0,
    ),
    # M/s Mehta Builders: "M/s" marks a firm in Indian legal writing
    Rule(
        "messrs",
        rf"\bM/[sS]\.?[ \t]+(?P<value>{_CAP_WORD}(?:[ \t]+{_CAP_WORD}){{0,5}})",
        0.85,
        flags=0,
    ),
]

MATTER_CODENAME_RULES = [
    Rule(
        "project_codename",
        r"\b(?:Project|Operation)\s+(?!(?:Manager|Management|Report|Plan|Team|Lead|Director|Engineer"
        r"|Office|Site|Cost|Phase|Account|Head|Work|Details)\b)[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?\b",
        0.8,
        flags=0,
    ),
]

# Party names in pleadings and intake notes. The role cue itself is not part of the span;
# rules/roles.py reads it again to choose [CLIENT_n], [PARTY_n] or [WITNESS_n].
_ROLE_WORD = (
    r"(?:client|petitioner|respondent|plaintiff|defendant|complainant|accused|appellant|applicant"
    r"|witness|opposing\s+party|opposite\s+party|counterparty|landlord|tenant|drawer|payee|deponent"
    r"|borrower|guarantor|builder)"
)
_HONORIFIC = r"(?:(?:Mr|Mrs|Ms|Dr|Shri|Sri|Smt|Kumari|Adv|Advocate)\.?\s+)"
_NOT_A_NAME = r"(?!(?:The|In|On|At|Has|Had|Was|Is|Will|And|For|To|Of|Name|No|Details)\b)"

LEGAL_NAME_RULES = [
    # "your dispute with Sunanda Khan": the other side of a matter
    Rule(
        "dispute_with",
        rf"\b(?i:dispute|case|matter|complaint|quarrel)[ \t]+(?i:with|against)[ \t]+{_HONORIFIC}?(?!M/[sS]){_NOT_A_NAME}(?P<value>{_NAME})",
        0.7,
        flags=0,
    ),
    Rule(
        "role_name",
        rf"\b(?i:{_ROLE_WORD})(?:\s+(?i:name))?\s*(?::|-|,)?\s+{_HONORIFIC}?{_NOT_A_NAME}(?P<value>{_NAME})",
        0.7,
        flags=0,
    ),
    Rule(
        "indian_honorific",
        rf"\b(?:Shri|Sri|Smt|Kumari|Adv|Advocate)\.?\s+(?P<value>{_NAME})",
        0.75,
        flags=0,
    ),
    Rule("justice", rf"\b(?:Justice|Hon'ble\s+Justice)\s+(?P<value>{_NAME})", 0.75, flags=0),
    # "Rohan Mehta v. Sunita Kale": both sides of a cause title
    Rule("versus_left", rf"(?P<value>{_NAME})\s+(?:v\.|vs\.?|versus)\s+", 0.7, flags=0),
    Rule(
        "versus_right",
        rf"\s(?:v\.|vs\.?|versus)\s+{_HONORIFIC}?(?!M/[sS])(?P<value>{_NAME})",
        0.7,
        flags=0,
    ),
]

LEGAL_OCCUPATIONS = [
    "cardiologist",
    "surgeon",
    "dentist",
    "chartered accountant",
    "advocate",
    "civil engineer",
    "software engineer",
    "shop owner",
    "small business owner",
    "builder",
    "contractor",
    "landlord",
    "government employee",
    "bank manager",
    "retired government officer",
    "professor",
]
# Job titles common in Indian intake notes, matched wherever they appear. Role words that
# are also jobs in some sentences (builder, landlord, contractor, advocate) are left to the
# "is a ..." rule above, which needs the context.
LEGAL_JOBS = [
    "cardiologist",
    "dentist",
    "orthopaedic surgeon",
    "paediatrician",
    "pediatrician",
    "general physician",
    "gynaecologist",
    "radiologist",
    "schoolteacher",
    "school teacher",
    "college lecturer",
    "professor",
    "software engineer",
    "civil engineer",
    "mechanical engineer",
    "chartered accountant",
    "bank manager",
    "insurance agent",
    "shopkeeper",
    "textile trader",
    "restaurant owner",
    "hardware dealer",
    "government clerk",
    "railway employee",
    "truck driver",
    "taxi driver",
    "auto-rickshaw driver",
    "farmer",
    "architect",
    "fashion designer",
    "pharmacist",
    "nurse",
    "journalist",
    "electrician",
    "plumber",
    "retired bank officer",
    "police constable",
]
_NOT_AFTER_DETERMINER = "".join(
    rf"(?<!\b{w} )" for w in ("my", "the", "his", "her", "our", "your", "their", "a", "an")
)
LEGAL_OCCUPATION_RULES = [
    Rule(
        "legal_is_a_occupation",
        rf"\b(?:am|is|was|a|an|as)\s+(?:an?\s+)?{_AGE_GENDER}(?P<value>{_alt(LEGAL_OCCUPATIONS)})s?\b",
        0.7,
    ),
    # not after a possessive or article: "my gynaecologist" is the treating doctor, not the
    # person described ("is a cardiologist" is the rule above's job)
    Rule(
        "legal_job_title",
        rf"\b{_NOT_AFTER_DETERMINER}(?P<value>{_alt(LEGAL_JOBS)})s?\b",
        0.6,
    ),
]

# Indian cities, matched by name: spaCy reads some of them (Kozhikode, Hubballi) as people.
INDIAN_CITIES = [
    "Mumbai",
    "Pune",
    "Nagpur",
    "Nashik",
    "Thane",
    "Aurangabad",
    "Kolhapur",
    "Solapur",
    "Bengaluru",
    "Bangalore",
    "Mysuru",
    "Mysore",
    "Mangaluru",
    "Mangalore",
    "Hubballi",
    "Hubli",
    "Belagavi",
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Tiruchirappalli",
    "Kochi",
    "Ernakulam",
    "Thiruvananthapuram",
    "Kozhikode",
    "Thrissur",
    "Hyderabad",
    "Warangal",
    "Hanamkonda",
    "Ahmedabad",
    "Surat",
    "Vadodara",
    "Rajkot",
    "New Delhi",
    "Delhi",
    "Gurugram",
    "Noida",
    "Kolkata",
    "Howrah",
    "Lucknow",
    "Kanpur",
    "Varanasi",
    "Jaipur",
    "Udaipur",
    "Jodhpur",
    "Panaji",
    "Margao",
    "Bhopal",
    "Indore",
    "Chandigarh",
    "Patna",
    "Bhubaneswar",
    "Guwahati",
]
INDIAN_PLACE_RULES = [
    Rule(
        "indian_city",
        rf"\b(?P<value>{'|'.join(INDIAN_CITIES)})\b(?![ \t]+(?:District|High|Sessions|Family|Labour|Commercial|Civil|Bench)\b)",
        0.6,
        flags=0,
    ),
]


LAYERS: dict[str, list[tuple[str, str, list[Rule]]]] = {
    "spec": [
        ("AgeRecognizer", "AGE", AGE_RULES),
        ("PhysicalRecognizer", "PHYSICAL", PHYSICAL_RULES),
        ("RelationRecognizer", "RELATION", RELATION_RULES),
    ],
    "general": [
        ("NameRecognizer", "NAME", NAME_RULES),
        ("AddressRecognizer", "ADDRESS", ADDRESS_RULES),
        ("LabelledIdRecognizer", "ID_NUMBER", ID_RULES),
        ("OccupationRecognizer", "OCCUPATION", OCCUPATION_RULES),
    ],
    "healthcare": [
        ("FacilityRecognizer", "LOCATION", FACILITY_RULES),
    ],
    "email": [
        ("EmailPhoneRecognizer", "PHONE", EMAIL_PHONE_RULES),
        ("SignatureTitleRecognizer", "OCCUPATION", SIGNATURE_TITLE_RULES),
    ],
    "legal": [
        ("LegalIdRecognizer", "ID_NUMBER", LEGAL_ID_RULES),
        ("FinancialRecognizer", "FINANCIAL", FINANCIAL_RULES),
        ("CaseNumberRecognizer", "CASE_NUMBER", CASE_NUMBER_RULES),
        ("CourtRecognizer", "COURT", COURT_RULES),
        ("MoneyRecognizer", "MONEY", MONEY_RULES),
        ("IndianPhoneRecognizer", "PHONE", INDIAN_PHONE_RULES),
        ("IndianAddressRecognizer", "ADDRESS", INDIAN_ADDRESS_RULES),
        ("CompanyRecognizer", "COMPANY", ORGANIZATION_RULES),
        ("MatterCodenameRecognizer", "MATTER_CODENAME", MATTER_CODENAME_RULES),
        ("LegalNameRecognizer", "NAME", LEGAL_NAME_RULES),
        ("LegalOccupationRecognizer", "OCCUPATION", LEGAL_OCCUPATION_RULES),
        ("IndianPlaceRecognizer", "LOCATION", INDIAN_PLACE_RULES),
    ],
}
ALL_LAYERS: tuple[str, ...] = ("core", *LAYERS)


def custom_recognizers(layers: tuple[str, ...] = ALL_LAYERS) -> list[RegexRecognizer]:
    unknown = set(layers) - set(ALL_LAYERS)
    if unknown:
        raise ValueError(f"unknown layer(s) {sorted(unknown)}; choose from {ALL_LAYERS}")
    return [
        RegexRecognizer(name, entity, rules)
        for layer in LAYERS
        if layer in layers
        for name, entity, rules in LAYERS[layer]
    ]
