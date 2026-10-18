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
    Rule(
        "pronoun_is_n",
        r"\b(?:i\s+am|i'm|im|he\s+is|she\s+is|he's|she's|is\s+now|turned"
        r"|(?:father|mother|dad|mom|mum|son|daughter|husband|wife|brother|sister|grandfather"
        r"|grandmother|child|baby|patient)\s+is)\s+(?P<value>\d{1,3})\b" + _NOT_A_QUANTITY,
        0.6,
    ),
    Rule("decade", rf"\b(?:my|his|her|their)\s+(?:early\s+|mid[\s-]|late\s+)?{_DECADE}", 0.7),
    Rule(
        "died_at_n",
        r"\b(?:died|passed\s+away|deceased|expired)\b[^.\n]{0,40}?\bat\s+(?:the\s+age\s+of\s+)?(?P<value>\d{1,3})\b"
        + _NOT_A_QUANTITY,
        0.6,
    ),
    Rule("ordinal_birthday", r"\b\d{1,3}(?:st|nd|rd|th)\s+birthday\b", 0.8),
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


LAYERS: dict[str, list[tuple[str, str, list[Rule]]]] = {
    "spec": [
        ("AgeRecognizer", "AGE", AGE_RULES),
        ("PhysicalRecognizer", "PHYSICAL", PHYSICAL_RULES),
        ("RelationRecognizer", "RELATION", RELATION_RULES),
    ],
    "general": [
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
