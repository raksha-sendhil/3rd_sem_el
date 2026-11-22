"""Agent 1's finance layer.  OWNER: Person B.

Identities, all removed:

* ``ID_NUMBER``: account, card (Luhn), IFSC, PAN, UPI, IBAN / BBAN, MICR, SWIFT / BIC, routing
  and sort codes, customer / loan IDs, SSN, and what opens an account: PIN, security code,
  password, API key, user name.
* ``NAME`` after a form's label, ``ADDRESS`` after a label or ending in a street type, ``PHONE``
  in international or labelled form.
* ``LOCATION`` for a bank, employer or other named organisation. Agent 3 says what kind it is
  ("a public-sector bank", "a company").

Content, marked so it is kept: ``AMOUNT`` and ``PRODUCT``.

Tuned on Gretel's train split (``tools/finance/agent1_check.py``), reported on its test split.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sanitise_finance.rules.money import AMOUNT, is_amount
from sanitise_finance.rules.rule import Rule

_TABLES = Path(__file__).resolve().parents[3] / "data" / "finance" / "ontology"


def _names(table: str) -> str:
    path = _TABLES / f"{table}.json"
    keys = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    words = sorted(keys, key=len, reverse=True)
    return "|".join(r"[\s-]+".join(re.escape(w) for w in re.split(r"[\s-]+", k)) for k in words)


def luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2)
                for i, d in enumerate(reversed(digits)))  # fmt: skip
    return total % 10 == 0


def _not_money(value: str) -> bool:
    return not is_amount(value)


def _secret(value: str) -> bool:
    """A password or key, not the next word of a sentence: it has a digit or a symbol inside."""
    return bool(re.search(r"\d|[^\w\s.,;:!?'\"-]", value)) and _not_money(value)


# between a label and its value: "Account No.: 5010", "PIN (4 digits): 1234", "| PIN | 1234"
_SEP = r"(?:[ \t]*\([^)\n]{0,30}\))?[ \t]*(?:is|was|of|:|#|=|-|\.|\||,)?[ \t:*_\"'(\[>|]*"

# ---------------------------------------------------------------- ID_NUMBER

FINANCE_ID_RULES = [
    Rule(
        "account_number",
        r"\b(?:a/c|acct|acc|account)\.?(?:[ _]?(?:number|no\.?|num|id|#))?" + _SEP
        + r"(?P<value>[xX*]{0,12}\d(?:[ -]?\d){8,17})(?![\w]|[.,]\d)",
        0.95,
        check=_not_money,
    ),
    Rule(
        "account_number_short",  # 6-8 digits only when the label says it is a number
        r"\b(?:a/c|acct|acc|account)\.?[ _]?(?:number|no\.?|num|#)" + _SEP
        + r"(?P<value>[xX*]{0,12}\d(?:[ -]?\d){3,7})(?![\w]|[.,]\d)",
        0.9,
        check=_not_money,
    ),
    Rule("card_number", r"(?<![\w.,-])(?:\d[ -]?){12,18}\d(?![\w-]|[.,]\d)", 0.95, check=luhn),
    Rule(
        "card_number_labelled",  # synthetic and mistyped numbers fail the checksum
        r"\bcard[ _]?(?:number|no\.?|num|#)" + _SEP + r"(?P<value>[\dxX*](?:[ -]*[\dxX*]){7,22})(?!\w)",
        0.9,
        check=lambda v: any(c.isdigit() for c in v),
    ),
    Rule("long_number", r"(?<![\w.,])\d{14,22}(?![\w]|[.,]\d)", 0.7),
    Rule("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.95, flags=0),
    Rule("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.95, flags=0),
    Rule(
        "upi",
        r"(?<![\w.@-])[\w.-]{2,}@(?:upi|ybl|ibl|axl|paytm|ptyes|ptaxis|pthdfc|ptsbi|okaxis|okhdfcbank"
        r"|okicici|oksbi|apl|yapl|sbi|hdfcbank|icici|axisbank|kotak|barodampay|cnrb|pnb|fbl|idfcbank"
        r"|indus|jio|freecharge|airtel)(?![\w-]|\.\w)",
        0.95,
    ),
    Rule("upi_labelled", r"\b(?:UPI(?:[ _]?ID)?|VPA)" + _SEP + r"(?P<value>[\w.-]{2,}@[a-z]{2,20})", 0.9),
    Rule("iban", r"\b[A-Z]{2}\d{2}[A-Z]{4}[A-Z0-9]{6,20}\b|\b[A-Z]{2}\d{2}(?:\d{14,28})\b", 0.95, flags=0),
    Rule(
        "iban_bban_labelled",
        r"\b(?i:IBAN|BBAN)(?![A-Za-z])" + _SEP + r"(?P<value>[A-Z0-9](?:[ ]?[A-Z0-9]){9,33})(?![A-Za-z0-9])",
        0.9,
        flags=0,
        check=lambda v: sum(c.isdigit() for c in v) >= 6,
    ),
    Rule("bban", r"\b[A-Z]{4}\d{14}\b", 0.9, flags=0),
    Rule(
        "micr",
        r"\bMICR(?:[ _]?(?:code|number|no\.?))?" + _SEP + r"(?P<value>\d{3}[ ]?\d{3}[ ]?\d{3})(?!\d)",
        0.95,
    ),
    Rule("micr_line", r"⑆\s?\d{9}\s?⑆", 0.95),
    Rule(
        "routing_sort_code",
        r"\b(?:routing|ABA|sort[ _]code|transit)(?:[ _]?(?:transit[ _]?)?(?:code|number|no\.?))?"
        + _SEP + r"(?P<value>\d{9}|\d\d[ -]?\d\d[ -]?\d\d|\d{5}-\d{3})(?![\w-])",
        0.95,
    ),
    Rule(
        "swift_bic_labelled",
        r"\b(?i:swift|bic)[^\n]{0,30}?\b(?P<value>[A-Z]{6}(?:[A-Z0-9]{2}\d{3}|[A-Z0-9]{2}XXX|[A-Z]\d|\d[A-Z0-9]))\b",
        0.95,
        flags=0,
    ),
    Rule("swift_bic", r"\b[A-Z]{6}[A-Z0-9]{2}\d{3}\b", 0.8, flags=0),
    Rule(
        "customer_loan_id",
        r"\b(?:customer|client|cust|loan|application|applicant|borrower|employee|emp|staff|merchant"
        r"|transaction|txn|mandate|folio|nominee|user|cheque|check|passport|policy|membership"
        r"|driver'?s?[ _]licen[cs]e|driving[ _]licen[cs]e|national[ _]insurance)"
        r"[ _]?(?:id(?:entification)?(?:[ _]?(?:number|no\.?))?|number|no\.?|ref(?:erence)?(?:[ _]?(?:number|no\.?))?|#)"
        r"(?![a-z])" + _SEP + r"(?P<value>(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,}[A-Z0-9])(?![\w]|[.,]\d)",
        0.9,
    ),
    Rule("ssn", r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])", 0.95),
    # "A12 34 5678": how US forms print a specimen or masked SSN in its three boxes
    Rule("ssn_spaced", r"(?<![\w-])[A-Z0-9]\d{2}[ ]{1,3}\d{2}[ ]{1,3}\d{4}(?![\w-])", 0.8, flags=0),
    Rule(
        "ssn_labelled",
        r"\b(?:SSN|social[ _]security(?:[ _]?(?:number|no\.?))?)(?![a-z])" + _SEP
        + r"(?P<value>[A-Z0-9*]{2,3}(?:[ -]?[A-Z0-9*]){4,9})(?!\w)",
        0.9,
        check=lambda v: sum(c.isdigit() for c in v) >= 4,
    ),
    Rule("pin", r"\bpin(?:[ _](?:number|code|for account access))?" + _SEP + r"(?P<value>\d{4,6})(?![\w]|[.,]\d)", 0.9),
    Rule(
        "security_code",
        r"\b(?:security[ _]code|CVV2?|CVC2?)(?:[ _](?:of|is|on the back of my card is|for this transaction is))?"
        + _SEP + r"(?P<value>\d{3,4})(?![\w]|[.,]\d)",
        0.9,
    ),
    Rule(
        "password",
        r"\b(?:password|passcode|passphrase)(?:[ \t]+is)?[ \t]*[:=,]?[ \t>'\"(]*(?P<value>[^\s<>\"]{6,40}?)"
        r"(?=[.,;)'\"]?(?:\s|$|<))",
        0.9,
        check=_secret,
    ),
    Rule(
        "api_key",
        r"\bapi[ _-]?key" + _SEP + r"(?P<value>[A-Za-z0-9][\w.-]{7,}[A-Za-z0-9])",
        0.9,
        check=lambda v: any(c.isdigit() for c in v),
    ),
    Rule(
        "api_key_format",
        r"\b(?:xox[abpr]-[\w-]{10,}|AKIA[A-Z0-9]{16}|sq0atp-[\w-]{20,}|SG\.[\w-]{16,}\.[\w-]{16,}"
        r"|sl\.[A-Za-z0-9]{20,}|SK[A-Za-z0-9]{32,})\b",
        0.9,
        flags=0,
    ),
    Rule("long_key", r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{24,}\b", 0.7),
    Rule(
        "user_name",
        r"\b(?:user[ _]?name|login[ _]?id)" + _SEP + r"(?P<value>[^\W_][\w.-]{3,30})(?<![.])",
        0.85,
        check=lambda v: v.lower() not in {"password", "number", "email", "field", "required"},
    ),
    Rule("ipv6", r"(?<![\w:])[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){7}(?![\w:])", 0.9),
    Rule(
        "coordinates",
        r"(?<![\w.])-?\d{1,3}\.\d{4,}°?[ ]?[NSEW]?[,; ]{1,2}(?:longitude:?[ ]?)?-?\d{1,3}\.\d{4,}°?(?:[ ]?[NSEW]\b)?",
        0.85,
    ),
]  # fmt: skip

# ---------------------------------------------------------------- PHONE

FINANCE_PHONE_RULES = [
    Rule(
        "international",
        r"(?<![\w+])\(?\+\d{1,3}\)?(?:[ .-]?\(\d{1,4}\))?(?:[ .-]?\d{2,5}){2,5}(?!\w)",
        0.8,
        check=lambda v: 9 <= sum(c.isdigit() for c in v) <= 15,
    ),
    Rule("us_bracket", r"(?<![\w(])\(\d{3}\)[ ]?\d{3}[-. ]\d{4}(?!\w)", 0.85),
    Rule("toll_free", r"(?<![\w-])(?:1[-. ])?(?:800|888|877|866|855|0800|1800)[-. ]\d{3}[-. ]?\d{4}(?!\w)", 0.8),
    Rule(
        "phone_labelled",
        r"\b(?:phone|tel|telephone|mobile|cell|fax|contact[ _]?(?:number|no\.?|phone))(?:[ _]?(?:number|no\.?))?"
        + _SEP + r"(?P<value>\+?\d[\d ()./-]{5,20}\d)(?![\w]|[.,]\d)",
        0.85,
        check=lambda v: 7 <= sum(c.isdigit() for c in v) <= 15 and _not_money(v),
    ),
]  # fmt: skip

# ---------------------------------------------------------------- NAME

_WORD = r"(?:[A-ZÀ-ÖØ-Þ](?:[^\W\d_]|['’-](?=[^\W\d_]))*\.?)"
_PARTICLE = r"(?:van|von|de|der|den|da|di|du|del|della|le|la|bin|al)"
_PERSON = rf"{_WORD}(?:,[ ]{_WORD}(?=[ \n]|$))?(?:[ ]{{1,2}}(?:{_PARTICLE}[ ])*{_WORD}){{0,4}}"
_FORM_WORDS = set(
    "account address amount applicant application balance bank borrower business card company "
    "contact credit customer date debit details description email employer employment financial "
    "full holder income information insurance interest lender limit loan mortgage name number "
    "payment phone print printed signature statement total transaction type none n/a na "
    "confidential required here insert first last middle legal".split()
)


def _person(value: str) -> bool:
    words = [w.strip(".,").lower() for w in value.split()]
    return not set(words) & _FORM_WORDS and any(len(w) > 1 for w in words)


FINANCE_NAME_RULES = [
    Rule(
        "name_labelled",
        r"\b(?i:full(?:[ _]legal)?[ _]name|legal[ _]name|(?:first|last|middle|given|family)[ _]?name|surname"
        r"|(?:customer|client|applicant|borrower|co-applicant|employee|taxpayer|insured|policyholder"
        r"|cardholder|card[ _]?holder|account[ _]?holder|account|payee|guarantor|nominee|agent|recipient"
        r"|patient|contact)(?:'s)?[ _]?name"
        r"|account[ _]?holder|card[ _]?holder|policy[ _]?holder|borrower|co-applicant|applicant|guarantor"
        r"|nominee|payee|taxpayer|signature|signed(?:[ ]by)?|agent|attn|prepared[ ]by|approved[ ]by)"
        r"(?:[ ]?\([^)\n]{0,30}\))?[ \t*\"']*[:=|][ \t*_\"'>]*"
        rf"(?P<value>{_PERSON})(?=[ \t]*(?:[^\w \t]|$|\n)|[ \t]{{2,}})",
        0.8,
        flags=re.MULTILINE,
        check=_person,
    ),
    Rule(
        "name_line",  # "Name: Asha Rao", but not "Employer Name: ..." (an organisation)
        r"^[ \t>*#-]*(?:\d+\.[ ]*)?(?i:name|customer|client|user|from|dear|pay)(?:[ \t*]*[:=|][ \t*_\"']*|[ \t]+)"
        rf"(?P<value>{_PERSON})(?=[ \t]*(?:[,(\[*|<!.]|$)|[ \t]+OR[ ]BEARER)",
        0.75,
        flags=re.MULTILINE,
        check=_person,
    ),
]  # fmt: skip

# ---------------------------------------------------------------- ADDRESS

# USPS street suffixes, without those that are ordinary words of finance (Union, Points, Key)
_SUFFIX = (
    "Alley|Avenue|Ave|Bend|Boulevard|Blvd|Branch|Bridge|Brook|Brooks|Burgs|Bypass|Camp|Canyon"
    "|Causeway|Centers|Circle|Circles|Cliff|Cliffs|Close|Club|Common|Commons|Corner|Corners|Course"
    "|Court|Courts|Cove|Coves|Creek|Crescent|Crest|Crossing|Crossroad|Curve|Dale|Dam|Divide|Drive"
    "|Drives|Estate|Estates|Expressway|Extension|Falls|Ferry|Field|Fields|Flats|Ford|Forest|Forge"
    "|Forges|Fork|Forks|Fort|Freeway|Garden|Gardens|Gateway|Glen|Glens|Green|Greens|Grove|Groves"
    "|Harbor|Harbors|Haven|Heights|Highway|Hill|Hills|Hollow|Inlet|Island|Islands|Isle|Junction"
    "|Junctions|Knoll|Knolls|Lake|Lakes|Landing|Lane|Light|Lights|Loaf|Lock|Locks|Lodge|Loop|Mall"
    "|Manor|Manors|Meadow|Meadows|Mews|Mill|Mills|Motorway|Mountain|Mountains|Nagar|Orchard|Oval"
    "|Overpass|Park|Parks|Parkway|Parkways|Passage|Path|Pike|Pine|Pines|Place|Plain|Plains|Plaza"
    "|Port|Ports|Prairie|Radial|Ramp|Ranch|Rapids|Rest|Ridge|Ridges|River|Road|Roads|Route|Row|Run"
    "|Shoal|Shoals|Shore|Shores|Skyway|Spring|Springs|Spur|Square|Squares|Station|Stravenue|Stream"
    "|Street|Streets|Terrace|Throughway|Trace|Trafficway|Trail|Tunnel|Turnpike|Underpass|Valley"
    "|Valleys|Viaduct|View|Views|Village|Villages|Ville|Vista|Walk|Walks|Way|Ways|Well|Wells"
    "|Gasse|Strasse|Straße|Rue|Viale|Borgo|Rd|St|Ln|Dr|Ct|Pkwy|Hwy"
)
_UNIT = r"(?:Apt|Apartment|Suite|Ste|Unit|Flat|Floor|Room|Studio)\.?[ ]?#?[ ]?\d{1,5}[A-Za-z]?"
_CITY = r"[A-ZÀ-Þ][^\W\d_]+(?:[ -][A-ZÀ-Þ][^\W\d_]+){0,2}"
_POSTCODE = r"(?:\d{5,6}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?[ ]?\d[A-Z]{2}|[A-Z]\d[A-Z][ ]?\d[A-Z]\d)"
_NOT_CITY = (
    "Account|Email|Phone|Tel|Date|Amount|Borrower|Lender|Customer|Dear|Name|Total|Loan|Balance"
)

FINANCE_ADDRESS_RULES = [
    Rule(
        "numbered_street",
        r"(?<![\w$£€₹,.])(?:(?i:" + _UNIT + r"),?[ ])?\d{1,6}[A-Za-z]?(?:/\d{1,4})?,?[ ]"
        rf"(?:[A-ZÀ-Þ][\w'’.-]*[ ]){{1,4}}(?i:{_SUFFIX})(?![\w-])\.?"
        rf"(?:,?[ ](?i:{_UNIT}))?(?:,[ ]?{_POSTCODE})?"
        rf"(?:,[ ](?!(?:{_NOT_CITY})\b){_CITY}){{0,2}}(?:,?[ ]{_POSTCODE}\b)?",
        0.85,
        flags=0,
    ),
    Rule(
        "address_labelled",
        r"(?<!IP )(?<!email )(?<!e-mail )\b(?:address(?:[ ](?:at|of|is))?|resident of|residing at|resides at"
        r"|lives at|head office(?:[ ]located)?(?:[ ]at)?|registered office(?:[ ]at)?|place of business(?:[ ]at)?)"
        r"(?![a-z])[ \t*_\"':=|>]{0,6}"
        r"(?P<value>(?=[^\n@|<>]{0,20}\d)[^\n@|<>\"()\[\]*;$£€₹]{6,100}?)"
        r"(?=[ \t]*(?:$|[\n|<>\"()\[\]*;$£€₹])|,?[ ](?:hereinafter|and|with|whose|email|phone|tel)\b"
        r"|[ ]-[ ]|(?<!\bApt)(?<!\bSt)(?<!\bNo)\.(?:[ ]|$))",
        0.85,
        flags=re.IGNORECASE | re.MULTILINE,
        check=lambda v: "://" not in v and not re.fullmatch(r"[\d. :a-f]+", v),
    ),
    Rule("unit", rf"(?<!\w){_UNIT}(?!\w)", 0.7, check=lambda v: "floor" not in v.lower()),
    Rule("ca_postcode", r"\b[A-Z]\d[A-Z][ ]\d[A-Z]\d\b", 0.8, flags=0),
]  # fmt: skip

# ---------------------------------------------------------------- LOCATION: organisations

_LEAD = (
    r"(?!(?:The|Your|Our|My|This|That|Dear|To|From|At|By|With|In|For|And|Of|Issuing|Receiving"
    r"|Beneficiary|Beneficiary's|Sending|Intermediary|Correspondent|Central|Name|Account|An?|Or"
    r"|Between|Contact|Welcome|Thank|Savings|Choosing|Please|Same|Other|Any|Each|Lender|Borrower"
    r"|Payee|Payer|Customer|Online|Mobile|Monthly)\b)"
)
_ORG_WORD = r"[A-Z0-9][\w&'’-]*\.?"
_STRONG = (
    r"Ltd|Limited|Inc|Incorporated|LLC|LLP|L\.L\.C|PLC|Plc|plc|Corp|Corporation|GmbH|Bancorp"
    r"|Holdings|Associates|Pvt|Co\.|S\.A\.|N\.A\.|Bank|BANK|Credit Union|Building Society"
    r"|Insurance Company|Trust Company|Mortgage Company|Finance Company|Trading Company"
    r"|Logistics|Enterprises|Ventures|Securities|Bankers"
)
_ORG_STOP = set(
    "bank company employer lender merchant business name address account details n/a none "
    "information statement insert number email phone required self-employed".split()
)


def _organisation(value: str) -> bool:
    words = [w.strip(".,").lower() for w in value.split()]
    return bool(set(words) - _ORG_STOP) and not any("[" in w for w in words)


_BANKS = _names("banks")
FINANCE_ORG_RULES = [
    Rule(
        "organisation_suffix",
        rf"\b{_LEAD}{_ORG_WORD}(?:(?:[ ]|,[ ](?=(?:Inc|Ltd|LLC|PLC)\b)|[ ](?:and|&|of)[ ]){_ORG_WORD}){{0,5}}?"
        rf"(?:,?[ ](?:{_STRONG})){{1,2}}(?![\w])\.?",
        0.7,
        flags=0,
        check=_organisation,
    ),
    Rule(
        "bank_of",
        rf"\bBank of (?!(?:The|Your|Our)\b){_ORG_WORD}(?:[ ]{_ORG_WORD}){{0,1}}(?<![.])",
        0.7,
        flags=0,
    ),
    Rule(
        "organisation_labelled",
        r"\b(?i:employer(?:'s)?(?:[ _]name)?|current employer|company(?:[ _]name)?|business[ _]name"
        r"|bank[ _]name|lender(?:[ _]name)?|merchant(?:[ _]name)?|organi[sz]ation(?:[ _]name)?"
        r"|issued by|employed (?:by|at|with))[ \t*\"']*(?:[:=|]|(?<=by)|(?<=at)|(?<=with))[ \t*_\"'>]*"
        rf"(?P<value>{_ORG_WORD}(?:[ ](?:and[ ]|&[ ]|of[ ])?{_ORG_WORD}){{0,5}})(?=[ \t]*(?:[^\w \t]|$|\n))",
        0.7,
        flags=re.MULTILINE,
        check=_organisation,
    ),
    Rule(
        "party",  # ', ... (hereinafter "Lender")', 'Acme-Smith, a company organised ...'
        rf"\b{_LEAD}(?P<value>{_ORG_WORD}(?:[ ](?:&[ ]|and[ ])?{_ORG_WORD}){{0,4}}),[ ]an?[ ]"
        r"(?:company|corporation|bank|lender|financial institution|firm)\b",
        0.75,
        flags=0,
    ),
    *([Rule("known_bank", rf"(?<![\w@.-])(?:{_BANKS})(?![\w@-])", 0.8)] if _BANKS else []),
]  # fmt: skip

# ---------------------------------------------------------------- content, kept

_PRODUCTS = _names("products")
PRODUCT_RULES = (
    [Rule("product", rf"(?<![\w-])(?:{_PRODUCTS})s?(?![\w-])", 0.6)] if _PRODUCTS else []
)
AMOUNT_RULES = [Rule("amount", AMOUNT.pattern, 0.6)]

FINANCE_LAYER = [
    ("FinanceIdRecognizer", "ID_NUMBER", FINANCE_ID_RULES),
    ("FinancePhoneRecognizer", "PHONE", FINANCE_PHONE_RULES),
    ("FinanceNameRecognizer", "NAME", FINANCE_NAME_RULES),
    ("FinanceAddressRecognizer", "ADDRESS", FINANCE_ADDRESS_RULES),
    ("OrganisationRecognizer", "LOCATION", FINANCE_ORG_RULES),
    ("AmountRecognizer", "AMOUNT", AMOUNT_RULES),
    ("ProductRecognizer", "PRODUCT", PRODUCT_RULES),
]
