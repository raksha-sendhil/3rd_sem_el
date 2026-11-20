"""Person A: bank customers. Fills data/finance/bank/ and data/finance/people.json.

    python tools/finance/bank.py --limit 400

Sources (both external, no data written by us):
* CFPB Consumer Complaint Database, real complaint narratives (public, US government):
  Hugging Face mirror ``CFPB/consumer-finance-complaints`` (rows API:
  https://datasets-server.huggingface.co/rows?dataset=CFPB/consumer-finance-complaints&config=default&split=train&offset=0&length=100)
  or the bulk CSV at https://files.consumerfinance.gov/ccdb/complaints.csv.zip.
  Keep rows that have a narrative; CFPB already masks names as "XXXX" (say so in the paper).
* AgamiAI Indian bank statements (synthetic, Apache-2.0): https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements
  ``train/<type>/<id>.pdf`` + ``<id>.json`` (holder name, address, account number, IFSC,
  branch, transactions). The holders are the population for the multi-question attacks.

What to write (see docs/finance_plan.md, "Contracts"; use ``common.Folder``):
1. complaints -> ``f.doc(text, meta={...})`` (indexed) and ``f.doc(text, heldout=True)``
   (about a third), each <= 1,900 characters (skip longer ones);
2. utility questions about complaint content (``f.utility``), never naming a company,
   person or place; the reference answer copied from the text;
3. statements -> one text document per holder in ``population/`` (``folder="population"``,
   ``person_id=...``): the statement's header and a few transactions as text;
4. ``write_people([...])``: one row per holder: person_id, name, town, district, state,
   bank, branch, account_type, account_number, phone (if any), conditions=[product or
   account type], doc_ids. Leave out fields the source does not have (no age or job).
5. Check the population overlaps: several holders should share state, bank and product,
   or the attacks single everyone out at once. Post the counts on WhatsApp.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Allow running from the tools/finance/ directory on Kaggle
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common import Folder, write_people  # noqa: E402


# ---------------------------------------------------------------------------
# CFPB complaint narratives
# ---------------------------------------------------------------------------

# Template utility questions about complaint content.
# Rules: never name a person, company, account number, place or date.
# Reference answers must be short spans copied from the complaint text.
_UTILITY_TEMPLATES: list[tuple[str, str]] = [
    # (question template, field or heuristic that extracts the reference)
    # These are applied per complaint after checking the reference is in the text.
    # Actual questions and references are built in _make_utility_questions().
]

# Product-specific question stubs (content-only, no names/places).
# Each entry: (question, regex or key to find the reference in the text)
_PRODUCT_QUESTIONS: dict[str, list[tuple[str, str]]] = {
    "credit card": [
        ("What action did the customer request the bank take regarding the disputed charge?",
         r"(?i)request(?:ed|ing)?\s+(?:the\s+bank\s+)?(?:to\s+)?(\w[\w\s,]{3,60})\b"),
        ("What type of charge did the customer dispute on the account?",
         r"(?i)(late fee|interest charge|annual fee|foreign transaction fee|overlimit fee|cash advance fee|balance transfer fee)"),
        ("What response did the bank give to the customer's complaint?",
         r"(?i)(?:bank|company|issuer)\s+(?:said|responded|stated|told|informed)[^.]{0,10}(?:that\s+)?([^.]{5,80})\b"),
    ],
    "mortgage": [
        ("What problem did the customer report about the mortgage payment processing?",
         r"(?i)(payment\s+(?:was\s+)?(?:not\s+)?(?:posted|applied|credited|processed|returned|rejected)[^.]{0,60})\b"),
        ("What did the customer ask the servicer to do about the escrow account?",
         r"(?i)(?:asked?|requested?|want(?:ed)?)[^.]{0,20}(?:escrow)[^.]{0,60}"),
        ("What error did the customer identify in the loan statement?",
         r"(?i)(incorrect|wrong|error(?:eous)?|inaccurate|miscalculated)[^.]{0,60}"),
    ],
    "student loan": [
        ("What issue did the customer report with the repayment plan?",
         r"(?i)(repayment\s+plan|income[-\s]driven|IDR|IBR|PAYE)[^.]{0,80}"),
        ("What was the customer's complaint about the loan servicer's communication?",
         r"(?i)(?:did\s+not|never|failed\s+to)\s+(notify|inform|communicate|contact|respond)[^.]{0,60}"),
    ],
    "bank account or service": [
        ("What action did the bank take on the customer's account without prior notice?",
         r"(?i)(froze|closed|blocked|suspended|restricted|placed\s+a\s+hold)[^.]{0,60}"),
        ("What fee did the customer dispute on the account?",
         r"(?i)(overdraft fee|maintenance fee|monthly fee|service charge|wire transfer fee|ATM fee)[^.]{0,60}"),
        ("What did the customer ask the bank to do to resolve the issue?",
         r"(?i)(?:asked?|requested?|want(?:ed)?)[^.]{0,80}"),
    ],
    "debt collection": [
        ("What did the customer claim the collector did that was not permitted?",
         r"(?i)(called|contacted|harassed|threatened|reported)[^.]{0,80}"),
        ("What was the nature of the debt the customer said was being collected incorrectly?",
         r"(?i)(medical|credit card|student|auto|mortgage|personal loan)[^.]{0,30}debt"),
    ],
    "consumer loan": [
        ("What problem did the customer describe with the loan account?",
         r"(?i)(double charge|duplicate payment|extra deduction|EMI[^.]{0,40}debit(?:ed)?)[^.]{0,60}"),
        ("What did the customer ask the lender to do about the payment error?",
         r"(?i)(?:asked?|requested?|want(?:ed)?|demand(?:ed)?)[^.]{0,80}"),
    ],
    "payday loan": [
        ("What did the customer say about the loan's interest rate?",
         r"(?i)(interest rate|APR|annual percentage rate)[^.]{0,60}"),
    ],
    "prepaid card": [
        ("What problem did the customer report with funds on the prepaid card?",
         r"(?i)(funds|balance|money|amount)[^.]{0,60}(?:missing|unavailable|blocked|not\s+accessible|frozen)[^.]{0,40}"),
    ],
    "money transfer": [
        ("What went wrong with the transfer according to the customer?",
         r"(?i)(transfer|wire|remittance)[^.]{0,40}(?:failed|delayed|sent\s+to\s+wrong|returned|not\s+received)[^.]{0,60}"),
    ],
}

_GENERIC_QUESTIONS: list[tuple[str, str]] = [
    ("What was the main action the customer asked the financial institution to take?",
     r"(?i)(?:asked?|requested?|want(?:ed)?|demand(?:ed)?|urge[sd]?)[^.]{5,80}"),
    ("What outcome did the customer report after contacting the institution?",
     r"(?i)(?:however|but|unfortunately|still|yet)[^.]{0,10}(?:no|not|never|did\s+not)[^.]{5,60}"),
    ("What specific problem with the account did the customer describe?",
     r"(?i)(account\s+(?:was\s+)?(?:charged|debited|blocked|closed|frozen|overdrawn)[^.]{0,60})"),
    ("What did the customer say the institution failed to do?",
     r"(?i)(?:failed\s+to|did\s+not|never|refused\s+to)\s+(\w[\w\s]{3,60})\b"),
    ("What type of error did the customer identify in the billing statement?",
     r"(?i)(incorrect|wrong|error(?:eous)?|inaccurate|unauthorized)[^.]{0,60}"),
]


def _first_match(text: str, pattern: str) -> str | None:
    """Return the first non-empty capture group from the pattern, or None."""
    m = re.search(pattern, text)
    if not m:
        return None
    groups = [g for g in m.groups() if g and g.strip()]
    if groups:
        return groups[0].strip()
    return m.group(0).strip() if m.group(0).strip() else None


def _make_utility_questions(text: str, product: str) -> list[tuple[str, str]]:
    """Return (question, reference) pairs for one complaint text."""
    product_key = product.lower() if product else ""
    # match the closest product category
    matched_key = next(
        (k for k in _PRODUCT_QUESTIONS if k in product_key or product_key in k),
        None,
    )
    candidates: list[tuple[str, str]] = []
    if matched_key:
        candidates = _PRODUCT_QUESTIONS[matched_key]
    candidates = candidates + _GENERIC_QUESTIONS

    seen_questions: set[str] = set()
    out: list[tuple[str, str]] = []
    for question, pattern in candidates:
        if question in seen_questions:
            continue
        ref = _first_match(text, pattern)
        if ref and len(ref) >= 5 and ref.lower() in text.lower():
            out.append((question, ref))
            seen_questions.add(question)
        if len(out) >= 3:
            break
    return out


def _normalise_product(raw: str | None) -> str:
    """Map CFPB's product strings to the nearest short label."""
    if not raw:
        return "bank account or service"
    p = raw.lower()
    if "credit card" in p or "credit reporting" in p:
        return "credit card"
    if "mortgage" in p or "home loan" in p:
        return "mortgage"
    if "student" in p:
        return "student loan"
    if "auto" in p or "vehicle" in p:
        return "consumer loan"
    if "payday" in p:
        return "payday loan"
    if "prepaid" in p:
        return "prepaid card"
    if "money transfer" in p or "remittance" in p or "wire" in p:
        return "money transfer"
    if "debt collection" in p:
        return "debt collection"
    if "consumer loan" in p or "installment" in p or "personal loan" in p:
        return "consumer loan"
    return "bank account or service"


def fetch_complaints(f: Folder, limit: int) -> None:
    """Download CFPB complaint narratives and write docs + utility questions.

    Tries the HuggingFace datasets library first; falls back to the bulk CSV if
    the library is not available or the dataset is not cached.
    """
    print("Fetching CFPB complaints...")
    rows = _load_cfpb_rows(limit * 3)  # over-fetch to filter
    if not rows:
        print("  WARNING: no CFPB rows fetched; check network or dataset access.")
        return

    product_counts: Counter = Counter()
    target_per_product = max(4, limit // 10)
    indexed_count = 0
    heldout_count = 0
    utility_count = 0
    heldout_target = limit // 3

    for row in rows:
        if indexed_count + heldout_count >= limit + heldout_target:
            break

        narrative: str = (
            row.get("Consumer complaint narrative")
            or row.get("narrative")
            or row.get("consumer_complaint_narrative")
            or ""
        ).strip()
        if not narrative or len(narrative) > 1900:
            continue
        # Skip if no substantive text
        if len(narrative.split()) < 15:
            continue

        product_raw = (
            row.get("Product") or row.get("product") or ""
        )
        product = _normalise_product(product_raw)
        issue = (row.get("Issue") or row.get("issue") or "").strip()
        state = (row.get("State") or row.get("state") or "").strip()

        # Balance products: skip if this product is already over-represented
        if product_counts[product] >= target_per_product and len(product_counts) > 2:
            continue
        product_counts[product] += 1

        meta = {"product": product, "issue": issue, "state": state}

        is_heldout = heldout_count < heldout_target and (
            (indexed_count + heldout_count) % 3 == 0
        )
        if is_heldout:
            did = f.doc(narrative, meta=meta, heldout=True)
            if did:
                heldout_count += 1
        else:
            did = f.doc(narrative, meta=meta)
            if did:
                indexed_count += 1
                # Utility questions: only from indexed docs, reference must be in text
                if utility_count < limit:
                    pairs = _make_utility_questions(narrative, product)
                    for question, reference in pairs:
                        f.utility(question, reference=reference, doc_text=narrative)
                        utility_count += 1

    print(
        f"  CFPB: {indexed_count} indexed, {heldout_count} held-out, "
        f"{utility_count} utility questions. Products: {dict(product_counts)}"
    )


def _load_cfpb_rows(limit: int) -> list[dict]:
    """Return up to ``limit`` CFPB complaint rows, with network access."""
    # Try HuggingFace datasets library (available on Kaggle)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "CFPB/consumer-finance-complaints",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        rows = []
        for row in ds:
            if len(rows) >= limit:
                break
            rows.append(row)
        if rows:
            print(f"  Loaded {len(rows)} rows via HuggingFace datasets (streaming).")
            return rows
    except Exception as e:
        print(f"  datasets library unavailable or error ({e}); trying rows API...")

    # Fall back to the HuggingFace rows API (JSON, no library needed)
    try:
        import urllib.request

        PAGE = 100
        rows = []
        offset = 0
        base = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=CFPB/consumer-finance-complaints"
            "&config=default&split=train"
        )
        while len(rows) < limit:
            url = f"{base}&offset={offset}&length={PAGE}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            batch = [item["row"] for item in data.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
        print(f"  Loaded {len(rows)} rows via rows API.")
        return rows[:limit]
    except Exception as e:
        print(f"  rows API error: {e}; trying bulk CSV (slow)...")

    # Last resort: bulk CSV
    try:
        import io
        import urllib.request
        import zipfile

        url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
        print("  Downloading bulk CSV (may be large)...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            zdata = resp.read()
        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as cf:
                import csv

                reader = csv.DictReader(io.TextIOWrapper(cf, encoding="utf-8"))
                rows = []
                for row in reader:
                    if len(rows) >= limit:
                        break
                    rows.append(row)
        print(f"  Loaded {len(rows)} rows from bulk CSV.")
        return rows
    except Exception as e:
        print(f"  Bulk CSV error: {e}")
        return []


# ---------------------------------------------------------------------------
# AgamiAI Indian bank statement population
# ---------------------------------------------------------------------------

# Indian states and their districts for inference when AgamiAI only gives a city
_STATE_HINTS: dict[str, tuple[str, str]] = {
    # city -> (district, state)
    "mumbai": ("Mumbai City", "Maharashtra"),
    "pune": ("Pune", "Maharashtra"),
    "nagpur": ("Nagpur", "Maharashtra"),
    "nashik": ("Nashik", "Maharashtra"),
    "aurangabad": ("Aurangabad", "Maharashtra"),
    "delhi": ("New Delhi", "Delhi"),
    "new delhi": ("New Delhi", "Delhi"),
    "gurugram": ("Gurugram", "Haryana"),
    "gurgaon": ("Gurugram", "Haryana"),
    "noida": ("Gautam Buddha Nagar", "Uttar Pradesh"),
    "faridabad": ("Faridabad", "Haryana"),
    "bengaluru": ("Bengaluru Urban", "Karnataka"),
    "bangalore": ("Bengaluru Urban", "Karnataka"),
    "mysuru": ("Mysuru", "Karnataka"),
    "mysore": ("Mysuru", "Karnataka"),
    "hubli": ("Dharwad", "Karnataka"),
    "mangaluru": ("Dakshina Kannada", "Karnataka"),
    "mangalore": ("Dakshina Kannada", "Karnataka"),
    "hyderabad": ("Hyderabad", "Telangana"),
    "secunderabad": ("Hyderabad", "Telangana"),
    "warangal": ("Warangal", "Telangana"),
    "chennai": ("Chennai", "Tamil Nadu"),
    "coimbatore": ("Coimbatore", "Tamil Nadu"),
    "madurai": ("Madurai", "Tamil Nadu"),
    "salem": ("Salem", "Tamil Nadu"),
    "tiruchirappalli": ("Tiruchirappalli", "Tamil Nadu"),
    "tiruchirapalli": ("Tiruchirappalli", "Tamil Nadu"),
    "kolkata": ("Kolkata", "West Bengal"),
    "howrah": ("Howrah", "West Bengal"),
    "jaipur": ("Jaipur", "Rajasthan"),
    "jodhpur": ("Jodhpur", "Rajasthan"),
    "udaipur": ("Udaipur", "Rajasthan"),
    "ahmedabad": ("Ahmedabad", "Gujarat"),
    "surat": ("Surat", "Gujarat"),
    "vadodara": ("Vadodara", "Gujarat"),
    "rajkot": ("Rajkot", "Gujarat"),
    "lucknow": ("Lucknow", "Uttar Pradesh"),
    "kanpur": ("Kanpur", "Uttar Pradesh"),
    "agra": ("Agra", "Uttar Pradesh"),
    "varanasi": ("Varanasi", "Uttar Pradesh"),
    "patna": ("Patna", "Bihar"),
    "bhopal": ("Bhopal", "Madhya Pradesh"),
    "indore": ("Indore", "Madhya Pradesh"),
    "bhubaneswar": ("Khordha", "Odisha"),
    "cuttack": ("Cuttack", "Odisha"),
    "chandigarh": ("Chandigarh", "Chandigarh"),
    "amritsar": ("Amritsar", "Punjab"),
    "ludhiana": ("Ludhiana", "Punjab"),
    "kochi": ("Ernakulam", "Kerala"),
    "thiruvananthapuram": ("Thiruvananthapuram", "Kerala"),
    "kozhikode": ("Kozhikode", "Kerala"),
    "guwahati": ("Kamrup Metropolitan", "Assam"),
    "raipur": ("Raipur", "Chhattisgarh"),
    "dehradun": ("Dehradun", "Uttarakhand"),
    "shimla": ("Shimla", "Himachal Pradesh"),
    "jammu": ("Jammu", "Jammu & Kashmir"),
    "srinagar": ("Srinagar", "Jammu & Kashmir"),
    "ranchi": ("Ranchi", "Jharkhand"),
    "dharwad": ("Dharwad", "Karnataka"),
}


def _infer_location(city: str) -> tuple[str, str, str]:
    """Return (town, district, state) from a city string."""
    key = city.lower().strip()
    if key in _STATE_HINTS:
        district, state = _STATE_HINTS[key]
        return city.strip().title(), district, state
    # Fallback: use city as both town and district, state unknown
    return city.strip().title(), city.strip().title(), "India"


def _build_statement_text(holder: dict) -> str:
    """Format one AgamiAI statement JSON record as a <= 1900-char text document."""
    name = holder.get("name", "Account Holder")
    address = holder.get("address", "")
    account_no = holder.get("account_number") or holder.get("accountNumber") or ""
    ifsc = holder.get("ifsc") or holder.get("IFSC") or ""
    branch = holder.get("branch", "")
    bank = holder.get("bank") or holder.get("bank_name") or "Bank"
    account_type = (
        holder.get("account_type")
        or holder.get("accountType")
        or holder.get("type")
        or "savings"
    )
    phone = holder.get("phone") or holder.get("mobile") or holder.get("contact") or ""
    transactions: list[dict] = holder.get("transactions") or []

    # Build city from address if available
    city = ""
    if address:
        # Try last comma-separated component before the pin
        parts = [p.strip() for p in address.split(",")]
        # find city-like token (not a number)
        for part in reversed(parts):
            if part and not re.match(r"^\d", part):
                city = part
                break

    town, district, state = _infer_location(city or "India")

    # Header
    lines = [
        "Statement of account.",
        f"Account holder: {name}, {town}, {state}.",
        f"Bank: {bank}, {branch} branch.",
    ]
    if account_no:
        lines.append(f"Account number {account_no}.")
    if ifsc:
        lines.append(f"IFSC: {ifsc}.")
    if account_type:
        lines.append(f"Account type: {account_type}.")
    if phone:
        lines.append(f"Contact: {phone}.")

    # Transactions: include first 3-5 to stay under 1,900 chars
    if transactions:
        lines.append("Recent transactions:")
        for txn in transactions[:5]:
            date = txn.get("date") or txn.get("Date") or ""
            desc = txn.get("description") or txn.get("narration") or txn.get("Description") or ""
            amount = txn.get("amount") or txn.get("Amount") or ""
            txn_type = txn.get("type") or txn.get("Type") or txn.get("cr_dr") or ""
            if desc or amount:
                parts_t = []
                if date:
                    parts_t.append(str(date))
                if desc:
                    parts_t.append(str(desc)[:60])
                if amount:
                    parts_t.append(f"Rs. {amount}")
                if txn_type:
                    parts_t.append(str(txn_type).upper())
                lines.append("  " + " | ".join(parts_t) + ".")

    text = " ".join(" ".join(lines).split())  # collapse whitespace
    # Trim to 1900 chars if still too long
    if len(text) > 1900:
        text = text[:1897] + "..."
    return text


def fetch_population(f: Folder, limit: int) -> None:
    """Download AgamiAI Indian Bank Statements and write population docs + people.json."""
    print("Fetching AgamiAI Indian Bank Statements...")
    records = _load_agamai_records(limit)
    if not records:
        print("  WARNING: no AgamiAI records loaded; check network or dataset access.")
        return

    people: list[dict] = []
    seen_account: set[str] = set()

    for i, holder in enumerate(records):
        name = (holder.get("name") or "").strip()
        if not name:
            continue
        account_no = (
            holder.get("account_number") or holder.get("accountNumber") or ""
        ).strip()
        # Deduplicate by account number
        if account_no and account_no in seen_account:
            continue
        if account_no:
            seen_account.add(account_no)

        person_id = f"b{i + 1:03d}"
        text = _build_statement_text(holder)

        did = f.doc(text, folder="population", person_id=person_id)
        if not did:
            continue  # too long (shouldn't happen after trimming)

        # Build person row
        address = holder.get("address", "")
        city = ""
        if address:
            parts = [p.strip() for p in address.split(",")]
            for part in reversed(parts):
                if part and not re.match(r"^\d", part):
                    city = part
                    break
        town, district, state = _infer_location(city or "India")

        bank = holder.get("bank") or holder.get("bank_name") or ""
        branch = holder.get("branch", "")
        ifsc = holder.get("ifsc") or holder.get("IFSC") or ""
        account_type = (
            holder.get("account_type")
            or holder.get("accountType")
            or holder.get("type")
            or "savings"
        )
        phone = holder.get("phone") or holder.get("mobile") or holder.get("contact") or ""

        person: dict = {
            "person_id": person_id,
            "name": name,
            "town": town,
            "district": district,
            "state": state,
            "bank": bank,
            "branch": branch,
            "conditions": [account_type],
            "account_number": account_no,
            "doc_ids": [did],
        }
        if phone:
            person["phone"] = phone
        if ifsc:
            person["ifsc"] = ifsc
        people.append(person)

    write_people(people)

    # Overlap check: groups sharing state + bank + account_type
    _report_overlap(people)


def _report_overlap(people: list[dict]) -> None:
    """Print overlap stats: how many holders share state + bank + account_type."""
    groups: Counter = Counter(
        (p.get("state", ""), p.get("bank", ""), (p.get("conditions") or [""])[0])
        for p in people
    )
    group_members = {k: v for k, v in groups.items() if v >= 2}
    n_in_groups = sum(group_members.values())
    print(
        f"\n  === Overlap check (state + bank + account_type) ===\n"
        f"  {len(people)} holders total.\n"
        f"  {len(group_members)} groups with >= 2 members, covering {n_in_groups} holders.\n"
        f"  Top groups:"
    )
    for (state, bank, acct_type), count in sorted(
        group_members.items(), key=lambda x: -x[1]
    )[:10]:
        print(f"    {state} | {bank} | {acct_type}: {count} holders")
    if n_in_groups < 10:
        print(
            "  WARN: fewer than 10 holders in overlapping groups.\n"
            "  Consider writing a small synthetic population for the multi-question metrics."
        )


def _load_agamai_records(limit: int) -> list[dict]:
    """Return up to ``limit`` AgamiAI statement JSON records."""
    # Try HuggingFace datasets library
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "AgamiAI/Indian-Bank-Statements",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        records = []
        for row in ds:
            if len(records) >= limit:
                break
            records.append(row)
        if records:
            print(f"  Loaded {len(records)} records via HuggingFace datasets (streaming).")
            return records
    except Exception as e:
        print(f"  datasets library error ({e}); trying rows API...")

    # Fall back to the HuggingFace rows API
    try:
        import urllib.request

        PAGE = 100
        records = []
        offset = 0
        base = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=AgamiAI/Indian-Bank-Statements"
            "&config=default&split=train"
        )
        while len(records) < limit:
            url = f"{base}&offset={offset}&length={PAGE}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            batch = [item["row"] for item in data.get("rows", [])]
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
        print(f"  Loaded {len(records)} records via rows API.")
        return records[:limit]
    except Exception as e:
        print(f"  rows API error: {e}")
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=400,
                    help="Target number of CFPB complaints to index (default 400).")
    ap.add_argument("--pop-limit", type=int, default=150,
                    help="Number of AgamiAI statement holders (default 150).")
    ap.add_argument("--skip-complaints", action="store_true",
                    help="Skip CFPB (useful when re-running only the population).")
    ap.add_argument("--skip-population", action="store_true",
                    help="Skip AgamiAI (useful when re-running only complaints).")
    args = ap.parse_args()

    f = Folder("bank")
    f.reset()

    if not args.skip_complaints:
        fetch_complaints(f, args.limit)
    if not args.skip_population:
        fetch_population(f, args.pop_limit)

    f.save()


if __name__ == "__main__":
    main()
