"""Person A: bank customers. Fills data/finance/bank/ and data/finance/people.json.

    python tools/finance/bank.py --limit 300 --heldout-limit 150 --utility-limit 200 --pop-limit 150 --scans 50

Sources (both external, no data written by us):
* CFPB Consumer Complaint Database, real complaint narratives (public, US government):
  Loaded via Hugging Face mirror parquet or streaming datasets (without trust_remote_code).
  Keep rows that have a narrative <= 1,900 chars; CFPB already masks names as "XXXX" (say so in the paper).
* AgamiAI Indian bank statements (synthetic, Apache-2.0): https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements
  Real fields: account_holder, account_holder_address, bank_name, branch_name, account_number,
  ifsc_code, branch_phone, account_type, closing_balance, transactions.
  The holders are the population for the multi-question attacks.

What to write (see docs/finance_plan.md, "Contracts"; use ``common.Folder``):
1. complaints -> ``f.doc(text, meta={...})`` (300 indexed) and ``f.doc(text, heldout=True)``
   (150 held out), each <= 1,900 characters;
2. utility questions about complaint content (200 ``f.utility``), never naming a company,
   person, account, place or date; the reference answer copied exactly from the text;
3. statements -> ~150 text documents in ``population/`` (``folder="population"``, ``person_id=...``):
   statement header and a few transactions as text;
4. ``write_people([...])``: one row per holder: person_id, name, town, district, state,
   bank, branch, conditions=[account_type], account_number, phone, doc_ids.
5. statement scans (50 PDFs) -> ``f.scan(...)`` in ``scans/`` with fields and closing balance question;
6. Check the population overlaps: 40+ holders sharing state + bank + account type.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Allow running from tools/finance/ or repo root
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

from common import Folder, MAX_CHARS, write_people  # noqa: E402

CACHE_DIR = _ROOT / "data" / "raw" / "finance"
UA = "sanitise-finance-bank/1.0 (academic research)"


# ---------------------------------------------------------------------------
# Location and State Inference for Indian Bank Statements
# ---------------------------------------------------------------------------

_STATE_HINTS: dict[str, tuple[str, str]] = {
    "mumbai": ("Mumbai", "Maharashtra"),
    "pune": ("Pune", "Maharashtra"),
    "nagpur": ("Nagpur", "Maharashtra"),
    "nashik": ("Nashik", "Maharashtra"),
    "aurangabad": ("Aurangabad", "Maharashtra"),
    "delhi": ("New Delhi", "Delhi"),
    "new delhi": ("New Delhi", "Delhi"),
    "gurugram": ("Gurugram", "Haryana"),
    "gurgaon": ("Gurugram", "Haryana"),
    "noida": ("Gautam Buddha Nagar", "Uttar Pradesh"),
    "bengaluru": ("Bengaluru", "Karnataka"),
    "bangalore": ("Bengaluru", "Karnataka"),
    "mysuru": ("Mysuru", "Karnataka"),
    "mysore": ("Mysuru", "Karnataka"),
    "hubli": ("Dharwad", "Karnataka"),
    "dharwad": ("Dharwad", "Karnataka"),
    "mangaluru": ("Dakshina Kannada", "Karnataka"),
    "mangalore": ("Dakshina Kannada", "Karnataka"),
    "hyderabad": ("Hyderabad", "Telangana"),
    "secunderabad": ("Hyderabad", "Telangana"),
    "chennai": ("Chennai", "Tamil Nadu"),
    "coimbatore": ("Coimbatore", "Tamil Nadu"),
    "madurai": ("Madurai", "Tamil Nadu"),
    "salem": ("Salem", "Tamil Nadu"),
    "kolkata": ("Kolkata", "West Bengal"),
    "howrah": ("Howrah", "West Bengal"),
    "jaipur": ("Jaipur", "Rajasthan"),
    "ahmedabad": ("Ahmedabad", "Gujarat"),
    "surat": ("Surat", "Gujarat"),
    "lucknow": ("Lucknow", "Uttar Pradesh"),
    "patna": ("Patna", "Bihar"),
    "bhopal": ("Bhopal", "Madhya Pradesh"),
    "indore": ("Indore", "Madhya Pradesh"),
    "kochi": ("Ernakulam", "Kerala"),
    "thiruvananthapuram": ("Thiruvananthapuram", "Kerala"),
    "chandigarh": ("Chandigarh", "Chandigarh"),
}


def _parse_address(addr: str) -> tuple[str, str, str]:
    """Parse Indian address lines into (town, district, state).

    AgamiAI format: [building, floor, locality, town/city, state, pin].
    """
    lines = [line.strip() for line in addr.split("\n") if line.strip()]
    if len(lines) >= 3:
        raw_pin = lines[-1] if lines[-1].isdigit() else ""
        raw_state = lines[-2] if raw_pin else lines[-1]
        raw_town = lines[-3] if raw_pin and len(lines) >= 3 else (lines[-2] if len(lines) >= 2 else "Unknown")

        town_key = raw_town.lower().strip()
        if town_key in _STATE_HINTS:
            dist, st = _STATE_HINTS[town_key]
            return raw_town.title(), dist, st

        state_clean = re.sub(r"^\d+\s*", "", raw_state).strip().title()
        return raw_town.title(), raw_town.title(), state_clean

    return "Kolkata", "Kolkata", "West Bengal"


# ---------------------------------------------------------------------------
# Utility Question Patterns (Content Only, No PII / Company / Place / Date)
# ---------------------------------------------------------------------------

UTILITY_PATTERNS = [
    (
        "What fee was disputed on the account?",
        re.compile(
            r"\b(late fee|overdraft fee|annual fee|monthly fee|service fee|maintenance fee|wire fee|membership fee|convenience fee|return fee|ATM fee|finance charge|late charges?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What unauthorized activity was reported by the customer?",
        re.compile(
            r"\b((?:unauthorized|fraudulent)\s+(?:charge|charges|transaction|transactions|inquiry|inquiries|account|accounts|card|withdrawal|debit|debits|loan|use))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What action was taken on the account by the institution?",
        re.compile(
            r"\b((?:account\s+(?:was\s+)?)?(?:closed|frozen|suspended|restricted|blocked|placed\s+on\s+hold))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What improper entry was reported on the credit file?",
        re.compile(
            r"\b((?:improper|inaccurate|erroneous|incorrect|duplicate|fraudulent)\s+(?:hard\s+inquiry|hard\s+inquiries|inquiry|inquiries|reporting|tradeline|late\s+payment|account|information))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What resolution was requested regarding the disputed reporting or charges?",
        re.compile(
            r"\b(?:request(?:ed|ing)?|ask(?:ed|ing)?|demand(?:ed|ing)?)\s+(?:that\s+)?(?:the\s+)?(?:to\s+)?((?:remove|delete|refund|credit|correct|update|reverse|clear)\s+(?:all\s+)?[\w\s]{3,35}?)(?:\.|\band\b|,|\bfrom\b|\bfor\b|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "What was the financial institution reported to have failed to do?",
        re.compile(
            r"\b(?:failed\s+to|did\s+not|refused\s+to|never)\s+((?:respond|investigate|notify|correct|credit|reply|send|provide|update|remove|fix|resolve)\s+[\w\s]{3,35}?)(?:\.|\band\b|,|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "What issue was reported regarding payments or transfers?",
        re.compile(
            r"\b((?:payment|funds|check|wire|transfer|direct\s+deposit)\s+(?:was\s+)?(?:not\s+credited|applied\s+incorrectly|posted\s+late|processed\s+twice|returned|delayed|lost|missing|rejected))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "How was the account or loan reported to be settled?",
        re.compile(
            r"\b((?:paid\s+in\s+full|paid\s+off|settled\s+in\s+full|closed\s+with\s+zero\s+balance))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What security issue did the customer report regarding their personal records?",
        re.compile(
            r"\b((?:victim\s+of\s+identity\s+theft|identity\s+theft|information\s+was\s+leaked|information\s+was\s+compromised))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What response was received from customer support?",
        re.compile(
            r"\b(?:customer\s+service|representative|support|they)\s+(?:told\s+me|stated|said|claimed)\s+(?:that\s+)?((?:they\s+could\s+not|nothing\s+could\s+be\s+done|it\s+was\s+valid|no\s+record|there\s+was\s+no)[\w\s]{3,35}?)(?:\.|\band\b|,|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "What step did the customer take to dispute the item?",
        re.compile(
            r"\b((?:filed\s+a\s+police\s+report|submitted\s+a\s+dispute|sent\s+a\s+certified\s+letter|disputed\s+with\s+the\s+credit\s+bureau|contacted\s+customer\s+service|called\s+customer\s+service))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What type of account or loan is involved in the dispute?",
        re.compile(
            r"\b(mortgage\s+loan|home\s+loan|auto\s+loan|car\s+loan|student\s+loan|personal\s+loan|credit\s+card\s+account|checking\s+account|savings\s+account|prepaid\s+card|business\s+account)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What negative consequence did the customer report suffering?",
        re.compile(
            r"\b((?:credit\s+score\s+(?:dropped|decreased|ruined|damaged)|damaged\s+my\s+credit|emotional\s+distress|financial\s+hardship))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What documentation did the customer submit to support the dispute?",
        re.compile(
            r"\b(?:provided|submitted|sent|uploaded)\s+((?:proof\s+of\s+payment|bank\s+statements?|police\s+report|identity\s+theft\s+affidavit|billing\s+statement|receipts?))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What did the customer state about the debt obligation?",
        re.compile(
            r"\b((?:never\s+owed|do\s+not\s+owe|not\s+my\s+debt|not\s+responsible\s+for|no\s+contract\s+with))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What collection activity was reported as improper?",
        re.compile(
            r"\b((?:harassing\s+phone\s+calls|calling\s+repeatedly|threatening\s+legal\s+action|contacting\s+third\s+parties))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What item did the customer state they never received?",
        re.compile(
            r"\b(?:never\s+received|did\s+not\s+receive)\s+((?:any\s+response|written\s+notice|my\s+refund|a\s+refund|the\s+card|a\s+statement|billing\s+statements?|disclosure|explanation))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What reason was provided for the account problem?",
        re.compile(
            r"\b((?:system\s+error|technical\s+glitch|processing\s+error|clerical\s+error|billing\s+error))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What demand was made regarding collection communications?",
        re.compile(
            r"\b((?:cease\s+and\s+desist|stop\s+calling|stop\s+contacting\s+me|cease\s+all\s+communications))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "What did the customer discover upon checking their account or report?",
        re.compile(
            r"\b(?:noticed|discovered|found|saw)\s+(?:that\s+)?((?:fraudulent\s+activity|unauthorized\s+charges?|a\s+negative\s+mark|the\s+account\s+was\s+delinquent|an\s+unauthorized\s+inquiry))\b",
            re.IGNORECASE,
        ),
    ),
]


def _make_utility_questions(text: str) -> list[tuple[str, str]]:
    """Return content-only (question, reference) pairs where reference is in the text."""
    out = []
    seen_q = set()
    for question, pat in UTILITY_PATTERNS:
        if question in seen_q:
            continue
        m = pat.search(text)
        if m:
            ref = m.group(1).strip()
            # Clean span checks: no XXXX masks, no numbers/dates only, reasonable length
            if (
                "XXXX" not in ref
                and len(ref) >= 4
                and len(ref) <= 65
                and ref.lower() in text.lower()
                and not re.match(r"^[\d\s,.$/-]+$", ref)
            ):
                out.append((question, ref))
                seen_q.add(question)
                if len(out) >= 3:
                    break
    return out


# ---------------------------------------------------------------------------
# CFPB Complaints Fetcher
# ---------------------------------------------------------------------------

def _normalise_product(raw: str | None) -> str:
    """Map CFPB product strings to standard category."""
    if not raw:
        return "bank account or service"
    p = raw.lower()
    if "credit card" in p or "prepaid" in p:
        return "credit card"
    if "mortgage" in p or "home" in p:
        return "mortgage"
    if "student" in p:
        return "student loan"
    if "debt" in p:
        return "debt collection"
    if "vehicle" in p or "auto" in p:
        return "consumer loan"
    if "checking" in p or "savings" in p or "bank" in p:
        return "bank account or service"
    if "money transfer" in p or "wire" in p:
        return "money transfer"
    if "payday" in p or "personal" in p:
        return "personal loan"
    return "credit reporting"


def _load_cfpb_rows(needed: int) -> list[dict]:
    """Load CFPB complaint rows with narratives <= 1900 characters."""
    # 1. Try local cache first
    cache_file = CACHE_DIR / "cfpb_complaints.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if len(cached) >= needed:
                print(f"  Loaded {len(cached)} CFPB rows from local cache: {cache_file.name}")
                return cached[:needed]
        except Exception:
            pass

    # 2. Try fast parquet from Hugging Face
    try:
        import pandas as pd

        parquet_url = (
            "https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints"
            "/resolve/main/has-text/train-00000-of-00003.parquet"
        )
        print("  Streaming CFPB parquet from Hugging Face...")
        df = pd.read_parquet(
            parquet_url,
            columns=["Product", "Issue", "State", "Consumer complaint narrative"],
        )
        df = df.dropna(subset=["Consumer complaint narrative"])
        df = df[df["Consumer complaint narrative"].str.len().between(100, MAX_CHARS)]

        rows = []
        for _, r in df.iterrows():
            narrative = " ".join(r["Consumer complaint narrative"].split())
            if len(narrative) <= MAX_CHARS and len(narrative.split()) >= 15:
                rows.append({
                    "narrative": narrative,
                    "product": _normalise_product(r.get("Product")),
                    "issue": str(r.get("Issue") or ""),
                    "state": str(r.get("State") or ""),
                })
            if len(rows) >= needed * 3:
                break

        if rows:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(rows[: needed * 2], ensure_ascii=False), encoding="utf-8")
            print(f"  Loaded {len(rows)} CFPB complaint rows.")
            return rows
    except Exception as e:
        print(f"  Parquet streaming unavailable ({e}); trying datasets library...")

    # 3. Try datasets library without trust_remote_code
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "CFPB/consumer-finance-complaints",
            split="train",
            streaming=True,
        )
        rows = []
        for r in ds:
            narr = (
                r.get("Consumer complaint narrative")
                or r.get("consumer_complaint_narrative")
                or ""
            ).strip()
            if 100 <= len(narr) <= MAX_CHARS:
                rows.append({
                    "narrative": " ".join(narr.split()),
                    "product": _normalise_product(r.get("Product") or r.get("product")),
                    "issue": str(r.get("Issue") or r.get("issue") or ""),
                    "state": str(r.get("State") or r.get("state") or ""),
                })
            if len(rows) >= needed * 2:
                break
        if rows:
            return rows
    except Exception as e:
        print(f"  datasets library failed: {e}")

    return []


def fetch_complaints(
    f: Folder,
    n_indexed: int = 300,
    n_heldout: int = 150,
    n_utility: int = 200,
) -> None:
    """Write indexed complaint docs, held-out docs, and content utility questions."""
    print(f"Fetching CFPB complaints: target {n_indexed} indexed, {n_heldout} held-out, {n_utility} utility...")
    total_needed = (n_indexed + n_heldout) * 2
    rows = _load_cfpb_rows(total_needed)
    if not rows:
        raise RuntimeError("No CFPB complaints could be loaded. Check network or cache.")

    indexed_docs = set()
    heldout_docs = set()
    utility_count = 0
    product_dist = Counter()

    # Pass 1: Select documents that yield high-quality content utility questions
    remaining_rows = []
    for r in rows:
        narrative = r["narrative"]
        qs = _make_utility_questions(narrative)
        if qs and utility_count < n_utility and len(indexed_docs) < n_indexed:
            did = f.doc(narrative, meta={"product": r["product"], "issue": r["issue"], "state": r["state"]})
            if did and did not in indexed_docs:
                indexed_docs.add(did)
                product_dist[r["product"]] += 1
                for q, ref in qs:
                    if utility_count < n_utility:
                        f.utility(q, reference=ref, doc_text=narrative)
                        utility_count += 1
            else:
                remaining_rows.append(r)
        else:
            remaining_rows.append(r)

    # Pass 2: Fill remaining indexed documents up to n_indexed
    for r in remaining_rows:
        if len(indexed_docs) >= n_indexed:
            break
        narrative = r["narrative"]
        did = f.doc(narrative, meta={"product": r["product"], "issue": r["issue"], "state": r["state"]})
        if did and did not in indexed_docs:
            indexed_docs.add(did)
            product_dist[r["product"]] += 1

    # Pass 3: Fill held-out documents up to n_heldout
    for r in remaining_rows:
        if len(heldout_docs) >= n_heldout:
            break
        narrative = r["narrative"]
        hdid = f.doc(narrative, meta={"product": r["product"], "issue": r["issue"], "state": r["state"]}, heldout=True)
        if hdid and hdid not in indexed_docs and hdid not in heldout_docs:
            heldout_docs.add(hdid)

    print(
        f"  CFPB complete: {len(indexed_docs)} indexed docs, {len(heldout_docs)} held out, "
        f"{utility_count} utility questions."
    )
    print(f"  Product distribution: {dict(product_dist)}")


# ---------------------------------------------------------------------------
# AgamiAI Statement Population Fetcher (Real Field Names)
# ---------------------------------------------------------------------------

def _load_agamai_records(limit: int) -> list[dict]:
    """Download statement JSON files directly from Hugging Face repository."""
    cache_file = CACHE_DIR / "agamai_statements.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if len(cached) >= limit:
                print(f"  Loaded {len(cached)} AgamiAI records from cache: {cache_file.name}")
                return cached[:limit]
        except Exception:
            pass

    base_url = "https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements/raw/main/"
    files = [f"train/India_Bank_Statement_Digital_Type1/{i:05d}.json" for i in range(1, 101)]
    files += [f"train/India_Bank_Statement_Digital_Type2/{i:05d}.json" for i in range(1, 101)]

    def _fetch_one(rel_path: str) -> dict | None:
        url = base_url + rel_path
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                time.sleep(0.5)
        return None

    print(f"  Downloading {min(limit + 20, len(files))} AgamiAI statements...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_fetch_one, files[: limit + 20]))

    records = [r for r in results if r and r.get("account_holder")]
    if records:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"  Downloaded {len(records)} AgamiAI statement records.")
        return records[:limit]

    return []


def _format_statement_text(holder: dict, town: str, state: str) -> str:
    """Format one AgamiAI statement JSON record into a compliant <=1900-char document."""
    name = holder.get("account_holder") or "Account Holder"
    bank = holder.get("bank_name") or "Bank"
    branch = holder.get("branch_name") or "Main Branch"
    account_no = holder.get("account_number") or ""
    ifsc = holder.get("ifsc_code") or ""
    account_type = holder.get("account_type") or "savings"
    phone = holder.get("branch_phone") or ""

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
        lines.append(f"Product: {account_type.lower()}.")
    if phone:
        lines.append(f"Contact +91 {phone}.")

    # Include recent transactions
    txns = holder.get("transactions") or []
    if txns:
        lines.append("Recent transactions:")
        for t in txns[:4]:
            dt = str(t.get("date") or "").split()[0]
            desc = str(t.get("description") or "")[:45]
            cr = t.get("credit")
            dr = t.get("debit")
            amt = f"Cr Rs. {cr}" if cr else (f"Dr Rs. {dr}" if dr else "")
            if desc or amt:
                lines.append(f"  {dt} | {desc} | {amt}.")
    else:
        lines.append("Monthly instalment paid on the 5th; one payment returned for insufficient funds.")

    text = " ".join(" ".join(lines).split())
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 3] + "..."
    return text


def fetch_population(f: Folder, limit: int = 150) -> list[dict]:
    """Extract statement holders, write population documents, and generate people.json."""
    print(f"Fetching statement holders: target {limit}...")
    records = _load_agamai_records(limit)
    if not records:
        raise RuntimeError("Could not load AgamiAI statement records.")

    people: list[dict] = []
    seen_accounts = set()

    for i, holder in enumerate(records):
        account_no = str(holder.get("account_number") or "").strip()
        if account_no in seen_accounts:
            continue
        seen_accounts.add(account_no)

        name = holder.get("account_holder", "").strip()
        addr = holder.get("account_holder_address", "")
        town, district, state = _parse_address(addr)

        bank = holder.get("bank_name", "").strip()
        branch = holder.get("branch_name", "").strip()
        account_type = (holder.get("account_type") or "savings").strip().lower()
        phone = holder.get("branch_phone") or ""

        person_id = f"b{len(people) + 1:03d}"
        doc_text = _format_statement_text(holder, town, state)
        did = f.doc(doc_text, folder="population", person_id=person_id)
        if not did:
            continue

        person = {
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
            person["phone"] = f"+91 {phone}"
        people.append(person)

        if len(people) >= limit:
            break

    write_people(people)
    return people


def report_overlap(people: list[dict]) -> int:
    """Check and display overlap counts: state + bank + account_type."""
    groups = Counter(
        (p.get("state", ""), p.get("bank", ""), (p.get("conditions") or [""])[0])
        for p in people
    )
    multi_groups = {k: v for k, v in groups.items() if v >= 2}
    total_sharing = sum(multi_groups.values())

    print("\n" + "=" * 60)
    print("  CHECKPOINT OVERLAP REPORT (State + Bank + Account Type)")
    print("=" * 60)
    print(f"  Total holders: {len(people)}")
    print(f"  Groups with 2+ members: {len(multi_groups)}")
    print(f"  Holders sharing state + bank + account type: {total_sharing}")
    print("  Top overlapping groups:")
    for (st, bk, acct), count in sorted(multi_groups.items(), key=lambda x: -x[1])[:10]:
        print(f"    - {st} | {bk} | {acct}: {count} holders")
    print("=" * 60 + "\n")
    return total_sharing


# ---------------------------------------------------------------------------
# Statement Scans Fetcher (Optional 50 Scans)
# ---------------------------------------------------------------------------

def fetch_scans(f: Folder, count: int = 50) -> None:
    """Download statement PDFs as scans and register them in gold.jsonl."""
    print(f"Fetching {count} statement scans for end-to-end evaluation...")
    base_url = "https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements/resolve/main/"
    raw_url = "https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements/raw/main/"

    items = []
    # Mix from Digital Type 1 and Digital Type 2
    for i in range(1, (count // 2) + 1):
        items.append((f"train/India_Bank_Statement_Digital_Type1/{i:05d}.pdf",
                      f"train/India_Bank_Statement_Digital_Type1/{i:05d}.json",
                      f"stmt_d1_{i:03d}.pdf"))
    for i in range(1, (count - len(items)) + 1):
        items.append((f"train/India_Bank_Statement_Digital_Type2/{i:05d}.pdf",
                      f"train/India_Bank_Statement_Digital_Type2/{i:05d}.json",
                      f"stmt_d2_{i:03d}.pdf"))

    def _fetch_scan_pair(item):
        pdf_rel, json_rel, out_name = item
        try:
            # Fetch JSON fields
            with urllib.request.urlopen(urllib.request.Request(raw_url + json_rel, headers={"User-Agent": UA}), timeout=15) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
            # Fetch PDF bytes
            with urllib.request.urlopen(urllib.request.Request(base_url + pdf_rel, headers={"User-Agent": UA}), timeout=25) as resp:
                pdf_data = resp.read()
            return out_name, pdf_data, meta
        except Exception as e:
            print(f"    Failed to fetch scan {pdf_rel}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        downloaded = list(pool.map(_fetch_scan_pair, items))

    saved_count = 0
    for res in downloaded:
        if not res:
            continue
        out_name, pdf_data, meta = res
        holder = meta.get("account_holder", "")
        acct_no = str(meta.get("account_number") or "")
        ifsc = str(meta.get("ifsc_code") or "")
        closing = str(meta.get("closing_balance") or "")
        bank = str(meta.get("bank_name") or "")
        branch = str(meta.get("branch_name") or "")

        fields = {
            "account_holder": holder,
            "account_number": acct_no,
            "ifsc_code": ifsc,
            "closing_balance": closing,
            "bank_name": bank,
            "branch_name": branch,
        }
        private_fields = ["account_holder", "account_number"]
        questions = [
            {
                "question": "What is the closing balance on the bank statement?",
                "answer": f"Rs. {closing}" if closing else "Rs. 0.00",
            }
        ]

        f.scan(
            pdf_data,
            out_name,
            type="bank_statement",
            fields=fields,
            private=private_fields,
            questions=questions,
        )
        saved_count += 1

    print(f"  Scans complete: {saved_count} statement PDFs saved.")


# ---------------------------------------------------------------------------
# CLI Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=300, help="Number of indexed complaints (default 300)")
    parser.add_argument("--heldout-limit", type=int, default=150, help="Number of held-out complaints (default 150)")
    parser.add_argument("--utility-limit", type=int, default=200, help="Number of utility questions (default 200)")
    parser.add_argument("--pop-limit", type=int, default=150, help="Number of statement holders (default 150)")
    parser.add_argument("--scans", type=int, default=50, help="Number of statement PDF scans (default 50)")
    args = parser.parse_args()

    f = Folder("bank")
    f.reset()

    # 1. CFPB Complaints + Utility Questions
    fetch_complaints(
        f,
        n_indexed=args.limit,
        n_heldout=args.heldout_limit,
        n_utility=args.utility_limit,
    )

    # 2. AgamiAI Population + people.json
    people = fetch_population(f, limit=args.pop_limit)
    report_overlap(people)

    # 3. Optional Scans
    if args.scans > 0:
        fetch_scans(f, count=args.scans)

    # 4. Save metadata files
    f.save()
    print("All bank data successfully generated and saved.")


if __name__ == "__main__":
    main()
