"""The clean test's population: people the pipelines' tables, code and tests have never seen.

The people come from a file written outside this project, ``data/eval_clean/clean_people.json``
(health's person table format). Check it before committing it:

    python tools/eval_suite/population.py data/eval_clean/clean_people.json

Anyone whose name, phone, town or job is already in a table, in the generated corpora or written
in the pipelines' code is dropped. Their records are then written from templates that differ
from the development corpus's. Legal uses the same new people, with legal matters that are in
no vocabulary. The pipeline is never tuned on this population: if it changes after a result has
been seen, the population is no longer clean and needs new people.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PEOPLE_FILE = "data/eval_clean/clean_people.json"
MIN_PEOPLE = 60
FIELDS = ("person_id", "name", "sex", "age", "age_band", "town", "district", "state", "occupation",
          "occupation_group", "conditions", "phone")  # fmt: skip
# matters outside the legal vocabulary
ISSUES = ("motor accident compensation claim", "trademark infringement", "flat possession delay",
          "medical negligence claim", "succession certificate application", "defamation suit",
          "insurance claim repudiation", "copyright infringement", "gratuity recovery",
          "school fee dispute")  # fmt: skip
TABLES = {
    "town": ("data/ontology/places.json", "data/legal/ontology/places.json"),
    "occupation": ("data/ontology/occupations.json", "data/legal/ontology/occupations.json"),
}
DEV_PEOPLE = ("data/corpus/people.json", "data/legal/clients.json", "data/sample/people.json")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def _digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))[-10:]


def known_values(root: Path) -> dict[str, set[str]]:
    """What development has seen, by field."""
    known: dict[str, set[str]] = {f: set() for f in ("name", "phone", "town", "occupation")}
    for field, paths in TABLES.items():
        for path in paths:
            if (root / path).is_file():
                known[field] |= {_norm(k) for k in json.loads((root / path).read_text("utf-8"))}
    for path in DEV_PEOPLE:
        if not (root / path).is_file():
            continue
        for p in json.loads((root / path).read_text("utf-8")):
            known["name"].add(_norm(p.get("name")))
            known["phone"].add(_digits(p.get("phone")))
            known["town"].add(_norm(p.get("town") or p.get("city")))
            known["occupation"].add(_norm(p.get("occupation")))
    literals: set[str] = set()
    for pkg in ("sanitise", "sanitise_legal"):
        for path in (root / "src" / pkg).rglob("*.py"):
            found = re.findall(r"[\"']([^\"'\n]{3,60})[\"']", path.read_text(encoding="utf-8"))
            literals |= {_norm(x) for x in found}
    for field in ("name", "town", "occupation"):
        known[field] |= literals
    return known


def validate(path: Path, root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """(the people who are unseen, why the others were dropped)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON list of people")
    known = known_values(root)
    used: dict[str, set[str]] = {"person_id": set(), "name": set(), "phone": set()}
    kept, dropped = [], []
    for i, row in enumerate(rows):
        who = row.get("person_id") or f"row {i + 1}"
        if lacking := [f for f in FIELDS if not row.get(f)]:
            dropped.append(f"{who}: no {', '.join(lacking)}")
            continue
        conditions = row["conditions"]
        conditions = [conditions] if isinstance(conditions, str) else list(conditions)
        phone = _digits(row["phone"])
        why = [f"{f} '{row[f]}' is not new" for f in ("name", "town", "occupation")
               if _norm(row[f]) in known[f]]  # fmt: skip
        if len(phone) != 10 or phone in known["phone"]:
            why.append("phone is not a new 10-digit number")
        keys = {"person_id": _norm(who), "name": _norm(row["name"]), "phone": phone}
        why += [f"{f} repeated in the file" for f, v in keys.items() if v in used[f]]
        if not str(row["age"]).isdigit():
            why.append("age is not a number")
        if why:
            dropped.append(f"{who}: {'; '.join(why)}")
            continue
        for field, value in keys.items():
            used[field].add(value)
        low = int(row["age"]) // 10 * 10
        kept.append({**{f: row[f] for f in FIELDS}, "age": int(row["age"]),
                     "age_band": f"{low}-{low + 9}", "conditions": conditions,
                     "phone": f"+91 {phone[:5]} {phone[5:]}"})  # fmt: skip
    return kept, dropped


# ---------------------------------------------------------------- the records

HEALTH = [
    "TELECONSULT LOG\nCaller: {name} ({age}, {occupation})\nCalling from: {town}, {district}\n"
    "Number: {phone}\nConcern: {condition} for {weeks} weeks, worse at night.\n"
    "Nurse's note: asked the caller to keep a symptom diary and come for a review in a week.",
    "Good evening, {name} here from {town}. My {condition} flared up again after long days of "
    "work as a {occupation}. I turned {age} this year. Should I change my treatment? My mobile "
    "is {phone}.",
    "Referral\nThank you for seeing {name}, aged {age}, who lives at {town} in {state} and earns "
    "a living as a {occupation}. Problem: {condition}, not settled after {weeks} weeks of first "
    "line treatment. Please advise on further tests. The patient can be reached on {phone}.",
    "Patient: I'm {name}, {age} years old, writing from {town}. I have had {condition} for "
    "{weeks} weeks and it makes my job as a {occupation} hard. You may ring {phone}.\n"
    "Doctor: Please rest, continue the current medicines and book an examination if it is no "
    "better in five days.",
    "REGISTRATION SLIP\nName: {name}\nSex: {sex}\nAge: {age}\nTown: {town}\nDistrict: {district}\n"
    "Work: {occupation}\nMobile: {phone}\nReason for visit: follow-up of {condition}.",
    "Clinic SMS thread\n{phone}: sir this is {name}, reports for my {condition} are ready, can I "
    "come tomorrow from {town}?\nClinic: Yes, bring the earlier prescriptions and do not stop "
    "the medicines before the review.",
    "Discharge advice for {name} ({sex}, {age} yrs) of {town}, {district}. Diagnosis: "
    "{condition}. The patient, a {occupation}, should avoid strain for {weeks} weeks, take the "
    "medicines as written and return at once if symptoms worsen. Phone on file {phone}.",
    "Subject: sick note request\nHello team, {name} ({occupation}, {town}) has asked for a sick "
    "note of {weeks} weeks because of {condition}. They are {age}. Please call them on {phone} "
    "once the doctor signs.",
]
LEGAL = [
    "NEW MATTER\nClient: {name}, {age}, {occupation}\nResides: {town}, {district}, {state}\n"
    "Mobile: {phone}; e-mail: {email}; PAN: {pan}\nMatter: {issue} against {opposing_party}.\n"
    "To collect: the agreement, all notices and payment proofs.",
    "Dear {name},\nOn your {issue} ({case_number}) before the {court}: we advise sending a "
    "written demand to {opposing_party} within fifteen days and keeping every receipt. We will "
    "write to you at {email} once they reply.\nRegards",
    "Status sheet, {case_number}\n{name}, a {occupation} from {town} aged {age}, sued "
    "{opposing_party} at the {court} over a {issue}. Outcome so far: interim relief granted. "
    "Next step: file the evidence affidavit. Phone {phone}.",
    "Hello, I am {name} from {town}, working as a {occupation}. I need help with a {issue} "
    "involving {opposing_party}. I am {age}. Please reach me on {phone} or {email}. What papers "
    "should I bring?",
    "Hearing note\nCourt: {court}\nCase: {case_number}, filed {filing_date}\nParty: {name} "
    "(PAN {pan}) v. {opposing_party}\nSubject: {issue}, claim Rs. {claim_amount}.\n"
    "Advice given: consider mediation before the next date.",
    "Call record: {phone}. {name} ({occupation}, {district}) asked about the {issue}. Advised "
    "that the limitation period must be checked first and that originals should not be handed "
    "to {opposing_party}.",
]
SEX = {"f": "female", "m": "male"}
FIRMS = ("Quenvar", "Istrello", "Bhavyansh", "Oritek", "Zelmora", "Navkiran", "Tessaro", "Ulvani")
TRADES = ("Motors", "Housing", "Assurance", "Publishers", "Academy", "Healthcorp", "Retail")


def _doc(text: str, pid: str, template: int) -> dict:
    return {"doc_id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "source": "clean-population", "template": f"clean_{template}", "person_id": pid,
            "text": text}  # fmt: skip


def _records(row: dict, values: dict, templates: list[str], i: int, rng) -> list[dict]:
    picks = rng.sample(range(len(templates)), 3 if i % 2 == 0 else 2)
    return [_doc(templates[t].format(**values), row["person_id"], t) for t in picks]


def health(people: list[dict], seed: int = 2027) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    table, docs = [], []
    for i, p in enumerate(people):
        values = {**p, "condition": " and ".join(p["conditions"]), "weeks": rng.randint(2, 7),
                  "sex": SEX.get(str(p["sex"]).lower(), p["sex"])}  # fmt: skip
        mine = _records(p, values, HEALTH, i, rng)
        docs += mine
        table.append({**p, "doc_ids": [d["doc_id"] for d in mine]})
    return table, docs


def legal(people: list[dict], seed: int = 2027) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    matter = sorted({p["conditions"][0] for p in people})
    table, docs = [], []
    for i, p in enumerate(people):
        issue = ISSUES[matter.index(p["conditions"][0]) % len(ISSUES)]
        year = rng.randint(2020, 2025)
        surname = re.sub(r"[^a-z]", "", p["name"].lower().split()[-1]) or "client"
        row = {
            **{k: v for k, v in p.items() if k not in ("person_id", "town", "conditions")},
            "client_id": p["person_id"], "city": p["town"], "legal_issue": issue, "region": "",
            "rare_issue": False, "team": "general", "court": f"{p['district']} District Court",
            "case_number": f"O.S. No. {rng.randint(101, 9899)}/{year}",
            "filing_date": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "claim_amount": rng.randint(8, 240) * 10000,
            "email": f"{surname}.{i}{rng.randint(10, 99)}@mailbox.example",
            "pan": "".join(rng.choices("BCDFGHJKLM", k=3)) + "P" + surname[0].upper()
            + f"{rng.randint(1000, 9999)}" + rng.choice("QRSTUVWX"),
            "opposing_party": f"{rng.choice(FIRMS)} {rng.choice(TRADES)} Pvt Ltd",
            "witness": None, "matter_codename": None, "pattern": "clean",
        }  # fmt: skip
        values = {**row, "town": p["town"], "issue": issue}
        mine = _records(p, values, LEGAL, i, rng)
        docs += mine
        table.append({**row, "doc_ids": [d["doc_id"] for d in mine]})
    return table, docs


def build(domain: str, people_file: Path, out: Path, root: Path = ROOT) -> dict:
    """``out/people.json`` and ``out/docs/*.json``; the counts, for the report."""
    if not people_file.is_file():
        raise SystemExit(
            f"the clean test needs new people at {people_file}: save the file there, commit and "
            'push it, and run again. SCALE = "quick" does not need it.'
        )
    people, dropped = validate(people_file, root)
    if len(people) < MIN_PEOPLE:
        raise SystemExit(
            f"only {len(people)} of the people are unseen, {MIN_PEOPLE} are needed. Dropped:\n"
            + "\n".join(dropped[:40])
        )
    table, docs = (health if domain == "health" else legal)(people)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (out / "docs" / f"{doc['doc_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")  # fmt: skip
    (out / "people.json").write_text(json.dumps(table, indent=1, ensure_ascii=False),
                                     encoding="utf-8")  # fmt: skip
    return {"people": len(people), "people_dropped": len(dropped), "documents": len(docs),
            "people_file_sha256": hashlib.sha256(people_file.read_bytes()).hexdigest()}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", type=Path)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    people, dropped = validate(args.file)
    for line in dropped:
        print("dropped", line)
    pairs: dict[tuple[str, str], int] = {}
    for p in people:
        key = (p["conditions"][0], p["state"])
        pairs[key] = pairs.get(key, 0) + 1
    targets = sum(n for n in pairs.values() if n >= 2)
    print(f"{len(people)} unseen people ({len(dropped)} dropped); {targets} can be attack targets")
    if len(people) < MIN_PEOPLE or targets < 40:
        raise SystemExit("not enough: replace the dropped people with new ones and check again")
    print("OK: commit and push this file as data/eval_clean/clean_people.json")


if __name__ == "__main__":
    main()
