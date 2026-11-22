"""Build data/legal/real/indian300/: 300 real Indian judgments and orders, each checked
automatically, for testing Agents 1-3.

OWNER: Person C. Sources: the open datasets of the Supreme Court of India and of the Indian High
Courts on the AWS Open Data Registry (CC-BY-4.0), which hold the courts' own PDFs and case
metadata. Re-running with the same seed gives the same documents.

    pip install pandas pyarrow pypdf
    python tools/legal/build_indian_corpus.py            # downloads into data/raw/ (ignored)

Each document is an excerpt of at most 1,800 characters (one part for the agents) from the
start of the judgment: the cause title and the opening facts, page furniture removed. A
document is kept only if all of these hold:

* the PDF has an English text layer that reads as words (no scans, no Hindi-only orders);
* it is not about a sexual offence, a child or a minor (Indian law restricts naming them);
* at least one party who is a person (from the court's own metadata, not from our agents)
  appears in the excerpt: these names, plus phone numbers, emails, PAN, Aadhaar-style numbers,
  vehicle registrations and S/o-W/o-D/o names found by pattern, are its ``must_hide`` list;
* at least one legal term (a section, an Act, or a word such as bail or eviction) appears:
  these are its ``must_keep`` list.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "indian_judgments"
OUT = ROOT / "data" / "legal" / "real" / "indian300"
SC_BUCKET = "https://indian-supreme-court-judgments.s3.amazonaws.com"
HC_BUCKET = "https://indian-high-court-judgments.s3.amazonaws.com"
LICENCE = "CC-BY-4.0 (dataset); judgments are public records"
MAX_CHARS = 1800

SC_YEARS = {2023: 30, 2024: 30, 2025: 30}
# (short name, court code, bench, documents)
HIGH_COURTS = [
    ("kar", "29_3", "karnataka_bng_old", 24),
    ("del", "7_26", "dhcdb", 24),
    ("bom", "27_1", "hcaurdb", 24),
    ("ker", "32_4", "highcourtofkerala", 24),
    ("mad", "33_10", "hc_cis_mas", 24),
    ("cal", "19_16", "calcutta_appellate_side", 24),
    ("tel", "36_29", "taphc", 22),
    ("guj", "24_17", "gujarathc", 22),
    ("gau", "18_6", "asghccis", 22),
]
HC_YEAR = 2024

SENSITIVE = re.compile(
    r"\b(POCSO|rape|sexual(ly)?|prosecutrix|minor|minors|juvenile|child|children|girl|"
    r"boy|daughter-in-law dowry death|acid attack|HIV|AIDS|transgender|adoption|"
    r"custody of the|376|354-?A|354-?D|498-?A|304-?B)\b",
    re.I,
)
SKIP_DISPOSAL = re.compile(r"non.?prosecution|default|withdrawn|not pressed|abated|transfer", re.I)
ORG_WORDS = re.compile(
    r"\b(STATE|UNION|GOVT|GOVERNMENT|COMMISSIONER|COMMISSION|OFFICER|DIRECTOR|LTD|LIMITED|PVT|"
    r"PRIVATE|BANK|CORPORATION|COMPANY|CO\.|INSURANCE|MUNICIPAL|MUNICIPALITY|BOARD|AUTHORITY|"
    r"DEPARTMENT|DEPT|COLLECTOR|POLICE|STATION|PS|M/S|LLP|TRUST|SOCIETY|UNIVERSITY|COLLEGE|"
    r"INDIA|CBI|ENFORCEMENT|DIRECTORATE|REGISTRAR|TAHSILDAR|TAHASILDAR|MANAGER|SECRETARY|"
    r"PANCHAYAT|COUNCIL|CORP|ASSOCIATION|OFFICIAL|LIQUIDATOR|INSPECTOR|CHIEF|ENGINEER|"
    r"DIVISIONAL|DISTRICT|REVENUE|TRIBUNAL|COURT|CUSTOMS|TAX|INCOME|GST|ASSESSING|NATIONAL|"
    r"CENTRAL|HOSPITAL|SCHOOL|PUBLIC|PROSECUTOR|PRINCIPAL|EXECUTIVE|DEPUTY|ASSISTANT|"
    r"SUPERINTENDENT|SUB|ENTERPRISES|INDUSTRIES|TRADERS|INFRA|FINANCE|SERVICES|LABORATORIES|"
    r"HOSPITALITY|CONSTRUCTION|CONSTRUCTIONS|DEVELOPERS|BUILDERS|ELECTRICITY|TRANSPORT|"
    r"RAILWAY|RAILWAYS|PORT|AIRPORTS|TELANGANA|KERALA|KARNATAKA|ASSAM|GUJARAT|MAHARASHTRA|"
    r"DELHI|BENGAL|NCT|TAMIL|NADU|LOKAYUKTHA|LOKAYUKTA|VIGILANCE|ANTI|BUREAU|AGENCY|NIA|"
    r"CUSTODIAN|HOUSING|DEVELOPMENT|INSTITUTE|ACADEMY|FOUNDATION|TEMPLE|DEVASWOM|WAKF|"
    r"CHURCH|MOSQUE|MANDIR|MATH|MUTT|EMPLOYEES|WORKERS|STAFF|COOPERATIVE|CO-OPERATIVE)\b",
    re.I,
)
HONORIFIC = re.compile(
    r"^(SRI|SHRI|SMT|SMT\.|MR|MRS|MS|DR|KUM|KUMARI|MISS|MASTER|LATE|SRI\.|SHRI\.|MR\.|MRS\.|MS\.|"
    r"DR\.|THIRU|TMT|SELVI|ADVOCATE)\.?\s+",
    re.I,
)
PARTY_TAIL = re.compile(
    r"\s*(&|\bAND)\s*(ORS|ANR|OTHERS|ANOTHER|\d+\s*(ORS|OTHERS))\b\.?.*$|"
    r"\s*\((DEAD|DECEASED|D)\).*$|\s+(DEAD|DECEASED)\b.*$|"
    r"\s+(THR|THROUGH|THRU)\.?\s+.*$|\s+ETC\.?$|\s+AND\s+\d+\s+OTHERS.*$|[,.;:\s]+$",
    re.I,
)
STATUTE = re.compile(
    r"\b(?:Section|Sec\.|u/s\.?|Article|Order)[ \t]*\d+[A-Z]?(?:[ \t]*\(\d+\))?"
    r"|\b(?:[A-Z][A-Za-z.&()]*[ \t]+){1,7}Act\b(?:,?[ \t]*\d{4})?"
    r"|\b(?:Cr\.?\s?P\.?\s?C|C\.?\s?P\.?\s?C|I\.?\s?P\.?\s?C|BNSS|BNS|N\.?\s?I\.?\s?Act)\b"
)
LEGAL_WORDS = (
    "bail",
    "writ petition",
    "eviction",
    "tenant",
    "compensation",
    "maintenance",
    "arbitration",
    "specific performance",
    "decree",
    "injunction",
    "quash",
    "FIR",
    "cheque",
    "appeal",
    "revision",
    "partition",
    "sale deed",
    "award",
    "acquittal",
    "conviction",
    "anticipatory",
    "termination",
    "pension",
    "acquisition",
    "insurance",
    "contempt",
    "suit",
    "affidavit",
)
IDENTIFIERS = [
    re.compile(r"\b[6-9]\d{9}\b"),  # Indian mobile
    re.compile(r"\b0\d{2,4}[- ]\d{6,8}\b"),  # landline
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),  # email
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # PAN
    re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),  # Aadhaar-style
    re.compile(r"\b[A-Z]{2}[- ]?\d{1,2}[- /]?[A-Z]{1,3}[- /]?\d{4}\b"),  # vehicle registration
    # the name after S/o, W/o, D/o on the same line (not the "AGED ABOUT" or address after it)
    re.compile(
        r"\b(?:S/O|W/O|D/O|S/o|W/o|D/o|son of|wife of|daughter of)\.?[ \t]+"
        r"(?:(?:SRI|SHRI|SMT|LATE|Late|Sri|Shri|Smt)\.?[ \t]+)*"
        r"((?:(?!AGED\b|Aged\b|AGE\b|Age\b|RESIDENT\b|Resident\b|R/[OA]T?\b|VILL\b|Vill\b|AT\b|"
        r"NO\b|OCC\b|Occ\b|OCCUPATION\b|Occupation\b|Profession\b|PROFESSION\b)"
        r"[A-Z][A-Za-z.]*[ \t]?){1,4})"
    ),
]
NOISE = [
    re.compile(p, re.I)
    for p in (
        r"^\s*\[\d{4}\]\s+\d+\s+S\.C\.R\.\s+\d+",
        r"^\s*\d+\s+\[\d{4}\]\s+\d+\s+S\.C\.R\.",
        r"^\s*(Digital\s+)?Supreme Court Reports\s*$",
        r"^\s*\*?\s*Ed\.? Note",
        r"^\s*\*\s*Author\s*$",
        r"digitally signed|signature not verified|signing date|^\s*location\s*:|^\s*reason\s*:",
        r"^\s*NC:\s*\d{4}:",
        r"^\s*-\s*\d+\s*-\s*$",
        r"^\s*page\s+\d+\s+of\s+\d+",
        r"^\s*\d+\s*$",
        r"click here to check corrigendum|downloaded from|authenticity of the order|"
        r"this is a digitally signed|order portal by scanning",
        r"^\s*\d+\s+(Hereinafter|Hereafter|For short|In short)\b",
        r"^\s*Neutral Citation",
        r"^\s*VERDICTUM\.IN\s*$",
        r"^\s*\$~",
        r"^\s*Page No\.\s*#",
        r"^\s*[A-H]\s*$",  # margin letters of the 2023 S.C.R. layout
        r"SUPREME COURT REPORTS",  # running page header
        r"^[A-Z0-9 .,&@'()/-]+ v\. [A-Z0-9 .,&@'()/-]+ \d{2,4}$",  # running "A v. B 1037"
        r"^\d{2,4} [A-Z0-9 .,&@'()/-]+ v\. [A-Z0-9 .,&@'()/-]+$",
        r"^\[[A-Z .,]+, J\.\]$",  # running "[B. R. GAVAI, J.]"
        r"(?-i:^[^a-z]* v\. [^a-z]*\b\d{2,4}\b[^a-z]*$)",  # running header, page number inside
    )
]
CASE_TYPES = {
    "WP": "writ petition",
    "W.P.(C)": "writ petition (civil)",
    "WP(C)": "writ petition (civil)",
    "CRL.P": "criminal petition",
    "CRLP": "criminal petition",
    "Crl.Pet.": "criminal petition",
    "CRL.A": "criminal appeal",
    "CRL.RP": "criminal revision",
    "CRL.REV.P.": "criminal revision",
    "CRL.M.C.": "criminal miscellaneous (quashing)",
    "BAIL APPLN.": "bail application",
    "Bail Appl.": "bail application",
    "Bail Appln.": "bail application",
    "MFA": "miscellaneous first appeal",
    "MAC.APP.": "motor accident claims appeal",
    "RSA": "regular second appeal",
    "RFA": "regular first appeal",
    "FA": "first appeal",
    "SA": "second appeal",
    "CRP": "civil revision",
    "C.R.P.": "civil revision",
    "RC.REV.": "rent control revision",
    "ARB.P.": "arbitration petition",
    "CS(COMM)": "commercial suit",
    "CS(OS)": "original suit",
    "MAT": "appeal (appellate side)",
    "APPLN": "criminal application",
    "SCA": "special civil application",
    "SCR.A": "special criminal application",
    "OP(C)": "original petition (civil)",
    "RPFC": "family court revision",
    "CMP": "civil miscellaneous petition",
    "CRM": "criminal miscellaneous",
    "CRR": "criminal revision",
    "AO": "appeal from order",
    "SLP": "special leave petition",
}


def _get(url: str, dest: Path) -> Path | None:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sanitise-legal-corpus"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        return dest
    except Exception:
        return None


def _pdf_text(path: Path) -> str:
    """The first 8 pages' text: poppler's pdftotext when installed (keeps words whole), else
    pypdf."""
    if shutil.which("pdftotext"):
        out = subprocess.run(
            ["pdftotext", "-l", "8", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.decode("utf-8", errors="replace")
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(path.read_bytes()))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:8])
    except Exception:
        return ""


def clean(text: str) -> str:
    """Page furniture out, whitespace tidied, wrapped lines of a paragraph joined."""
    lines = []
    for raw in text.replace("\u00a0", " ").replace("\x0c", "\n").splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            lines.append("")
            continue
        if any(p.search(line) for p in NOISE):
            continue
        lines.append(line)
    out: list[str] = []
    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        # S.C.R. margin letters left inside a sentence: "the High F Court", "(IPC). G"
        line = re.sub("(?<=[a-z,.;)\u201d\u2019]) [A-H](?= [a-z(]| ?$)", "", line)
        line = re.sub(r" [B-H](?=\s|$)(?!\.)", "", line)  # "First E Appeal", "2023 B"
        line = re.sub(r"(?<=\S) A$", "", line)
        if out and out[-1] == "" and len(out) > 1:
            before = out[-2]
            inside = re.match(r"^[a-z(]", line) or (len(before) >= 40 and before.upper() != before)
            if inside and not before.endswith((".", ":", ";", "-")):
                out.pop()  # a blank line inside a sentence: rejoin it
        prev = out[-1] if out else ""
        mid_sentence = (
            bool(prev) and re.match(r"^[a-z(]", line) and not prev.endswith((".", ":", ";"))
        )
        joins = mid_sentence or (
            prev
            and len(prev) >= 55
            and not prev.endswith((":", ";", "."))
            and not re.match(r"^(\(?\d+[.)]|\(?[a-z]{1,3}\)|[A-Z ]{6,}$)", line)
        )
        if joins:
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("�", "'")
    return text.strip()


def excerpt(text: str, limit: int = MAX_CHARS) -> str:
    """At most ``limit`` characters, cut at the last line break (or sentence end) before it."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    at = max(cut.rfind("\n"), cut.rfind(". "))
    return cut[: at + 1 if at > limit // 2 else limit].rstrip()


def reads_as_english(text: str) -> bool:
    letters = sum(c.isalpha() for c in text)
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    words = re.findall(r"[A-Za-z]+", text)
    if letters < 600 or ascii_letters / max(letters, 1) < 0.95 or len(words) < 120:
        return False
    singles = sum(len(w) == 1 for w in words) / len(words)
    return singles < 0.12 and 3.2 < sum(map(len, words)) / len(words) < 8.5


def person_names(party: str) -> list[str]:
    """The names of a person in a party field; [] for the State, companies and officials."""
    out = []
    for part in re.split(r"\s+@\s+|\s+ALIAS\s+", party or "", flags=re.I):
        name = PARTY_TAIL.sub("", part.strip())
        name = HONORIFIC.sub("", name).strip(" ,.;:-")
        name = re.sub(r"^\d+\.\s*", "", name)
        while HONORIFIC.match(name):  # "LATE SRI RAM SHARAN"
            name = HONORIFIC.sub("", name).strip()
        if not name or ORG_WORDS.search(name) or len(name.replace(" ", "")) < 4:
            continue
        if re.search(r",|\bAND\b|\bASSOCIATES\b|\bPARTNERS\b|&", name, re.I):  # firms, bodies
            continue
            continue
        if not re.search(r"[A-Za-z]{3}", name) or len(name.split()) > 6:
            continue
        out.append(name)
    return out


def find_literal(name: str, text: str) -> str | None:
    """``name`` as it is written in ``text`` (spacing and dots may differ), or None."""
    tokens = [t for t in re.split(r"[\s.]+", name) if t]
    if not tokens:
        return None
    pattern = r"\b" + r"[ \t.]*".join(map(re.escape, tokens)) + r"\b"  # on one line
    m = re.search(pattern, text, re.I)
    return m.group(0) if m else None


def spot_checks(text: str, parties: list[str]) -> tuple[list[str], list[str]]:
    hide: list[str] = []
    for party in parties:
        for name in person_names(party):
            lit = find_literal(name, text)
            if lit and lit.lower() not in (h.lower() for h in hide):
                hide.append(lit)
    if not hide:
        return [], []
    for rx in IDENTIFIERS:
        for m in rx.finditer(text):
            value = (m.group(1) if m.groups() else m.group(0)).strip(" ,.\t")
            if "\n" in value or (m.groups() and len(value.replace(" ", "")) < 4):
                continue
            if HONORIFIC.match(value + " "):  # "S/o Late" at the end of a line
                continue
            if len(value) >= 4 and not ORG_WORDS.search(value) and value not in hide:
                hide.append(value)
    keep: list[str] = []
    for m in STATUTE.finditer(text):
        value = re.sub(r"\s+", " ", m.group(0)).strip(" ,.")
        if 4 <= len(value) <= 70 and value not in keep:
            keep.append(value)
    for word in LEGAL_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text, re.I) and len(keep) < 3:
            keep.append(re.search(rf"\b{re.escape(word)}\b", text, re.I).group(0))
    return hide[:8], keep[:3]


def _sc_excerpt(text: str) -> str:
    """Caption (citation, parties, case number, date, bench) + the start of the judgment."""
    lines = text.split("\n")
    caption = []
    for line in lines[:12]:
        caption.append(line)
        if line.startswith("[") and line.rstrip().endswith("]") and "J" in line:
            break
    at = (
        re.search(r"Judgment\s*/\s*Order of the Supreme Court", text)
        or re.search(r"The (Judgment|Order) of the Court was (delivered|passed) by[^\n]*", text)
        or re.search(r"^\s*(J\s?U\s?D\s?G\s?M\s?E\s?N\s?T|O\s?R\s?D\s?E\s?R)\s*$", text, re.M)
    )
    if not at:
        return ""
    body = text[at.end() :].lstrip()
    head = "\n".join(caption).strip()
    return excerpt(head + "\n\n" + body)


def _sample(df, column: str, rng: random.Random) -> list[int]:
    """Row indices in round-robin order over the values of ``column`` (case types), shuffled
    within each, so no single case type dominates."""
    groups: dict[str, list[int]] = {}
    for idx, value in zip(df.index, df[column], strict=True):
        groups.setdefault(value, []).append(idx)
    for rows in groups.values():
        rng.shuffle(rows)
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    out: list[int] = []
    while any(groups.values()):
        for k in order:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def build(seed: int, workers: int) -> list[dict]:
    import pandas as pd

    rng = random.Random(seed)
    meta_dir = CACHE / "meta"
    docs: list[dict] = []

    def try_candidates(cands: list[dict], need: int, label: str) -> list[dict]:
        got: list[dict] = []
        batch = max(workers * 3, need)
        i = 0
        while len(got) < need and i < len(cands):
            chunk = cands[i : i + batch]
            i += batch
            with ThreadPoolExecutor(workers) as pool:
                paths = list(pool.map(lambda c: _get(c["url"], CACHE / c["cache"]), chunk))
            for cand, path in zip(chunk, paths, strict=True):
                if len(got) >= need or path is None:
                    continue
                full = clean(_pdf_text(path))
                if not reads_as_english(full) or SENSITIVE.search(full[:8000] + cand["title"]):
                    continue
                text = cand["make_excerpt"](full)
                if len(text) < 900 or not reads_as_english(text):
                    continue
                hide, keep = spot_checks(text, cand["parties"])
                if not hide or not keep:
                    continue
                got.append({**cand, "text": text, "must_hide": hide, "must_keep": keep})
            print(f"  {label}: {len(got)}/{need} after {min(i, len(cands))} tried", flush=True)
        return got

    for year, need in SC_YEARS.items():
        df = pd.read_parquet(meta_dir / f"sc_{year}.parquet")
        df = df[df["petitioner"].notna() & df["path"].notna()].reset_index(drop=True)
        df["kind"] = df["title"].str.contains("CRIMINAL|STATE OF", case=False, na=False)
        cands = []
        for idx in _sample(df, "kind", rng):
            r = df.loc[idx]
            cands.append(
                {
                    "id": f"sc_{r['path']}",
                    "court": "Supreme Court of India",
                    "title": str(r["title"]).replace("�", "'"),
                    "case": str(r.get("case_id") or r.get("citation") or ""),
                    "date": str(r["decision_date"]),
                    "doc_type": "Supreme Court judgment",
                    "parties": [str(r["petitioner"]), str(r["respondent"])],
                    "url": f"{SC_BUCKET}/data/pdf/year={year}/english/{r['path']}_EN.pdf",
                    "cache": f"sc/{r['path']}.pdf",
                    "make_excerpt": _sc_excerpt,
                }
            )
        docs += try_candidates(cands, need, f"Supreme Court {year}")

    for short, code, bench, need in HIGH_COURTS:
        f = meta_dir / f"hc_{code}_bench_{bench}.parquet"
        df = pd.read_parquet(
            f, columns=["title", "pdf_link", "cnr", "decision_date", "disposal_nature", "court"]
        )
        df = df[df["title"].str.contains(r" Vs ", na=False)]
        df = df[~df["disposal_nature"].fillna("").str.contains(SKIP_DISPOSAL)].reset_index(
            drop=True
        )
        df["kind"] = df["title"].str.extract(r"^\s*([^/]+)/")[0].fillna("?").str.strip()
        cands = []
        for idx in _sample(df, "kind", rng)[: need * 12]:
            r = df.loc[idx]
            m = re.match(r"^\s*([^/]+)/(\d+)/(\d{4}) of (.+?) Vs (.+)$", str(r["title"]).strip())
            if not m:
                continue
            kind, num, yr, pet, res = (g.strip() for g in m.groups())
            fname = str(r["pdf_link"]).rsplit("/", 1)[-1]
            cands.append(
                {
                    "id": f"hc_{short}_{fname.removesuffix('.pdf')}",
                    "court": str(r["court"]).replace("  ", " "),
                    "title": f"{pet} v. {res}",
                    "case": f"{kind} {num}/{yr}",
                    "date": str(r["decision_date"])[:10],
                    "doc_type": f"High Court {CASE_TYPES.get(kind, kind)}",
                    "parties": [pet, res],
                    "url": f"{HC_BUCKET}/data/pdf/year={HC_YEAR}/court={code}/bench={bench}/{fname}",
                    "cache": f"hc/{short}/{fname}",
                    "make_excerpt": excerpt,
                }
            )
        docs += try_candidates(cands, need, f"High Court {short}")
    return docs


def write(docs: list[dict]) -> None:
    (OUT / "docs").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "docs").glob("*.txt"):
        old.unlink()
    entries = []
    for d in docs:
        (OUT / "docs" / f"{d['id']}.txt").write_text(
            d["text"] + "\n", encoding="utf-8", newline="\n"
        )
        entries.append(
            {
                "id": d["id"],
                "file": f"docs/{d['id']}.txt",
                "title": d["title"],
                "jurisdiction": "India",
                "court": d["court"],
                "case": d["case"],
                "doc_type": d["doc_type"],
                "date": d["date"],
                "source": "Indian Supreme Court Judgments open dataset (AWS Open Data Registry)"
                if d["id"].startswith("sc_")
                else "Indian High Court Judgments open dataset (AWS Open Data Registry)",
                "url": d["url"],
                "licence": LICENCE,
                "must_hide": d["must_hide"],
                "must_keep": d["must_keep"],
            }
        )
    manifest = {
        "_comment": [
            "300 real Indian judgments and orders, built by tools/legal/build_indian_corpus.py.",
            "Each file is the start of the judgment (cause title and opening facts), at most 1,800 characters.",
            "must_hide: party names from the court's own case metadata that appear in the excerpt, plus phone numbers, emails, PAN, Aadhaar-style numbers, vehicle numbers and S/o-W/o-D/o names found by pattern.",
            "must_keep: sections, Acts and legal terms that appear in the excerpt. Both lists are checks, not a full annotation.",
        ],
        "documents": entries,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    docs = build(args.seed, args.workers)
    write(docs)
    print(f"{len(docs)} documents in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
