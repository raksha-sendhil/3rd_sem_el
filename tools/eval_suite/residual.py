"""What is left in the stored rewrites, with no question asked.

Target Info only sees what an answer happens to repeat; this looks at every rewritten document
in a store and counts the private details still in it:

* ``generated``: the generated records, per field of the person table. Exact fields (name, age,
  town, job, phone...) by the pipeline's own matcher and by ``loose_found`` (a first name or
  surname alone, a bare age, the last six digits); indirect fields (age band, district, state,
  job group), which the agents keep on purpose and which let an attacker narrow a person down.
* ``scans``: the end-to-end scans, per labelled private value (``e2e/labels.json``).
* ``pii``: an unseen slice of nvidia/Nemotron-PII (``fetch_pii``): about 200 documents of the
  domain from the *train* split (the 40 documents looked at during development came from the
  test split), each with the dataset's own labelled spans, split into direct identifiers and
  indirect details (``INDIRECT``).
"""

from __future__ import annotations

import ast
import json
import random
import re
import statistics
import urllib.parse
from collections import Counter
from pathlib import Path

NEMOTRON = "nvidia/Nemotron-PII"
NEMOTRON_DOMAIN = {"health": "Healthcare", "legal": "Legal", "finance": "Finance"}
FILTER = "https://datasets-server.huggingface.co/filter"
MAX_CHARS = 2500  # each document goes through Agents 1-3 once
PII_SEED = 2027
# Nemotron-PII labels that describe a person without pointing at them; every other label
# (names, numbers, contact details, addresses, dates of birth...) is a direct identifier
INDIRECT = {
    "occupation", "state", "county", "country", "city", "gender", "age", "blood_type",
    "employment_status", "race_ethnicity", "education_level", "language", "religious_belief",
    "political_view", "sexuality", "date", "time", "date_time", "company_name", "postcode",
}  # fmt: skip
COARSE = ("age_band", "district", "state", "occupation_group")
_SKIP = {"mr", "mrs", "ms", "dr", "the", "and", "of", "pvt", "ltd", "llp", "inc"}


# ---------------------------------------------------------------- loose matching


def loose_found(field: str, value: str, text: str) -> bool:
    """``value`` of ``field`` in ``text`` in a looser form than the pipeline's scorer: any part
    of a name of three letters or more, an age as a bare number, the last six digits of a
    number, the part of an email before the @, any other value as a phrase."""
    low, value = text.lower(), str(value)
    if field == "age":
        # not a decade ("in their 60s") nor part of a range or a larger number
        return bool(re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.]|'?s\b|\s*-\s*\d)", low))
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 6:
        runs = re.findall(r"\d[\d\s().-]*\d", low)
        return any(digits[-6:] in re.sub(r"\D", "", r) for r in runs)
    if field == "email":
        return value.split("@")[0].lower() in low
    if field.startswith("name") or field == "opposing_party":
        parts = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in _SKIP]
        return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low) for w in parts)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])", low))


# ---------------------------------------------------------------- the unseen annotated slice


def fetch_pii(domain: str, n: int, out: Path, cache: Path, seed: int = PII_SEED) -> int:
    """``n`` documents of the domain from Nemotron-PII's train split, at most MAX_CHARS each,
    into ``out/corpus`` (built like any other store) and their labels into ``out/labels.json``."""
    if n <= 0 or domain not in NEMOTRON_DOMAIN:
        return 0
    from eval_suite import realdata

    where = f"\"domain\"='{NEMOTRON_DOMAIN[domain]}'"

    def url(offset: int, length: int) -> str:
        return FILTER + "?" + urllib.parse.urlencode({
            "dataset": NEMOTRON, "config": "default", "split": "train", "where": where,
            "offset": offset, "length": length})  # fmt: skip

    total = json.loads(realdata._get(url(0, 1)))["num_rows_total"]
    offsets = list(range(0, total, 100))
    random.Random(seed).shuffle(offsets)
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[dict]] = {}
    for off in offsets:
        if len(labels) >= n:
            break
        try:
            rows = json.loads(realdata._get(url(off, 100), cache / f"nemotron_{domain}_{off}.json"))
        except RuntimeError as exc:  # one block unavailable: the next shuffled block
            print(f"skipped Nemotron-PII rows {off}-{off + 99}: {exc}", flush=True)
            continue
        for r in rows["rows"]:
            row = r["row"]
            text = row["text"]
            spans = row["spans"]
            spans = ast.literal_eval(spans) if isinstance(spans, str) else spans
            if len(labels) >= n or len(text) > MAX_CHARS or not spans:
                continue
            doc = realdata.doc_id(text)
            (corpus / f"{doc}.json").write_text(json.dumps(
                {"doc_id": doc, "text": text, "source": f"{NEMOTRON} train {row['uid']}"},
                ensure_ascii=False), encoding="utf-8")  # fmt: skip
            labels[doc] = [{"label": s["label"], "value": s["text"]} for s in spans]
    (out / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False),
                                     encoding="utf-8")  # fmt: skip
    print(f"Nemotron-PII ({NEMOTRON_DOMAIN[domain]}, train split, unseen): {len(labels)} documents, "
          f"{sum(map(len, labels.values()))} labelled values")  # fmt: skip
    return len(labels)


# ---------------------------------------------------------------- counting


def _texts(d, store: Path) -> dict[str, str]:
    if not store.exists():
        return {}
    opened = d.mod("rag.store").SqliteDocumentStore(store)
    try:
        return {doc: opened.sanitised(doc).rewriter.safe_text for doc in opened.ids()}
    finally:
        opened.close()


def _pair(left: Counter, total: Counter) -> dict[str, list[int]]:
    return {k: [left[k], total[k]] for k in sorted(total)}


def generated(d, store: Path) -> dict | None:
    texts = _texts(d, store)
    by_doc = {x: p for p in d.people() for x in p.doc_ids}
    texts = {k: v for k, v in texts.items() if k in by_doc}
    if not texts:
        return None
    from sanitise.evaluation.attacks import _found  # the coarse matcher, the same in every domain

    exact, loose, total, coarse, coarse_total = Counter(), Counter(), Counter(), Counter(), Counter()
    for doc, text in texts.items():
        p = by_doc[doc]
        got = d.disclosed_exact(p, text)
        for f, v in d.exact_fields(p).items():
            k = f.split("_")[0] if f.startswith("condition") else f
            total[k] += 1
            exact[k] += f in got
            loose[k] += loose_found(f, v, text)
        for f in COARSE:
            if v := getattr(p, f, None):
                coarse_total[f] += 1
                coarse[f] += _found(f, str(v), text.lower())
    return {"records": len(texts), "exact": _pair(exact, total), "loose": _pair(loose, total),
            "coarse": _pair(coarse, coarse_total)}  # fmt: skip


def scans(d, work: Path) -> dict:
    from eval_suite.e2e import value_found

    labels_path, manifest_path = work / "e2e" / "labels.json", work / "e2e" / "manifest.json"
    if not labels_path.exists() or not manifest_path.exists():
        return {}
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    manifest = {m["id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
    out = {}
    for store, key in (("e2e_gold", "gold_parts"), ("e2e_ocr", "ocr_parts")):
        texts = _texts(d, work / "stores" / f"{store}.sqlite")
        if not texts:
            continue
        left, total = Counter(), Counter()
        for scan, lab in labels.items():
            parts = [texts[x] for x in (manifest.get(scan, {}).get(key) or []) if x in texts]
            if not parts:
                continue
            joined = "\n".join(parts)
            for v in lab.get("private_values", []):
                kind = str(v.get("type", "value")).lower()
                total[kind] += 1
                left[kind] += value_found(str(v["value"]), joined)
        if total:
            out[store] = {"left": sum(left.values()), "total": sum(total.values()),
                          "by_type": _pair(left, total)}  # fmt: skip
    return out


def pii(d, work: Path) -> dict | None:
    from eval_suite.e2e import value_found

    path = work / "pii" / "labels.json"
    texts = _texts(d, work / "stores" / "pii.sqlite")
    if not path.exists() or not texts:
        return None
    labels = json.loads(path.read_text(encoding="utf-8"))
    left, total = Counter(), Counter()
    direct_docs, kept = [], []
    for doc, spans in labels.items():
        if doc not in texts:
            continue
        text = texts[doc]
        original = json.loads((work / "pii" / "corpus" / f"{doc}.json").read_text(encoding="utf-8"))
        kept.append(len(text) / max(len(original["text"]), 1))
        any_direct = False
        for s in {(x["label"], x["value"]) for x in spans}:
            total[s[0]] += 1
            if value_found(str(s[1]), text):
                left[s[0]] += 1
                any_direct |= s[0] not in INDIRECT
        direct_docs.append(any_direct)

    def part(direct: bool) -> list[int]:
        keys = [k for k in total if (k not in INDIRECT) == direct]
        return [sum(left[k] for k in keys), sum(total[k] for k in keys)]

    return {"documents": len(kept), "direct": part(True), "indirect": part(False),
            "documents_with_direct_left": sum(direct_docs) / max(len(direct_docs), 1),
            "length_kept_median": statistics.median(kept) if kept else float("nan"),
            "by_label": _pair(left, total)}  # fmt: skip


def compute(d, work: Path) -> dict:
    out = {}
    if g := generated(d, work / "stores" / "gen.sqlite"):
        out["generated"] = g
    if s := scans(d, work):
        out["scans"] = s
    if p := pii(d, work):
        out["pii"] = p
    return out


# ---------------------------------------------------------------- report section


def _frac(x: list[int]) -> str:
    left, total = x
    return f"{left} of {total}" + (f" ({100 * left / total:.0f}%)" if total else "")


def section(r: dict) -> list[str]:
    if not r:
        return []
    L = ["", "## What is left in the stored documents (no questions asked)", "",
         "Every rewritten document in the store, checked for the private details still in it. "
         "This tests Agents 1-3 directly; Target Info only sees what an answer repeats.", ""]  # fmt: skip
    if g := r.get("generated"):
        L += [f"Generated records ({g['records']}), per field of the person table:", "",
              "| Field | Left, exact | Left, loose | Kind |", "|---|---:|---:|---|"]
        for f in g["exact"]:
            kind = "content, kept on purpose" if f == "condition" else "identifying"
            L.append(f"| {f} | {_frac(g['exact'][f])} | {_frac(g['loose'][f])} | {kind} |")
        for f, x in g["coarse"].items():
            L.append(f"| {f.replace('_', ' ')} | {_frac(x)} | | indirect, kept on purpose |")
    if s := r.get("scans"):
        for store, x in s.items():
            label = {"e2e_gold": "gold text", "e2e_ocr": "scan + OCR"}[store]
            L += ["", f"End-to-end scans, {label}: {_frac([x['left'], x['total']])} labelled "
                      "private values left. " + "; ".join(
                          f"{k} {v[0]}/{v[1]}" for k, v in x["by_type"].items())]  # fmt: skip
    if p := r.get("pii"):
        L += ["", f"Unseen annotated documents (nvidia/Nemotron-PII, train split, "
                  f"{p['documents']} documents; the dataset's own labels; whole-value match):", "",
              "| | Left in the rewrite |", "|---|---:|",
              f"| direct identifiers (names, numbers, contact details, addresses...) | {_frac(p['direct'])} |",
              f"| indirect details (age, gender, city, job, dates...) | {_frac(p['indirect'])} |",
              f"| documents with any direct identifier left | {100 * p['documents_with_direct_left']:.1f}% |",
              f"| rewrite length / original length, median | {p['length_kept_median']:.2f} |", "",
              "| Label | Left |", "|---|---:|"]  # fmt: skip
        by = sorted(p["by_label"].items(), key=lambda kv: -kv[1][1])
        L += [f"| {k}{' (indirect)' if k in INDIRECT else ''} | {_frac(v)} |" for k, v in by]
    return L
