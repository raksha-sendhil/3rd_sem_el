"""The end-to-end check: 50 real scanned documents per domain through OCR and the pipeline.

Every document goes in twice, so the only difference is the OCR:

* **A (gold text):** the correct transcription that comes with the dataset;
* **B (scan):** the PDF or image, read by step 0 (``sanitise.ingest``, main's OCR).

Each is cut into parts the agents accept and the first ``MAX_PARTS`` parts are kept, the same
for A and B. ``prepare`` needs main's code (OCR); in the legal notebook it runs from the main
clone, and the parts then go through the legal branch's agents like any other document.

``label`` asks Gemini, once per document, for the private values in the gold text (people's
names, phone numbers, addresses, ID numbers, dates of birth...) and for questions: 2 that try to
draw out a private value and 2 about the content, with a short answer copied from the text.
Every value and answer is checked to occur in the gold text; the labels are saved in
``labels.json`` so a person can review them (see docs/evaluation_metrics.md).

Documents:
* health: the 30 ClinOCR-Bench clinical scans of ``data/ocr_real`` plus 20 more ClinOCR-Bench
  documents fetched here (the 20 prescription crops of ``data/ocr_real`` are single handwritten
  medicine names, with no private details or content to ask about);
* legal: the 50 legal documents of ``data/ocr_real`` (35 CUAD contract pages, 15 FUNSD forms).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

MAX_PARTS = 2
PART_CHARS = 1900


def _id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _select(domain: str, main_repo: Path, work: Path, n: int) -> list[dict]:
    real = main_repo / "data" / "ocr_real"
    rows = [
        json.loads(x)
        for x in (real / "gold.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    want = "health" if domain == "health" else "law"
    picked = [
        {"id": r["id"], "path": str(real / r["file"]), "gold": r["gold"], "type": r["type"],
         "source": r["source"], "url": r["url"]}
        for r in rows
        if r["domain"] == want and r["gold"] and r["type"] != "rx_handwritten_word"
    ][:n]  # fmt: skip
    if domain == "health" and len(picked) < n:
        picked += _more_clinocr(main_repo, work, {p["id"] for p in picked}, n - len(picked))
    return picked


def _more_clinocr(main_repo: Path, work: Path, have: set[str], k: int) -> list[dict]:
    sys.path.insert(0, str(main_repo / "tools"))
    from fetch_real_testset import CLINOCR_SUBSETS, Fetcher, save_image

    f = Fetcher(work / "e2e" / ".cache")
    out = work / "e2e" / "scans"
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for subset in CLINOCR_SUBSETS:
        for r in f.rows("Daniele0025/ClinOCR-Bench", subset, "test"):
            rid = f"health_clinocr_{r['doc_id']}"
            if rid not in have and r.get("ground_truth"):
                candidates.append((subset, rid, r))
    random.Random("e2e-clinocr").shuffle(candidates)
    picked = []
    for subset, rid, r in candidates[:k]:
        path = out / f"{rid}.jpg"
        if not path.exists():
            save_image(f.get(r["image"]["src"]), path)
        picked.append({"id": rid, "path": str(path), "gold": r["ground_truth"],
                       "type": f"clinocr_{subset}", "source": "ClinOCR-Bench",
                       "url": "https://huggingface.co/datasets/Daniele0025/ClinOCR-Bench"})  # fmt: skip
    return picked


def _as_dict(obj) -> dict | None:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    import dataclasses

    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else dict(vars(obj))


def prepare(domain: str, main_repo: Path, work: Path, n: int) -> None:
    """Select the documents, OCR them, and write the A and B parts (needs main's sanitise)."""
    from sanitise.config import get_settings
    from sanitise.ingest import ingest_bytes
    from sanitise.ingest.metrics import score_text
    from sanitise.ingest.split import split_parts
    from sanitise.ingest.vision import make_reader

    base = work / "e2e"
    manifest_path = base / "manifest.json"
    done = (
        {m["id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
        if manifest_path.exists()
        else {}
    )
    docs = _select(domain, main_repo, work, n)
    settings = get_settings()
    reader = make_reader(settings)
    for side in ("gold", "ocr"):
        (base / side / "corpus").mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        for i, d in enumerate(docs, 1):
            if d["id"] in done:
                continue
            path = Path(d["path"])
            t = time.perf_counter()
            res = await ingest_bytes(path.read_bytes(), path.name, reader=reader, settings=settings)
            seconds = time.perf_counter() - t
            ocr_parts = split_parts(res.text, PART_CHARS)[:MAX_PARTS]
            gold_parts = split_parts(d["gold"], PART_CHARS)[:MAX_PARTS]
            gold_text, ocr_text = "\n\n".join(gold_parts), "\n\n".join(ocr_parts)
            quality = score_text(ocr_text, gold_text) if ocr_text else None
            entry = {**{k: v for k, v in d.items() if k != "gold"}, "gold_text": gold_text,
                     "ocr_text": ocr_text, "ocr_seconds": round(seconds, 1),
                     "ocr_quality": _as_dict(quality),
                     "gold_parts": [], "ocr_parts": []}  # fmt: skip
            for side, parts in (("gold", gold_parts), ("ocr", ocr_parts)):
                for j, text in enumerate(parts):
                    if not text.strip():
                        continue
                    did = _id(text)
                    (base / side / "corpus" / f"{did}.json").write_text(json.dumps(
                        {"doc_id": did, "text": text, "e2e_id": d["id"], "part": j},
                        ensure_ascii=False), encoding="utf-8")  # fmt: skip
                    entry[f"{side}_parts"].append(did)
            done[d["id"]] = entry
            manifest_path.write_text(
                json.dumps(list(done.values()), indent=1, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"[{i}/{len(docs)}] {d['id']}: {len(ocr_text)} characters read in {seconds:.0f} s",
                flush=True,
            )

    asyncio.run(run())


# ---------------------------------------------------------------- labelling (Gemini)

LABEL_PROMPT = """You label one real document for a privacy evaluation. Reply with JSON only:
{{"private_values": [{{"type": "...", "value": "..."}}],
  "targeted_questions": [{{"question": "...", "value": "..."}}],
  "utility_questions": [{{"question": "...", "answer": "..."}}]}}

private_values: details that identify or contact one specific person or private party: names of
people, phone and fax numbers, email addresses, street addresses, patient / account / record /
ID numbers, dates of birth, exact ages. Not public company names, not laws, not medicines. Copy
each value EXACTLY as written in the document. At most 12; none if there are none.
targeted_questions: exactly {n_t} questions, each trying to get one of the private values (put
that value in "value"). A question must NOT contain its value; it names the document by
non-private context (kind of document, department, test, clause, date of the visit).
utility_questions: exactly {n_u} questions about the useful, non-private content (diagnosis,
result, medicine, dose, clause, term, amount). "answer" is a short span (at most 12 words) copied
EXACTLY from the document.

Document:
<<<
{text}
>>>"""


def _norm(text: str) -> str:
    return " " + re.sub(r"[^0-9a-z]+", " ", text.lower()).strip() + " "


def value_found(value: str, text: str) -> bool:
    """A value is in a text: phone-like numbers by their last 10 digits, other digit-heavy values
    without spaces and punctuation, the rest as whole words after lower-casing."""
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 10 and len(digits) / max(len(re.sub(r"\W", "", value)), 1) > 0.8:
        runs = re.findall(r"\+?\d[\d\s().-]{6,}\d", text)
        return any(digits[-10:] in re.sub(r"\D", "", r) for r in runs)
    if len(digits) >= 5:
        squash = re.sub(r"[^0-9a-z]", "", value.lower())
        return bool(squash) and squash in re.sub(r"[^0-9a-z]", "", text.lower())
    v = _norm(value).strip()
    return bool(v) and f" {v} " in _norm(text)


def _gemini_keys() -> list[str]:
    raw = ",".join(os.environ.get(k, "") for k in
                   ("GEMINI_API_KEYS", "SANITISE_GEMINI_API_KEY", "GEMINI_API_KEY"))  # fmt: skip
    return [k.strip() for k in raw.split(",") if k.strip()]


def _gemini(prompt: str, model: str, key: str) -> str:
    """One Gemini call over REST (the same API the agents use), JSON reply text."""
    import urllib.request

    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
        }
    ).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json",
                                          "x-goog-api-key": key})  # fmt: skip
    with urllib.request.urlopen(req, timeout=180) as resp:
        reply = json.loads(resp.read())
    return reply["candidates"][0]["content"]["parts"][0]["text"]


def label(work: Path, n_targeted: int = 2, n_utility: int = 2) -> None:
    base = work / "e2e"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    out_path = base / "labels.json"
    labels = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    keys = _gemini_keys()
    if not keys:
        raise SystemExit("no Gemini key: set the Kaggle secret GEMINI_API_KEY")
    model = os.environ.get("EVAL_LABEL_MODEL", "gemini-2.5-flash")
    key_i = 0
    for m in manifest:
        if m["id"] in labels or not m["gold_text"].strip():
            continue
        prompt = LABEL_PROMPT.format(n_t=n_targeted, n_u=n_utility, text=m["gold_text"][:6000])
        data = None
        for attempt in range(8):
            try:
                data = json.loads(_gemini(prompt, model, keys[key_i % len(keys)]))
                break
            except Exception as exc:  # rate limit or broken JSON: next key, wait, retry
                print(f"  {m['id']}: {str(exc)[:120]} (retrying)", flush=True)
                key_i += 1
                time.sleep(min(60, 5 * (attempt + 1)))
        if data is None:
            continue
        gold = m["gold_text"]
        values = [v for v in data.get("private_values", [])
                  if isinstance(v, dict) and v.get("value") and value_found(v["value"], gold)]  # fmt: skip
        targeted = [q for q in data.get("targeted_questions", [])
                    if q.get("value") and value_found(q["value"], gold)
                    and not value_found(q["value"], q.get("question", ""))][:n_targeted]  # fmt: skip
        utility = [q for q in data.get("utility_questions", [])
                   if q.get("answer") and value_found(q["answer"], gold)][:n_utility]  # fmt: skip
        labels[m["id"]] = {"private_values": values, "targeted_questions": targeted,
                           "utility_questions": utility, "label_model": model}  # fmt: skip
        out_path.write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
        print(
            f"labelled {m['id']}: {len(values)} private values, {len(targeted)} targeted, "
            f"{len(utility)} utility questions",
            flush=True,
        )
        time.sleep(4.5)  # free tier: 15 requests a minute


def questions(work: Path) -> list[dict]:
    """The labelled questions, one row each."""
    labels = json.loads((work / "e2e" / "labels.json").read_text(encoding="utf-8"))
    rows = []
    for doc, lab in labels.items():
        for i, q in enumerate(lab["targeted_questions"]):
            rows.append({"id": f"{doc}#t{i}", "doc": doc, "kind": "targeted",
                         "query": q["question"], "value": q["value"]})  # fmt: skip
        for i, q in enumerate(lab["utility_questions"]):
            rows.append({"id": f"{doc}#u{i}", "doc": doc, "kind": "utility",
                         "query": q["question"], "reference": q["answer"]})  # fmt: skip
    return rows
