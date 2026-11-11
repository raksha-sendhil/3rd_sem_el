"""Real public text for the single-question metrics, downloaded fresh in the notebook.

* **health**: HealthCareMagic-100k (ChatDoctor; Li et al. 2023), real patients' questions with
  real doctors' replies, the base paper's medical dataset. A document is one dialogue
  ("Patient: ... Doctor: ..."). Utility questions are held-out patients' questions; the reference
  answer is the real doctor's reply.
* **legal**: CUAD v1 (Hendrycks et al. 2021, CC BY 4.0), 510 real contracts filed with the SEC,
  labelled by lawyers. A document is a section of a contract (at most 1,800 characters) around
  one of the lawyers' labelled clauses, headed by the contract's name. Utility questions ask
  for that clause; the reference answer is the lawyers' labelled text.

Indexed and held-out documents never come from the same dialogue or contract.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "sanitise-eval/1.0 (student research project)"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
CHATDOCTOR = "lavita/ChatDoctor-HealthCareMagic-100k"
CUAD_JSON = (
    "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
)
MAX_CHARS = 1900  # under the agents' 2,000-character limit


def doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get(url: str, cache: Path | None = None) -> bytes:
    if cache is not None and cache.exists():
        return cache.read_bytes()
    last: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            return data
        except Exception as exc:  # network hiccups and rate limits: wait and retry
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


# ---------------------------------------------------------------- health: ChatDoctor


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def chatdoctor(n_indexed: int, n_heldout: int, cache: Path, seed: int = 2026) -> dict:
    """``n_indexed + n_heldout`` distinct dialogues of at most MAX_CHARS characters."""
    want = n_indexed + n_heldout
    q = urllib.parse.urlencode({"dataset": CHATDOCTOR, "config": "default", "split": "train",
                                "offset": 0, "length": 1})  # fmt: skip
    total = json.loads(_get(f"{HF_ROWS}?{q}"))["num_rows_total"]
    rng = random.Random(seed)
    offsets = list(range(0, total - 100, 100))
    rng.shuffle(offsets)
    seen: set[str] = set()
    dialogues: list[dict] = []
    for off in offsets:
        if len(dialogues) >= want:
            break
        q = urllib.parse.urlencode({"dataset": CHATDOCTOR, "config": "default", "split": "train",
                                    "offset": off, "length": 100})  # fmt: skip
        rows = json.loads(_get(f"{HF_ROWS}?{q}", cache / f"chatdoctor_{off}.json"))["rows"]
        for r in rows:
            patient, doctor = _clean(r["row"]["input"]), _clean(r["row"]["output"])
            text = f"Patient: {patient}\nDoctor: {doctor}"
            if len(patient) < 80 or len(doctor) < 80 or len(text) > MAX_CHARS or patient in seen:
                continue
            seen.add(patient)
            dialogues.append({"doc_id": doc_id(text), "text": text, "question": patient,
                              "reference": doctor, "source": f"{CHATDOCTOR} row {r['row_idx']}"})  # fmt: skip
    rng.shuffle(dialogues)
    indexed, heldout = dialogues[:n_indexed], dialogues[n_indexed:want]
    utility = [{"id": f"u{i:03d}", "query": d["question"], "reference": d["reference"],
                "doc_id": None} for i, d in enumerate(heldout)]  # fmt: skip
    return {"indexed": indexed, "heldout": heldout, "utility": utility}


# ---------------------------------------------------------------- legal: CUAD

QUESTIONS = {
    "Parties": "Who are the parties to the {c}?",
    "Agreement Date": "On what date was the {c} signed?",
    "Effective Date": "When does the {c} take effect?",
    "Expiration Date": "When does the {c} expire?",
    "Governing Law": "Which law governs the {c}?",
    "Renewal Term": "What is the renewal term of the {c}?",
    "Notice Period To Terminate Renewal": "How much notice is needed to stop the {c} renewing?",
    "Termination For Convenience": "Can a party end the {c} for convenience, and how?",
    "Cap On Liability": "What is the cap on liability in the {c}?",
    "License Grant": "What licence is granted under the {c}?",
    "Exclusivity": "What exclusivity does the {c} give?",
    "Non-Compete": "What non-compete obligations does the {c} contain?",
    "Insurance": "What insurance must be kept under the {c}?",
    "Minimum Commitment": "What minimum commitment does the {c} require?",
}
_CATEGORY = re.compile(r'related to "([^"]+)"')


def contract_name(title: str) -> str:
    """'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT' -> 'distributor agreement of
    Limeenergyco'; 'Magenta Therapeutics, Inc. - Master Development Agreement' -> 'master
    development agreement of Magenta Therapeutics, Inc.'."""
    if " - " in title:
        company, kind = (x.strip() for x in title.split(" - ", 1))
    else:
        company = title.split("_")[0].strip().title()
        kind = title.rsplit("-", 1)[-1].strip()
    return f"{kind.lower() or 'agreement'} of {company}"


def _window(context: str, start: int, length: int) -> tuple[int, int]:
    pad = max(0, (MAX_CHARS - 120 - length) // 2)
    lo, hi = max(0, start - pad), min(len(context), start + length + pad)
    while lo > 0 and not context[lo - 1].isspace():
        lo += 1
    while hi < len(context) and not context[hi].isspace():
        hi -= 1
    return lo, max(hi, start + length)


def _sections(contract: dict) -> list[dict]:
    title = contract["title"]
    name = contract_name(title)
    out: list[dict] = []
    taken: list[tuple[int, int]] = []
    for para in contract["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            m = _CATEGORY.search(qa["question"])
            if not m or m.group(1) not in QUESTIONS or not qa["answers"]:
                continue
            ans = qa["answers"][0]
            text_a, start = ans["text"].strip(), ans["answer_start"]
            if not 3 <= len(text_a) <= 600:
                continue
            lo, hi = _window(context, start, len(ans["text"]))
            if any(lo < b and a < hi for a, b in taken):
                continue
            body = re.sub(r"[ \t]+", " ", context[lo:hi]).strip()
            text = f"Contract: {name}\n{body}"
            if len(text) > MAX_CHARS + 100 or text_a not in text:
                continue
            taken.append((lo, hi))
            out.append({"doc_id": doc_id(text), "text": text, "contract": title,
                        "category": m.group(1), "reference": text_a,
                        "question": QUESTIONS[m.group(1)].format(c=name),
                        "source": f"CUAD v1, {title}"})  # fmt: skip
    return out


def cuad(n_indexed: int, n_heldout: int, cache: Path, seed: int = 2026) -> dict:
    data = json.loads(_get(CUAD_JSON, cache / "CUAD_v1.json"))["data"]
    rng = random.Random(seed)
    rng.shuffle(data)
    indexed: list[dict] = []
    heldout: list[dict] = []
    for contract in data:
        secs = _sections(contract)
        if not secs:
            continue
        if len(indexed) < n_indexed:
            indexed += secs[: max(1, min(len(secs), 4))]
        elif len(heldout) < n_heldout:
            heldout += secs[: max(1, min(len(secs), 4))]
        else:
            break
    indexed, heldout = indexed[:n_indexed], heldout[:n_heldout]
    pool = indexed[:]
    rng.shuffle(pool)
    utility = [{"id": f"u{i:03d}", "query": s["question"], "reference": s["reference"],
                "doc_id": s["doc_id"]} for i, s in enumerate(pool[:n_heldout])]  # fmt: skip
    return {"indexed": indexed, "heldout": heldout, "utility": utility}
