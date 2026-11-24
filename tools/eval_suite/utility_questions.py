"""Utility questions that ask about a record's content, not its wording (health and legal).

The first health utility questions quoted the patient ("A patient wrote: '...' What did the
doctor advise?"). The protected pipeline rewrites every record, so the quoted words are no
longer in it: the answer model could not tell which record was meant and said "The records do
not say", even with the right record retrieved (59 of 60 such answers in the 5h run). That
measured how much of the *wording* survives, not how much of the *content*.

``rewrite_health`` asks Gemini, once per question, to describe the case by its medical content
in its own words (who, the main symptoms or condition, a test result) and to end with "What did
the doctor advise?". A question that copies more than six words in a row from the patient's
message is asked for again, then dropped back to its first form. The reference answer (the
real doctor's reply) and the record are unchanged. Both pipelines get the same new question.

The first questions stay in ``data/utility_v1.json``; the answers to them are moved to
``answers/utility_v1.jsonl`` so the next ``single --only utility`` asks the new ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_suite.e2e import Gemini
from eval_suite.metrics import longest_common_run, tokens

PROMPT = """Below is a patient's message to an online doctor and the doctor's reply.
Write ONE harmless question that a clinician could ask a collection of such records to find the
advice given in this case.
- Identify the case only by its medical content, in your own words: the main symptoms or
  condition, the body part, and a test result or medicine the patient mentions. Use only what
  the message states.
- Nothing personal: no name, age, sex, job, place, date, contact detail or ID, and never guess
  one ("a man in his 40s"). Say "someone", or "a relative" when it is about someone else.
- Do not copy more than four words in a row from the message.
- End the question with "what did the doctor advise?", once.
Reply with JSON only: {{"question": "..."}}

{text}"""

MAX_COPIED = 6  # words in a row from the patient's message

LEGAL_PROMPT = """Below is a section of a real contract and a question about it.
Rewrite the question so that it names the contract only by its kind and what it is about (the
product, service or industry, in general words: "the supply agreement for a dermatology drug",
"the software licence between two technology companies"), never by a company, party or person
name, a place, a date or a reference number. Keep exactly what the question asks.
Reply with JSON only: {{"question": "..."}}

QUESTION: {question}

CONTRACT SECTION:
{text}"""


def _company_words(text: str) -> set[str]:
    """The words of the contract's name line ("Contract: supply agreement of Biofronteraag"),
    after "of": the company the question must no longer name."""
    first = text.split("\n", 1)[0]
    name = first.split(" of ", 1)[1] if " of " in first else ""
    return {w.lower() for w in name.replace(",", " ").split() if len(w) > 3}


def rewrite_legal(work: Path) -> None:
    """CUAD utility questions named the contract by its company ("the supply agreement of
    Biofronteraag"); the legal agents replace company and party names, so ours could not match
    them (79 of 100 "The records do not say" in the 5h run, the right section retrieved every
    time). Each question is rewritten to name the contract by its kind and subject."""
    data = work / "data"
    path = data / "utility.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    backup = data / "utility_v1.json"
    if not backup.exists():
        backup.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        old = work / "answers" / "utility.jsonl"
        if old.exists():
            old.replace(work / "answers" / "utility_v1.jsonl")
    gemini = Gemini()
    done = 0
    for item in items:
        if item.get("version") == 2:
            continue
        doc = data / "real" / "corpus" / f"{item['doc_id']}.json"
        text = json.loads(doc.read_text(encoding="utf-8"))["text"]
        company = _company_words(text)
        question = None
        for _ in range(2):
            reply = gemini.ask(LEGAL_PROMPT.format(question=item["query"], text=text[:4000]),
                               item["id"])  # fmt: skip
            q = (reply or {}).get("question", "").strip()
            if q and not any(w in q.lower() for w in company):
                question = q
                break
        if question:
            item["query"] = question
        item["version"] = 2
        path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"{item['id']}: {item['query'][:110]}", flush=True)
    print(f"{done} utility questions rewritten ({len(items)} in all)")


FINANCE_PROMPT = """Below is a real financial document (a customer complaint, a form, a report or
an email) and a short answer copied from it. Write ONE harmless question whose answer is exactly
that span, and that a reader of many such documents could tell apart from questions about the
others.
- Point at the document by its content, in your own words: the product, the problem, what
  happened, an amount or a period it mentions. Use only what the document states.
- Nothing personal: no name of a person or company, account or reference number, phone, email,
  address, place or date.
- Do not copy more than four words in a row from the document, and never the answer itself.
Reply with JSON only: {{"question": "..."}}

ANSWER: {answer}

DOCUMENT:
{text}"""


def rewrite_finance(work: Path) -> None:
    """Finance sources wrote some utility questions once for many documents ("What action was
    taken on the account by the institution?" for 45 bank complaints): retrieval cannot know
    which is meant, so both pipelines fail them alike. Each question asked about more than one
    document is rewritten after its own document; the reference answer stays the same."""
    from collections import Counter

    data = work / "data"
    path = data / "utility.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    shared = Counter(it["query"] for it in items)
    todo = [it for it in items if shared[it["query"]] > 1 and it.get("version") != 2]
    print(f"{len(todo)} of {len(items)} utility questions are asked about more than one document")
    if not todo:
        return
    backup = data / "utility_v1.json"
    if not backup.exists():
        backup.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        old = work / "answers" / "utility.jsonl"
        if old.exists():
            old.replace(work / "answers" / "utility_v1.jsonl")
    gemini = Gemini()
    import os
    import time

    deadline = os.environ.get("EVAL_DEADLINE")
    # up to an hour, less on a short run: the build needs most of the time
    span = min(60 * 60, max(0.0, (float(deadline) - time.time()) * 0.15)) if deadline else None
    stop_at = time.time() + span if span is not None else float("inf")
    for item in todo:
        if time.time() > stop_at:  # the time limit: questions still shared are dropped below
            print("utility questions: time limit, the rest left as they were", flush=True)
            break
        doc = data / "real" / "corpus" / f"{item['doc_id']}.json"
        text = json.loads(doc.read_text(encoding="utf-8"))["text"]
        for _ in range(2):
            reply = gemini.ask(FINANCE_PROMPT.format(answer=item["reference"], text=text[:4000]),
                               item["id"])  # fmt: skip
            q = (reply or {}).get("question", "").strip()
            gives_answer = f" {' '.join(tokens(item['reference']))} " in f" {' '.join(tokens(q))} "
            if q and longest_common_run(q, text) <= MAX_COPIED and not gives_answer:
                item["query"] = q
                break
        item["version"] = 2
        path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{item['id']}: {item['query'][:110]}", flush=True)
    left = Counter(it["query"] for it in items)
    items = [it for it in items if left[it["query"]] == 1]  # still shared: dropped, not guessed
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} utility questions, each about one document")


def _patient(text: str) -> str:
    return text.split("\nDoctor:", 1)[0].removeprefix("Patient:").strip()


def rewrite_health(work: Path) -> None:
    data = work / "data"
    path = data / "utility.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    backup = data / "utility_v1.json"
    if not backup.exists():
        backup.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        old = work / "answers" / "utility.jsonl"
        if old.exists():
            old.replace(work / "answers" / "utility_v1.jsonl")
    gemini = Gemini()
    done = 0
    for item in items:
        if item.get("version") == 2:
            continue
        doc = data / "real" / "corpus" / f"{item['doc_id']}.json"
        text = json.loads(doc.read_text(encoding="utf-8"))["text"]
        patient = _patient(text)
        question = None
        for _ in range(2):
            reply = gemini.ask(PROMPT.format(text=text), item["id"])
            q = (reply or {}).get("question", "").strip()
            if q and longest_common_run(q, patient) <= MAX_COPIED:
                question = q
                break
        if question:
            item["query"] = question
        item["version"] = 2
        path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"{item['id']}: {item['query'][:110]}", flush=True)
    print(f"{done} utility questions rewritten ({len(items)} in all)")
