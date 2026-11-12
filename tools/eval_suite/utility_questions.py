"""Utility questions that ask about a record's content, not its wording.

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
from eval_suite.metrics import longest_common_run

PROMPT = """Below is a patient's message to an online doctor and the doctor's reply.
Write ONE question that a clinician could ask a collection of such records to find the advice
given in this case. Describe the case only by its medical content, in your own words: who the
patient is in general terms ("a woman in her 30s", "a parent asking about a young child"), the
main symptoms or condition, and any test result the patient mentions. Do not copy more than
four words in a row from the message. No names, places, dates or contact details. End the
question with: "What did the doctor advise?"
Reply with JSON only: {{"question": "..."}}

{text}"""

MAX_COPIED = 6  # words in a row from the patient's message


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
