"""RAG answer prompt, shared by the baseline and the protected pipeline.

OWNER: Person C (Model, Platform and Integration). PLACEHOLDER: replace the prompt;
keep the signature (docs/phase3_plan.md, Person C, item 2).

``build_answer_prompt(question, contexts)`` returns ``(system, user)``. ``contexts`` are
``(doc_id, text)`` pairs in rank order: original texts for the baseline pipeline, sanitised
texts for the protected one. The same prompt serves both, so any difference in the answers
comes only from the text the model is given. The user message is exactly what the dashboard's
payload inspector shows.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You answer questions about a collection of records using only the records given in CONTEXT.
If the records do not contain the answer, say that they do not. Keep placeholder tokens such \
as [PERSON_1] exactly as written. Answer in two or three sentences.
"""


def build_answer_prompt(question: str, contexts: list[tuple[str, str]]) -> tuple[str, str]:
    blocks = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in contexts) or "(no records)"
    return SYSTEM_PROMPT, f"CONTEXT:\n{blocks}\n\nQUESTION: {question}"
