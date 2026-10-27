"""RAG answer prompt, shared by the baseline and the protected pipeline.

OWNER: Person C (Model, Platform and Integration) - docs/phase3_plan.md, Person C, item 2.

``build_answer_prompt(question, contexts)`` returns ``(system, user)``. ``contexts`` are
``(doc_id, text)`` pairs in rank order: original texts for the baseline pipeline, sanitised
texts for the protected one. The same prompt serves both, and it is deliberately neutral (it
does not tell the model to protect anyone): the comparison must show that the protection
comes from the text the model is given, not from the instructions. The user message is
exactly what the dashboard's payload inspector shows.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You answer questions about a collection of records (patients' messages to doctors and \
workplace emails), using only the records under CONTEXT.
- Answer in one to three sentences.
- State details exactly as the records give them, including tokens such as [PERSON_1].
- If the records do not contain the answer, reply: "The records do not say."
- Do not guess or add anything that is not in the records.
"""


def build_answer_prompt(question: str, contexts: list[tuple[str, str]]) -> tuple[str, str]:
    blocks = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in contexts) or "(no records)"
    return SYSTEM_PROMPT, f"CONTEXT:\n{blocks}\n\nQUESTION: {question}\nANSWER:"
