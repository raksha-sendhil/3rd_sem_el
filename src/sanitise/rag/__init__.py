"""Retrieval-augmented question answering over the stored documents (docs/phase3_plan.md).

================  ==========  ==================================================
module            owner       role
================  ==========  ==================================================
store.py          Person C    dual-track storage (original + SanitisedRecord)
retrieve.py       Person C    retrieval over the originals, doc IDs only
answer.py         Person C    ``answer`` / ``compare`` for both pipelines
sessions.py       Person C    session traces
answer_prompt.py  Person C    ``build_answer_prompt``
================  ==========  ==================================================
"""
