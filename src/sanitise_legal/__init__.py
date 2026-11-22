"""sanitise - privacy-preserving document sanitisation for RAG.

Phase 2 ships Agent 1, the Sensitive Information Detector:

    from sanitise_legal import detect, load_document
    from sanitise_legal.llm import MockLLM

    doc = load_document("data/fixtures/case_07.json")
    result = await detect(doc.doc_id, doc.text, MockLLM())
"""

from sanitise_legal.agents.detector.pipeline import detect, detect_with_trace
from sanitise_legal.loader import Document, load_document
from sanitise_legal.schemas import PIPELINE_VERSION

__version__ = PIPELINE_VERSION

__all__ = ["PIPELINE_VERSION", "Document", "detect", "detect_with_trace", "load_document"]
