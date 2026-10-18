"""sanitise - privacy-preserving document sanitisation for RAG.

Phase 2 ships Agent 1, the Sensitive Information Detector:

    from sanitise import detect, load_document
    from sanitise.llm import MockLLM

    doc = load_document("data/fixtures/case_07.json")
    result = await detect(doc.doc_id, doc.text, MockLLM())
"""

from sanitise.agents.detector.pipeline import detect, detect_with_trace
from sanitise.loader import Document, load_document
from sanitise.schemas import PIPELINE_VERSION

__version__ = PIPELINE_VERSION

__all__ = ["PIPELINE_VERSION", "Document", "detect", "detect_with_trace", "load_document"]
