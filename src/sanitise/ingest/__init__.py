"""Document input: turn a file into text for Agents 1-3.

Two ways in, one pipeline underneath:

* ``ingest_bytes`` (async, pipeline.py): PDFs page by page (text layer read directly, scanned
  pages through a vision model), photos and scans, electronic health records (HL7 v2, FHIR, CDA
  XML), repeated headers/footers removed, long text cut into parts. The API and the tools use it.
* ``ingest_file`` / ``extract_text_multimodal`` (sync, facade.py): one file to text plus its
  format, used by the document loader; PDFs and images go through the same pipeline when it can
  run, otherwise through the simpler parsers (pdf_parser.py, image_parser.py).
"""

from sanitise.ingest.facade import SUPPORTED_INGEST_SUFFIXES, extract_text_multimodal, ingest_file
from sanitise.ingest.models import IngestResult, PageResult
from sanitise.ingest.pipeline import ingest_bytes
from sanitise.ingest.vision import GeminiVisionReader, MockVisionReader, VisionReader, make_reader

__all__ = [
    "SUPPORTED_INGEST_SUFFIXES",
    "GeminiVisionReader",
    "IngestResult",
    "MockVisionReader",
    "PageResult",
    "VisionReader",
    "extract_text_multimodal",
    "ingest_bytes",
    "ingest_file",
    "make_reader",
]
