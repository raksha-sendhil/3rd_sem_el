"""What the ingest step returns for one file."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# text:   a PDF page with a real text layer, read directly (exact)
# vision: a scanned page or image, transcribed by the vision model
# plain:  a .txt / .md file
# electronic: an electronic health record (HL7 v2, FHIR, CDA XML, ECG report) turned into text
# failed: the page could not be read (see note)
PageMethod = Literal["text", "vision", "plain", "electronic", "failed"]
FileKind = Literal["pdf", "image", "text", "ehr"]


class PageResult(BaseModel):
    index: int  # 0-based page number in the file
    method: PageMethod
    text: str
    seconds: float = 0.0
    note: str | None = None  # why a page went the way it did (e.g. "no text layer")


class RemovedLine(BaseModel):
    page: int
    line: str


class IngestResult(BaseModel):
    filename: str
    kind: FileKind
    # pdf / image / text, or the record format: hl7 / fhir / clinical_xml / ecg_telemetry
    source_format: str = ""
    doc_id: str = ""  # of ``text``, as the document loader computes it
    char_count: int = 0
    pages: list[PageResult]
    text: str  # all pages joined, repeated headers/footers removed when asked
    # ``text`` cut at paragraph boundaries into pieces the agents accept (max_doc_chars each)
    parts: list[str]
    removed_lines: list[RemovedLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    seconds: float = 0.0
