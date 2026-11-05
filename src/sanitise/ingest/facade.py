"""One file to text plus its format, synchronously (the document loader uses this).

PDFs and images go through the page-by-page pipeline (pipeline.py: text layers read directly,
scanned pages and photos through the vision model, repeated headers removed) when it can run here;
if no vision model answers, or this is called inside a running event loop (use ``ingest_bytes``
there), the simpler parsers take over: pdf_parser.py (pypdf) and image_parser.py (Gemini Vision,
Tesseract or EasyOCR).

Supports:
- PDFs (searchable and scanned)
- Images (printed reports, handwritten doctor notes, prescriptions)
- Electronic Health Records (HL7 v2, FHIR JSON, CDA/CCDA XML)
- Electrical Telemetry (ECG interval reports)
- Text and structured JSON/CSV records
"""

from __future__ import annotations

import asyncio
import io
import mimetypes
from pathlib import Path
from typing import Any

from sanitise.config import Settings, get_settings
from sanitise.ingest.electronic_parser import parse_electronic
from sanitise.ingest.image_parser import SUPPORTED_IMAGE_SUFFIXES, parse_image
from sanitise.ingest.pdf_parser import parse_pdf
from sanitise.loader import Document, make_document

SUPPORTED_INGEST_SUFFIXES = (
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
    ".hl7",
    ".fhir",
    ".xml",
)


def ingest_file(
    path_or_bytes: str | Path | bytes | io.BytesIO,
    *,
    filename: str | None = None,
    settings: Settings | None = None,
) -> Document:
    """Ingest a file of any supported modality and wrap it as a sanitized Document."""
    cfg = settings or get_settings()
    text, source_fmt = extract_text_multimodal(path_or_bytes, filename=filename, settings=cfg)
    src_name = filename or (
        str(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else "upload"
    )
    return make_document(
        text, source=src_name, source_format=source_fmt, max_chars=cfg.max_doc_chars
    )


def extract_text_multimodal(
    source: str | Path | bytes | io.BytesIO,
    *,
    filename: str | None = None,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Extract raw text from PDF, image, handwritten document, or electronic record."""
    cfg = settings or get_settings()
    suffix = ""

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            suffix = p.suffix.lower()
            filename = filename or p.name
    elif filename:
        suffix = Path(filename).suffix.lower()

    # 0. PDFs and images: the full pipeline first
    if suffix == ".pdf" or suffix in SUPPORTED_IMAGE_SUFFIXES:
        read = _read_with_pipeline(source, filename or f"upload{suffix}", cfg)
        if read is not None:
            return read, "pdf" if suffix == ".pdf" else "image"

    # 1. PDF Documents
    if suffix == ".pdf":
        return parse_pdf(source), "pdf"

    # 2. Image and Handwritten Documents
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        mime, _ = mimetypes.guess_type(filename or "image.jpg")
        return parse_image(source, mime_type=mime, settings=cfg), "image"

    # 3. HL7 v2 Message
    if suffix == ".hl7":
        text, fmt = parse_electronic(source)
        return text, fmt

    # 4. FHIR JSON
    if suffix == ".fhir":
        text, fmt = parse_electronic(source)
        return text, fmt

    # 5. Clinical XML (CDA/CCDA)
    if suffix == ".xml":
        text, fmt = parse_electronic(source)
        return text, fmt

    # 6. JSON files - could be FHIR bundle, or standard fixture / chatdoctor / enron
    if suffix == ".json":
        # Check if content is FHIR / electronic or standard fixture
        try:
            if isinstance(source, (str, Path)):
                raw = Path(source).read_text(encoding="utf-8-sig")
            elif isinstance(source, (bytes, io.BytesIO)):
                raw = (source if isinstance(source, bytes) else source.getvalue()).decode(
                    "utf-8-sig", errors="ignore"
                )
            else:
                raw = str(source)

            import json

            data = json.loads(raw)
            if isinstance(data, dict) and ("resourceType" in data or "entry" in data):
                text, fmt = parse_electronic(data)
                return text, fmt
        except Exception:
            pass

    # 7. Fallback to reading as text and checking content
    if isinstance(source, (str, Path)) and Path(source).is_file():
        text_content = Path(source).read_text(encoding="utf-8-sig", errors="ignore")
    elif isinstance(source, bytes):
        text_content = source.decode("utf-8-sig", errors="ignore")
    elif isinstance(source, io.BytesIO):
        text_content = source.getvalue().decode("utf-8-sig", errors="ignore")
    else:
        text_content = str(source)

    # Check for HL7, XML or FHIR JSON inside plain text (no file name to go by)
    stripped = text_content.lstrip()
    if (
        stripped.startswith("MSH")
        or stripped.startswith("<")
        or (stripped.startswith(("{", "[")) and '"resourceType"' in stripped)
    ):
        return parse_electronic(stripped)

    return text_content, "text"


def _read_with_pipeline(source: Any, filename: str, settings: Settings) -> str | None:
    """The pipeline's text, or None when it cannot run here or a page could not be read."""
    try:
        asyncio.get_running_loop()
        return None  # inside async code: the caller should await ingest_bytes itself
    except RuntimeError:
        pass
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif isinstance(source, io.BytesIO):
        data = source.getvalue()
    elif isinstance(source, bytes):
        data = source
    else:
        return None

    from sanitise.errors import PipelineError
    from sanitise.ingest.pipeline import ingest_bytes
    from sanitise.ingest.vision import make_reader

    async def run():
        reader = make_reader(settings)
        try:
            return await ingest_bytes(
                data, filename, reader=reader, settings=settings, max_chars=10**9
            )
        finally:
            await reader.aclose()

    try:
        result = asyncio.run(run())
    except (PipelineError, ValueError):
        return None
    if any(page.method == "failed" for page in result.pages):
        return None
    return result.text
