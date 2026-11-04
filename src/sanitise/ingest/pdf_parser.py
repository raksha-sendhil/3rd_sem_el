"""PDF document parser for digital and scanned clinical documents.

Supports extracting text from PDF files using ``pypdf`` (primary) with fallbacks
to ``pdfplumber``, ``fitz`` (PyMuPDF), or basic uncompressed stream parsing.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def parse_pdf(source: str | Path | bytes | io.BytesIO) -> str:
    """Extract text from a PDF document.

    Args:
        source: Path to the PDF file, raw bytes, or BytesIO buffer.

    Returns:
        Cleaned text extracted from all pages of the document.
    """
    stream: io.BytesIO
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        raw_bytes = path.read_bytes()
        stream = io.BytesIO(raw_bytes)
    elif isinstance(source, bytes):
        raw_bytes = source
        stream = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        stream = source
        raw_bytes = source.getvalue()
    else:
        raise TypeError(f"Expected path, bytes, or BytesIO, got {type(source).__name__}")

    extracted: list[str] = []

    # 1. Try pypdf (preferred)
    try:
        import pypdf

        reader = pypdf.PdfReader(stream)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted.append(page_text.strip())
        if extracted:
            return _clean_pdf_text("\n\n".join(extracted))
    except ImportError:
        log.debug("pypdf not installed, trying alternative parsers")
    except Exception as exc:
        log.warning("pypdf extraction failed: %s", exc)

    # 2. Try pdfplumber
    try:
        import pdfplumber

        stream.seek(0)
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted.append(page_text.strip())
        if extracted:
            return _clean_pdf_text("\n\n".join(extracted))
    except ImportError:
        pass
    except Exception as exc:
        log.warning("pdfplumber extraction failed: %s", exc)

    # 3. Try fitz (PyMuPDF)
    try:
        import fitz

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                extracted.append(page_text.strip())
        if extracted:
            return _clean_pdf_text("\n\n".join(extracted))
    except ImportError:
        pass
    except Exception as exc:
        log.warning("fitz extraction failed: %s", exc)

    # 4. Fallback: stream pattern extraction for uncompressed text objects
    text_fallback = _extract_uncompressed_pdf_text(raw_bytes)
    if text_fallback.strip():
        return _clean_pdf_text(text_fallback)

    if not extracted and len(raw_bytes) > 0:
        # Document might be a scanned PDF containing only images
        return (
            "[SCANNED_PDF] The PDF appears to be a scanned document with no extractable text layer. "
            "Please use image OCR or provide an image/text version."
        )

    return ""


def _extract_uncompressed_pdf_text(data: bytes) -> str:
    """Best-effort regex extraction from uncompressed PDF streams."""
    lines: list[str] = []
    # Match text inside (text) Tj or [(t)(e)(x)(t)] TJ
    for match in re.finditer(rb"\(([^\(\)\\]*(?:\\.[^\(\)\\]*)*)\)\s*Tj", data):
        raw = match.group(1).decode("latin-1", errors="ignore")
        raw = raw.replace(r"\n", "\n").replace(r"\r", " ").replace(r"\t", " ")
        if raw.strip():
            lines.append(raw.strip())
    return "\n".join(lines)


def _clean_pdf_text(text: str) -> str:
    """Normalize whitespace and line endings from PDF extraction."""
    # Replace multiple spaces while keeping paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

