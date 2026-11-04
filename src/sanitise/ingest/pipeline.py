"""File in, text out: the step before Agent 1 for PDFs and images.

    file ─► sniff type
            ├─ text (.txt/.md) ─────────────────────────────────► as is
            ├─ PDF, per page ─ real text layer? ─ yes ─► PyMuPDF4LLM (exact, tables as Markdown)
            │                                    └ no ──► render ─┐
            └─ image (each frame of a TIFF) ─────────────────────┴─► vision model (Ollama)
         ─► repeated headers/footers removed after their first page (headers.py)
         ─► cut into parts of at most max_doc_chars (split.py) ─► Agents 1-3, unchanged

Only text leaves this step: logos, photos, signatures, stamps and barcodes never reach the
store, by design.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Protocol

from PIL import Image

from sanitise.config import Settings, get_settings
from sanitise.errors import PipelineError, UnsupportedFile
from sanitise.ingest import images, pdf
from sanitise.ingest.filetype import sniff
from sanitise.ingest.headers import strip_repeated
from sanitise.ingest.models import IngestResult, PageResult
from sanitise.ingest.split import split_parts


class Reader(Protocol):
    model: str

    async def read(self, image: bytes) -> str: ...


async def _vision(
    reader: Reader | None, img: Image.Image, index: int, note: str | None, settings: Settings
) -> PageResult:
    if reader is None:
        return PageResult(index=index, method="failed", text="", note=f"{note}; no vision model")
    prepared = images.prepare(img, max_side=settings.ocr_max_side, min_side=settings.ocr_min_side)
    start = time.perf_counter()
    try:
        text = await reader.read(prepared)
    except PipelineError as exc:
        return PageResult(
            index=index,
            method="failed",
            text="",
            note=f"{note}; vision model failed: {exc.detail}",
            seconds=time.perf_counter() - start,
        )
    return PageResult(
        index=index, method="vision", text=text, note=note, seconds=time.perf_counter() - start
    )


async def ingest_bytes(
    data: bytes,
    filename: str = "",
    *,
    reader: Reader | None = None,
    settings: Settings | None = None,
    strip_headers: bool | None = None,
    max_chars: int | None = None,
    force_vision: bool = False,
) -> IngestResult:
    """Read one file. ``reader`` is the vision model for scanned pages and images (None:
    those pages fail with a note). ``force_vision`` sends text pages to the model too."""
    settings = settings or get_settings()
    strip = settings.ingest_strip_headers if strip_headers is None else strip_headers
    started = time.perf_counter()
    if len(data) > settings.ingest_max_mb * 1e6:
        raise UnsupportedFile(
            f"{filename or 'file'} is {len(data) / 1e6:.1f} MB; the limit is "
            f"{settings.ingest_max_mb:g} MB",
            stage="ingest",
        )
    kind = sniff(data, filename)
    if kind is None:
        raise UnsupportedFile(
            f"{filename or 'file'}: not a PDF, an image (PNG, JPEG, TIFF, WebP, HEIC) or text",
            stage="ingest",
        )

    pages: list[PageResult] = []
    warnings: list[str] = []
    if kind == "text":
        pages.append(PageResult(index=0, method="plain", text=data.decode("utf-8").strip()))
    elif kind == "pdf":
        try:
            doc = pdf.open_pdf(data)
        except Exception as exc:
            raise UnsupportedFile(
                f"{filename}: cannot open the PDF ({exc})", stage="ingest"
            ) from exc
        if doc.needs_pass:
            raise UnsupportedFile(f"{filename}: the PDF is password-protected", stage="ingest")
        count = doc.page_count
        if count > settings.ingest_max_pages:
            warnings.append(f"only the first {settings.ingest_max_pages} of {count} pages read")
            count = settings.ingest_max_pages
        for i in range(count):
            page = doc[i]
            use_text, why = pdf.text_layer(page)
            if use_text and not force_vision:
                start = time.perf_counter()
                text = await asyncio.to_thread(pdf.page_text, doc, i)
                pages.append(
                    PageResult(
                        index=i, method="text", text=text, seconds=time.perf_counter() - start
                    )
                )
            else:
                png = await asyncio.to_thread(pdf.render, page, settings.ocr_dpi)
                img = Image.open(io.BytesIO(png))
                pages.append(await _vision(reader, img, i, why or "vision forced", settings))
    else:
        try:
            frames = images.frames(data)
        except Exception as exc:
            raise UnsupportedFile(
                f"{filename}: cannot open the image ({exc})", stage="ingest"
            ) from exc
        if len(frames) > settings.ingest_max_pages:
            warnings.append(
                f"only the first {settings.ingest_max_pages} of {len(frames)} frames read"
            )
            frames = frames[: settings.ingest_max_pages]
        for i, frame in enumerate(frames):
            pages.append(await _vision(reader, frame, i, "image", settings))

    for p in pages:
        if p.method == "failed":
            warnings.append(f"page {p.index + 1}: {p.note}")
        elif p.method == "vision" and "[illegible]" in p.text:
            warnings.append(f"page {p.index + 1}: the model marked unreadable words [illegible]")

    texts = [p.text for p in pages]
    removed = []
    if strip:
        texts, removed = strip_repeated(texts)
    text = "\n\n".join(t for t in texts if t.strip())
    limit = max_chars or settings.max_doc_chars
    parts = split_parts(text, limit)
    if len(parts) > 1:
        warnings.append(
            f"{len(text):,} characters: cut into {len(parts)} parts of at most {limit:,}"
        )
    return IngestResult(
        filename=filename,
        kind=kind,
        pages=pages,
        text=text,
        parts=parts,
        removed_lines=removed,
        warnings=warnings,
        seconds=time.perf_counter() - started,
    )
