"""Tell what a file is from its first bytes (the name is only a fallback)."""

from __future__ import annotations

from pathlib import PurePath

from sanitise.ingest.models import FileKind

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}
TEXT_SUFFIXES = {".txt", ".md", ".text"}
EHR_SUFFIXES = {".hl7", ".fhir", ".xml", ".json"}


def _looks_like_ehr(text: str) -> bool:
    head = text.lstrip()[:400]
    return (
        head.startswith("MSH|")
        or head.startswith("<?xml")
        or "<ClinicalDocument" in head
        or (head.startswith(("{", "[")) and '"resourceType"' in text[:5000])
    )


def sniff(data: bytes, filename: str = "") -> FileKind | None:
    """``pdf``, ``image``, ``ehr`` (HL7 / FHIR / CDA record), ``text`` or None (not supported)."""
    head = data[:16]
    if head.startswith(b"%PDF") or b"%PDF-" in data[:1024]:
        return "pdf"
    if (
        head.startswith(b"\x89PNG")
        or head.startswith(b"\xff\xd8\xff")  # JPEG
        or head[:4] in (b"II*\x00", b"MM\x00*")  # TIFF
        or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
        or head.startswith(b"BM")
        or head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc")
    ):
        return "image"
    suffix = PurePath(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES | EHR_SUFFIXES or suffix == "":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
        # an electronic health record (HL7 v2, FHIR JSON, CDA XML), whatever its name says
        if suffix in {".hl7", ".fhir"} or _looks_like_ehr(text):
            return "ehr"
        return "text" if suffix != ".json" and suffix != ".xml" else None
    return None
