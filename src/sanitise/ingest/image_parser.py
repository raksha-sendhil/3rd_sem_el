"""Image document and handwriting parser for clinical notes and prescriptions.

Uses Gemini Vision (via httpx) when an API key is available to transcribe
handwritten clinical notes, doctor prescriptions, and printed lab reports.
Falls back to pytesseract or easyocr if installed locally.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
from pathlib import Path

import httpx

from sanitise.config import Settings, get_settings

log = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

CLINICAL_VISION_PROMPT = (
    "You are an expert clinical document transcriber.\n"
    "Transcribe all text, numbers, dates, patient details, doctor handwriting, prescriptions, "
    "medications, dosages, and clinical findings from this image exactly as written.\n"
    "- If the document is handwritten (e.g. prescription, doctor's progress note, chart), decipher "
    "the handwriting carefully and transcribe every word verbatim.\n"
    "- If it is a printed report, lab result, or discharge summary, preserve sections, labels, and values.\n"
    "- Output ONLY the transcribed clinical text. Do NOT wrap in conversational intro, markdown codeblocks, or commentary."
)


def parse_image(
    source: str | Path | bytes | io.BytesIO,
    *,
    mime_type: str | None = None,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Extract and transcribe text from an image (printed or handwritten).

    Args:
        source: Image path, bytes, or BytesIO.
        mime_type: MIME type (e.g. 'image/png', 'image/jpeg').
        settings: Application settings.
        client: Optional httpx.Client for reuse.

    Returns:
        Transcribed clinical text.
    """
    cfg = settings or get_settings()
    data: bytes

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        data = p.read_bytes()
        if not mime_type:
            mime_type = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    elif isinstance(source, bytes):
        data = source
        mime_type = mime_type or "image/jpeg"
    elif isinstance(source, io.BytesIO):
        data = source.getvalue()
        mime_type = mime_type or "image/jpeg"
    else:
        raise TypeError(f"Expected path, bytes, or BytesIO, got {type(source).__name__}")

    # 1. Try Gemini Vision if API key is present
    api_key = cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            transcription = _transcribe_with_gemini(
                data, mime_type, api_key.split(",")[0].strip(), client=client
            )
            if transcription.strip():
                return transcription.strip()
        except Exception as exc:
            log.warning("Gemini Vision transcription failed: %s; falling back to local OCR", exc)

    # 2. Try pytesseract if available
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img)
        if text.strip():
            return text.strip()
    except (ImportError, Exception) as exc:
        log.debug("pytesseract unavailable or failed: %s", exc)

    # 3. Try easyocr if available
    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(data, detail=0)
        if results:
            return "\n".join(results)
    except (ImportError, Exception) as exc:
        log.debug("easyocr unavailable or failed: %s", exc)

    # 4. Fallback when no OCR engine or API key is available
    if len(data) > 0:
        return (
            "[IMAGE_PROCESSED] Clinical document image received. "
            "To enable automatic transcription of handwritten notes, configure GEMINI_API_KEY "
            "or install pytesseract."
        )

    return ""


def _transcribe_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    *,
    model: str = "gemini-2.5-flash",
    client: httpx.Client | None = None,
) -> str:
    """Call the Gemini API with an inline image for transcription."""
    b64_data = base64.b64encode(image_bytes).decode("ascii")
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data,
                        }
                    },
                    {"text": CLINICAL_VISION_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    should_close = False
    if client is None:
        client = httpx.Client(timeout=60.0)
        should_close = True

    try:
        res = client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)
    finally:
        if should_close:
            client.close()

