"""Prepare a photo, scan or rendered page for the vision model.

Rotation from the camera (EXIF) is applied, colour is kept (vision models read better from the
original than from black-and-white), and the size is brought into a range the model reads well:
small phone images are enlarged, huge scans reduced (more pixels cost time, not accuracy).
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, ImageSequence

try:  # HEIC photos from iPhones, when pillow-heif is installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass


def frames(data: bytes) -> list[Image.Image]:
    """Every page of an image file (a multi-page TIFF has several)."""
    img = Image.open(io.BytesIO(data))
    out = []
    for frame in ImageSequence.Iterator(img):
        out.append(ImageOps.exif_transpose(frame.copy()))
    return out


def prepare(img: Image.Image, *, max_side: int = 2000, min_side: int = 1200) -> bytes:
    """JPEG bytes, RGB, longest side between ``min_side`` and ``max_side``."""
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    img = img.convert("RGB")
    longest = max(img.size)
    scale = 1.0
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min(min_side / longest, 3.0)
    if scale != 1.0:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
