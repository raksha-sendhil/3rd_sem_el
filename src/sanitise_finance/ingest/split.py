"""Cut a long document into parts the agents accept (``max_doc_chars`` each).

Cuts fall, in order of preference, at blank lines (paragraphs, sections, tables), at line
ends, then at spaces; a part is never cut inside a word unless one word is longer than
the limit.
"""

from __future__ import annotations

import re


def _pieces(text: str, limit: int) -> list[str]:
    """Paragraphs, with any longer than ``limit`` broken into lines, then words."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            out.append(para)
            continue
        for line in para.split("\n"):
            while len(line) > limit:
                cut = line.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                out.append(line[:cut].strip())
                line = line[cut:].strip()
            if line.strip():
                out.append(line)
    return out


def split_parts(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        joined = f"{current}\n\n{piece}" if current else piece
        if len(joined) <= max_chars:
            current = joined
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts
