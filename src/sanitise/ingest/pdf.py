"""PDF pages: read the text layer when it is real, otherwise render the page for the vision model.

A page's text layer is used only when it is trustworthy:

* it has visible text (scanners often add an invisible OCR layer of poor quality; those pages
  are read again by the vision model);
* it has enough characters (a scan with only a stamped date on it is a scan);
* the words look like words (a broken font mapping gives "ÿþ#@..." garbage);
* it is not a full-page image with a few words on top.

Text pages go through PyMuPDF4LLM in its plain mode (no layout model): it keeps the printed
lines as lines (labels, letterheads and footers stay separate, which header removal needs) and
turns ruled tables into Markdown tables; if that fails, the plain text layer is used.
"""

from __future__ import annotations

import logging
import re

import pymupdf

log = logging.getLogger(__name__)

MIN_CHARS = 30
MIN_WORDLIKE = 0.6
IMAGE_COVER = 0.6  # share of the page under images that marks a scan
SCAN_MAX_CHARS = 200  # ...when the page also has fewer visible characters than this


def open_pdf(data: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=data, filetype="pdf")


def text_layer(page: pymupdf.Page) -> tuple[bool, str | None]:
    """(use the text layer?, why not)."""
    trace = page.get_texttrace()
    total = sum(len(s["chars"]) for s in trace)
    if total == 0:
        return False, "no text layer"
    invisible = sum(len(s["chars"]) for s in trace if s["type"] == 3)
    if invisible > 0.8 * total:
        return False, "only a hidden OCR layer"
    text = page.get_text()
    chars = len(re.sub(r"\s", "", text))
    if chars < MIN_CHARS:
        return False, "almost no text"
    words = re.findall(r"[^\W\d_]{2,}", text)
    if words:
        wordlike = sum(1 for w in words if len(w) <= 25 and re.search(r"[aeiouyAEIOUY]", w)) / len(
            words
        )
        if wordlike < MIN_WORDLIKE:
            return False, "text layer looks garbled"
    area = abs(page.rect)
    covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    if area and covered / area > IMAGE_COVER and chars < SCAN_MAX_CHARS:
        return False, "full-page image with little text"
    return True, None


_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_HEADING = re.compile(r"^#{1,6}\s+", re.M)
# underline, superscript etc. come out as HTML tags
_TAGS = re.compile(r"</?(?:u|sup|sub|s|i|b|em|strong|span|mark)\b[^>]*>", re.I)


def tidy_markdown(md: str) -> str:
    """Keep tables; drop emphasis marks, inline HTML tags and heading hashes, which the agents
    do not need."""
    md = _TAGS.sub("", md)
    md = _BOLD.sub(r"\2", md)
    md = _ITALIC.sub(r"\2", md)
    md = _HEADING.sub("", md)
    md = md.replace("<br>", " ")
    md = "\n".join(line.rstrip() for line in md.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def page_text(doc: pymupdf.Document, index: int) -> str:
    try:
        import pymupdf4llm

        # the layout mode reflows printed lines into paragraphs and loads a model; not wanted
        pymupdf4llm.use_layout(False)
        md = pymupdf4llm.to_markdown(doc, pages=[index], show_progress=False)
        text = tidy_markdown(md)
        if text:
            return text
    except Exception as exc:  # layout model missing or failed: the plain text layer is fine
        log.warning("pymupdf4llm failed on page %d (%s); using the plain text layer", index, exc)
    return tidy_markdown(doc[index].get_text("text", sort=True))


def render(page: pymupdf.Page, dpi: int) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")
