"""PDF / image input (sanitise.ingest): routing, header removal, splitting, metrics, API."""

from __future__ import annotations

import base64
import io

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from sanitise.api.app import create_app
from sanitise.config import Settings
from sanitise.errors import UnsupportedFile
from sanitise.ingest import MockVisionReader, ingest_bytes
from sanitise.ingest.filetype import sniff
from sanitise.ingest.headers import strip_repeated
from sanitise.ingest.metrics import detail_hits, normalise, score_text, strip_scores
from sanitise.ingest.split import split_parts
from sanitise.ingest.vision import tidy

SETTINGS = Settings(llm_backend="mock", max_doc_chars=2000)


def text_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        box = pymupdf.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_htmlbox(box, text.replace("\n", "<br>"))
    return doc.tobytes()


def image_bytes(fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (900, 600), "white")
    ImageDraw.Draw(img).text((40, 40), "Name: Ravi Kumar", fill="black")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def scanned_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=image_bytes())
    return doc.tobytes()


# ------------------------------------------------------------------ file type


def test_sniff() -> None:
    assert sniff(text_pdf(["hello"]), "x.bin") == "pdf"
    assert sniff(image_bytes("PNG"), "") == "image"
    assert sniff(image_bytes("JPEG"), "photo") == "image"
    assert sniff(b"Patient seen today.", "note.txt") == "text"
    assert sniff(b"\x00\x01\xff\xfe binary", "x.exe") is None


# ------------------------------------------------------------------ routing


async def test_text_pdf_is_read_directly() -> None:
    body = "Name: Ravi Kumar\nDiagnosis: Dengue fever with thrombocytopenia\nAdvice: fluids"
    reader = MockVisionReader()
    result = await ingest_bytes(text_pdf([body]), "a.pdf", reader=reader, settings=SETTINGS)
    assert [p.method for p in result.pages] == ["text"]
    assert reader.calls == 0
    assert "Ravi Kumar" in result.text and "Dengue fever" in result.text


async def test_scanned_pdf_and_image_go_to_the_vision_model() -> None:
    reader = MockVisionReader("Name: Ravi Kumar\nDx: viral fever")
    scan = await ingest_bytes(scanned_pdf(), "s.pdf", reader=reader, settings=SETTINGS)
    assert [p.method for p in scan.pages] == ["vision"]
    assert scan.pages[0].note == "no text layer"
    photo = await ingest_bytes(image_bytes(), "p.jpg", reader=reader, settings=SETTINGS)
    assert [p.method for p in photo.pages] == ["vision"]
    assert photo.text == "Name: Ravi Kumar\nDx: viral fever"
    assert reader.calls == 2


async def test_without_a_vision_model_scanned_pages_fail_with_a_warning() -> None:
    result = await ingest_bytes(scanned_pdf(), "s.pdf", reader=None, settings=SETTINGS)
    assert result.pages[0].method == "failed"
    assert result.warnings


async def test_unsupported_file() -> None:
    with pytest.raises(UnsupportedFile):
        await ingest_bytes(b"\x00\x01\x02 not a document", "x.exe", settings=SETTINGS)


async def test_long_text_is_cut_into_parts() -> None:
    body = "\n\n".join(f"Section {i}: " + "word " * 60 for i in range(12))
    result = await ingest_bytes(body.encode(), "long.txt", settings=SETTINGS)
    assert len(result.parts) > 1
    assert all(len(p) <= 2000 for p in result.parts)


# ------------------------------------------------------------------ header removal


def test_repeated_header_and_footer_removed_after_first_page() -> None:
    pages = [
        "CITY HOSPITAL\nName: Ravi Kumar  UHID: 1234\nDate: 12/03/2024\nDx: fever\nPage 1 of 3",
        "CITY HOSPITAL\nName: Ravi Kumar  UHID: 1234\nDate: 13/03/2024\nCourse: ok\nPage 2 of 3",
        "City Hospital\nName: Ravi Kumar UHID: 1234\nAdvice: rest\nPage 3 of 3",
    ]
    out, removed = strip_repeated(pages)
    assert out[0] == pages[0]  # the first page keeps everything
    assert "CITY HOSPITAL" not in out[1] and "Page 2 of 3" not in out[1]
    assert "Date: 13/03/2024" in out[1]  # different digits: never merged
    assert out[2] == "Advice: rest"
    assert len(removed) == 6


def test_single_page_and_unique_lines_untouched() -> None:
    assert strip_repeated(["A\nB"]) == (["A\nB"], [])
    pages = ["Report one\nresult 1", "Report two\nresult 2"]
    assert strip_repeated(pages)[0] == pages


def test_page_never_emptied() -> None:
    pages = ["CITY HOSPITAL\nName: Ravi", "CITY HOSPITAL\nName: Ravi"]
    out, removed = strip_repeated(pages)
    assert out == pages and removed == []


# ------------------------------------------------------------------ splitting


def test_split_parts_at_paragraphs() -> None:
    text = "\n\n".join("x" * 300 for _ in range(10))
    parts = split_parts(text, 1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(parts).replace("\n", "") == "x" * 3000
    assert split_parts("short", 1000) == ["short"]


# ------------------------------------------------------------------ vision output tidy


def test_tidy_removes_fences_and_runaway_repeats() -> None:
    raw = "```markdown\nName: A\nrow\nrow\nrow\nrow\nrow\n```"
    assert tidy(raw) == "Name: A\nrow\nrow"


# ------------------------------------------------------------------ metrics


def test_normalise_ignores_formatting() -> None:
    md = "| **Test** | Result |\n|---|---|\n| Hb | 12.1 |\n[signature]"
    assert normalise(md) == normalise("Test Result\nHb 12.1")


def test_cer_and_details() -> None:
    assert score_text("Ravi Kumar", "Ravi Kumar").cer == 0
    s = score_text("Ravl Kumar", "Ravi Kumar")
    assert s.char_errors == 1 and s.cer == pytest.approx(0.1)
    assert detail_hits("Phone +91 9845012345, Ravl Kumar", ["+91 98450 12345", "Ravi Kumar"]) == [
        True,
        False,
    ]


def test_strip_scores() -> None:
    expected = [{"page": 1, "line": "CITY HOSPITAL"}, {"page": 1, "line": "Page 2 of 2"}]
    removed = [
        {"page": 1, "line": "CITY HOSPITAL"},
        {"page": 1, "line": "Page 2 of 2"},
        {"page": 1, "line": "UHID: 1234"},  # a duplicate of kept text: safe
        {"page": 1, "line": "Diagnosis: fever"},  # gone from the text: information lost
    ]
    assert strip_scores(removed, expected, "CITY HOSPITAL UHID: 1234") == (2, 1, 2)


# ------------------------------------------------------------------ API


def test_ingest_endpoint() -> None:
    with TestClient(create_app(SETTINGS)) as client:
        pdf = text_pdf(["Name: Ravi Kumar<br>Diagnosis: Dengue fever with thrombocytopenia"])
        body = {"filename": "a.pdf", "content_base64": base64.b64encode(pdf).decode()}
        r = client.post("/v1/ingest", json=body)
        assert r.status_code == 200, r.text
        assert "Ravi Kumar" in r.json()["text"]
        assert r.json()["pages"][0]["method"] == "text"
        photo = {"filename": "p.jpg", "content_base64": base64.b64encode(image_bytes()).decode()}
        r = client.post("/v1/ingest", json=photo)
        assert r.json()["pages"][0]["method"] == "vision"  # mock reader in mock mode
        bad = client.post("/v1/ingest", json={"filename": "x", "content_base64": "%%%"})
        assert bad.status_code == 415
        assert bad.json()["error"]["code"] == "UNSUPPORTED_FILE"


# ------------------------------------------------------------------ vision request


async def test_vision_reader_sends_the_image_to_ollama() -> None:
    import json

    import httpx

    from sanitise.ingest import VisionReader

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "```\nName: Ravi Kumar\n```"}})

    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    reader = VisionReader(model="qwen3-vl:8b-instruct", http=http)
    assert await reader.read(b"jpeg-bytes") == "Name: Ravi Kumar"
    message = seen["messages"][0]
    assert seen["model"] == "qwen3-vl:8b-instruct" and seen["options"]["temperature"] == 0.0
    assert message["images"] == [base64.b64encode(b"jpeg-bytes").decode()]
    assert "[illegible]" in message["content"]
    await http.aclose()


def test_word_f1_ignores_order_and_hyphenation() -> None:
    gold = "INVOICE NO PEGIV-1030765\nTOTAL 193.00"
    s = score_text("TOTAL 193.00\nINVOICE NO PEGIV-1030765", gold)
    assert s.word_f1 == 1.0 and s.cer > 0  # same words, other order
    assert score_text("the constitutional question", "the consti-\ntutional question").cer == 0


def test_tidy_markdown_drops_inline_tags() -> None:
    from sanitise.ingest.pdf import tidy_markdown

    md = '## **Definitions**\n<u>Agreement</u> means _this_ contract<sup>1</sup>'
    assert tidy_markdown(md) == "Definitions\nAgreement means this contract1"
