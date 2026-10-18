from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sanitise.config import Settings
from sanitise.errors import DocTooLong
from sanitise.loader import (
    compute_doc_id,
    extract_text,
    iter_documents,
    load_document,
    make_document,
)


def test_doc_id_is_first_16_hex_of_sha256() -> None:
    text = "My son has been experiencing left-sided abdominal pain"
    assert compute_doc_id(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    doc = make_document(text)
    assert doc.doc_id == compute_doc_id(text) and doc.sha256.startswith(doc.doc_id)


def test_rejects_documents_over_limit() -> None:
    make_document("x" * 2000)
    with pytest.raises(DocTooLong) as info:
        make_document("x" * 2001)
    assert info.value.stage == "load"


def test_text_is_not_modified(tmp_path: Path) -> None:
    raw = "  line one\r\nline two\n"
    path = tmp_path / "doc.txt"
    path.write_bytes(raw.encode("utf-8"))
    assert load_document(path).text == raw


@pytest.mark.parametrize(
    ("record", "expected", "fmt"),
    [
        ({"text": "hello"}, "hello", "fixture"),
        (
            {"instruction": "be a doctor", "input": "I have a cough", "output": "Hi, ..."},
            "I have a cough",
            "chatdoctor",
        ),
        ({"file": "allen-p/1.", "message": "Subject: hi\n\nbody"}, "Subject: hi\n\nbody", "enron"),
        ({"body": "email body"}, "email body", "enron"),
    ],
)
def test_extract_text_formats(record: dict, expected: str, fmt: str) -> None:
    assert extract_text(record) == (expected, fmt)


def test_chatdoctor_can_include_answer() -> None:
    text, _ = extract_text({"input": "Q", "output": "A"}, include_answer=True)
    assert text == "Q\n\nA"


def test_unknown_record_shape() -> None:
    with pytest.raises(ValueError, match="unrecognised record"):
        extract_text({"foo": "bar"})


def test_multi_record_json_and_index(tmp_path: Path) -> None:
    path = tmp_path / "many.json"
    path.write_text(
        json.dumps([{"input": "a", "output": ""}, {"input": "b", "output": ""}]), encoding="utf-8"
    )
    assert load_document(path, index=1).text == "b"
    assert load_document(path, index=1).source.endswith("#1")
    with pytest.raises(IndexError):
        load_document(path, index=5)


def test_iter_documents_directory_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"text": "first"}), encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"text": "second"}\n{"text": "third"}\n', encoding="utf-8")
    (tmp_path / "ignored.md").write_text("nope", encoding="utf-8")
    assert [d.text for d in iter_documents(tmp_path)] == ["first", "second", "third"]


def test_iter_documents_csv(tmp_path: Path) -> None:
    path = tmp_path / "emails.csv"
    path.write_bytes(b'file,message\nx/1.,"Subject: a\n\nhello"\n')
    [doc] = list(iter_documents(path))
    assert doc.text == "Subject: a\n\nhello" and doc.source_format == "enron"


def test_iter_documents_skip_too_long(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps({"text": "ok"}) + "\n" + json.dumps({"text": "y" * 50}) + "\n", encoding="utf-8"
    )
    settings = Settings(max_doc_chars=10)
    with pytest.raises(DocTooLong):
        list(iter_documents(path, settings=settings))
    assert [d.text for d in iter_documents(path, settings=settings, skip_too_long=True)] == ["ok"]
