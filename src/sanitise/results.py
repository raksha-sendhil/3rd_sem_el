"""Reading and writing detector outputs and gold annotations on disk.

* ``sanitise detect --out out/`` writes one ``<doc_id>.json`` (a ``DetectorResult``) per document.
* ``sanitise score --gold data/gold/ --pred out/`` reads both directories back.

Gold files follow spec 8.3. Gold spans carry no ``method``; it is inferred as
``"rule"`` for spans with offsets and ``"llm"`` for offset-less (quasi) spans,
and offset-less spans without a rationale get ``"gold annotation"`` so they
pass ``SensitiveSpan`` validation. Scoring ignores both fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from sanitise.errors import OffsetOutOfRange, SchemaValidationFailed
from sanitise.llm.base import format_validation_error
from sanitise.schemas import DetectorOutput, DetectorResult, SensitiveSpan
from sanitise.validation import check_span_offsets

GOLD_RATIONALE = "gold annotation"


class GoldSpan(BaseModel):
    text: str
    category: str
    start: int | None = None
    end: int | None = None
    rationale: str | None = None


class GoldDocument(BaseModel):
    """One gold annotation file (spec 8.3)."""

    doc_id: str
    text: str
    spans: list[GoldSpan] = Field(default_factory=list)
    annotator: str | None = None
    reviewed_by: str | None = None

    def to_output(self) -> DetectorOutput:
        spans = [
            SensitiveSpan(
                text=s.text,
                category=s.category,
                start=s.start,
                end=s.end,
                method="rule" if s.start is not None else "llm",
                rationale=s.rationale or (None if s.start is not None else GOLD_RATIONALE),
            )
            for s in self.spans
        ]
        return DetectorOutput(doc_id=self.doc_id, spans=spans)


def _json_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".json" and p.is_file())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _invalid(path: Path, exc: ValidationError) -> SchemaValidationFailed:
    return SchemaValidationFailed(f"{path}: {format_validation_error(exc)}", stage="score")


def save_result(result: DetectorResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result.doc_id}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_gold_document(path: str | Path, *, check_offsets: bool = True) -> GoldDocument:
    path = Path(path)
    try:
        gold = GoldDocument.model_validate(_read_json(path))
        output = gold.to_output()  # validates categories and offsets
    except ValidationError as exc:
        raise _invalid(path, exc) from None
    if check_offsets:
        try:
            check_span_offsets(gold.text, output.spans, stage="score")
        except OffsetOutOfRange as exc:
            raise OffsetOutOfRange(f"{path}: {exc.detail}", stage="score") from None
    return gold


def load_gold(directory: str | Path, *, check_offsets: bool = True) -> list[DetectorOutput]:
    """All gold files in ``directory``, as ``DetectorOutput`` (what ``score()`` takes)."""
    return [
        load_gold_document(p, check_offsets=check_offsets).to_output()
        for p in _json_files(Path(directory))
    ]


def load_prediction(path: str | Path) -> DetectorOutput:
    """Accepts a ``DetectorResult`` (what ``detect --out`` writes) or a bare ``DetectorOutput``."""
    path = Path(path)
    data = _read_json(path)
    try:
        if isinstance(data, dict) and "output" in data and "meta" in data:
            return DetectorResult.model_validate(data).output
        return DetectorOutput.model_validate(data)
    except ValidationError as exc:
        raise _invalid(path, exc) from None


def load_predictions(directory: str | Path) -> list[DetectorOutput]:
    return [load_prediction(p) for p in _json_files(Path(directory))]
