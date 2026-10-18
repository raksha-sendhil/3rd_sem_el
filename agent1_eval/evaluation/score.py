"""
score.py - Agent 1 scoring harness (Person D).

Match rules (frozen on Day 1, spec section 3.2, plus the one-to-one rule below):

  1. Explicit spans (both gold and predicted have start/end):
       match  <=>  character ranges overlap  AND  category is equal.
       Partial overlap counts. Category mismatch does not.
  2. Offset-less spans (start or end is None; i.e. LLM-inferred quasi-identifiers):
       match  <=>  category is equal, within the same document. Offsets are ignored.
  3. An explicit span never matches an offset-less span.
  4. ONE-TO-ONE: each gold span can be credited by at most one prediction and each
     prediction can credit at most one gold span. Without this, a single huge
     predicted span could "hit" many gold spans, or 5 duplicate predictions could
     all count as true positives.  [Needs team sign-off - see docs/annotation_guide.md]

Metrics are micro-averaged over all documents. Division by zero gives 0.0.
Recall is the primary figure (a false negative is a leak); precision is alongside.

Offsets are Python string indices, end-exclusive.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from schemas import DetectorOutput, SensitiveSpan

log = logging.getLogger("score")


class SpanMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int
    per_category: dict[str, dict[str, float]]


# --------------------------------------------------------------------------- matching
def _has_offsets(s: SensitiveSpan) -> bool:
    return s.start is not None and s.end is not None


def _overlap(p: SensitiveSpan, g: SensitiveSpan) -> int:
    """Length of the character overlap (0 if none). Both must have offsets."""
    assert p.start is not None and p.end is not None
    assert g.start is not None and g.end is not None
    return max(0, min(p.end, g.end) - max(p.start, g.start))


@dataclass
class DocMatch:
    doc_id: str
    tp: list[tuple[SensitiveSpan, SensitiveSpan]] = field(default_factory=list)  # (pred, gold)
    fp: list[SensitiveSpan] = field(default_factory=list)
    fn: list[SensitiveSpan] = field(default_factory=list)


def match_document(pred: DetectorOutput | None, gold: DetectorOutput) -> DocMatch:
    """One-to-one matching of one document's predictions against its gold."""
    preds = list(pred.spans) if pred else []
    golds = list(gold.spans)
    result = DocMatch(doc_id=gold.doc_id)

    # Candidate (score, pred_idx, gold_idx). Higher score is matched first.
    candidates: list[tuple[int, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if p.category != g.category:
                continue
            if _has_offsets(p) and _has_offsets(g):
                ov = _overlap(p, g)
                if ov > 0:
                    candidates.append((ov, i, j))
            elif not _has_offsets(p) and not _has_offsets(g):
                candidates.append((1, i, j))  # category-only match

    # Greedy by largest overlap; ties broken by index so the result is deterministic.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        result.tp.append((preds[i], golds[j]))

    result.fp = [p for i, p in enumerate(preds) if i not in used_p]
    result.fn = [g for j, g in enumerate(golds) if j not in used_g]
    return result


def match_all(pred: list[DetectorOutput], gold: list[DetectorOutput]) -> list[DocMatch]:
    pred_by_id = {p.doc_id: p for p in pred}
    gold_ids = {g.doc_id for g in gold}
    extra = set(pred_by_id) - gold_ids
    if extra:
        log.warning("%d predicted doc_id(s) have no gold and are ignored: %s",
                    len(extra), sorted(extra))
    missing = gold_ids - set(pred_by_id)
    if missing:
        log.warning("%d gold doc_id(s) have no prediction (all spans counted as misses): %s",
                    len(missing), sorted(missing))
    return [match_document(pred_by_id.get(g.doc_id), g) for g in gold]


# --------------------------------------------------------------------------- metrics
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_matches(matches: list[DocMatch]) -> SpanMetrics:
    cat_tp: dict[str, int] = defaultdict(int)
    cat_fp: dict[str, int] = defaultdict(int)
    cat_fn: dict[str, int] = defaultdict(int)
    for m in matches:
        for p, _g in m.tp:
            cat_tp[p.category.value] += 1
        for p in m.fp:
            cat_fp[p.category.value] += 1
        for g in m.fn:
            cat_fn[g.category.value] += 1

    tp, fp, fn = sum(cat_tp.values()), sum(cat_fp.values()), sum(cat_fn.values())
    precision, recall, f1 = _prf(tp, fp, fn)

    per_category: dict[str, dict[str, float]] = {}
    for cat in sorted(set(cat_tp) | set(cat_fp) | set(cat_fn)):
        p, r, f = _prf(cat_tp[cat], cat_fp[cat], cat_fn[cat])
        per_category[cat] = {
            "precision": p, "recall": r, "f1": f,
            "support": float(cat_tp[cat] + cat_fn[cat]),
            "tp": float(cat_tp[cat]), "fp": float(cat_fp[cat]), "fn": float(cat_fn[cat]),
        }
    return SpanMetrics(precision=precision, recall=recall, f1=f1,
                       support=tp + fn, per_category=per_category)


def score(pred: list[DetectorOutput], gold: list[DetectorOutput]) -> SpanMetrics:
    """Interface from spec section 8.2."""
    return score_matches(match_all(pred, gold))


# --------------------------------------------------------------------------- I/O
def _load_output(path: Path) -> DetectorOutput:
    """Accepts a DetectorOutput file, a DetectorResult file, or a gold file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "output" in data and "spans" not in data:  # DetectorResult
        data = data["output"]
    return DetectorOutput.model_validate({"doc_id": data["doc_id"], "spans": data["spans"]})


def _load_gold(path: Path) -> DetectorOutput:
    """Gold files omit `method`, which the schema requires, so fill it in."""
    data = json.loads(path.read_text(encoding="utf-8"))
    spans = [{**s, "method": "rule" if s.get("start") is not None else "llm",
              "rationale": s.get("rationale") or ("gold" if s.get("start") is None else None)}
             for s in data["spans"]]
    return DetectorOutput.model_validate({"doc_id": data["doc_id"], "spans": spans})


def load_dir(directory: Path, gold: bool = False) -> list[DetectorOutput]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no .json files in {directory}")
    return [(_load_gold if gold else _load_output)(f) for f in files]


# --------------------------------------------------------------------------- reporting
def format_summary(metrics: SpanMetrics, n_docs: int) -> str:
    return (f"Recall {metrics.recall:.2f}  Precision {metrics.precision:.2f}  "
            f"F1 {metrics.f1:.2f}  ({n_docs} documents, {metrics.support} gold spans)")


def format_demo_line(metrics: SpanMetrics, n_docs: int) -> str:
    """The exact line shown in the spec's demo (section 6)."""
    return (f"Precision {metrics.precision:.2f}  Recall {metrics.recall:.2f}  "
            f"F1 {metrics.f1:.2f}  ({n_docs} documents, {metrics.support} gold spans)")


def format_table(metrics: SpanMetrics) -> str:
    rows = ["| Category | Support | Recall | Precision | F1 |", "|---|---:|---:|---:|---:|"]
    for cat, v in metrics.per_category.items():
        rows.append(f"| {cat} | {int(v['support'])} | {v['recall']:.2f} | "
                    f"{v['precision']:.2f} | {v['f1']:.2f} |")
    rows.append(f"| **ALL (micro)** | {metrics.support} | **{metrics.recall:.2f}** | "
                f"{metrics.precision:.2f} | {metrics.f1:.2f} |")
    return "\n".join(rows)


def format_errors(matches: list[DocMatch], gold_docs: dict[str, str] | None = None) -> str:
    """Misses (leaks) first, then false positives. For error analysis."""
    lines: list[str] = []
    for m in matches:
        if not m.fn and not m.fp:
            continue
        tag = (gold_docs or {}).get(m.doc_id, m.doc_id)
        lines.append(f"[{tag}]")
        for g in m.fn:
            pos = f"{g.start}-{g.end}" if g.start is not None else "no-offset"
            lines.append(f"   MISSED   {g.category.value:<18} {g.text!r} ({pos})")
        for p in m.fp:
            pos = f"{p.start}-{p.end}" if p.start is not None else "no-offset"
            lines.append(f"   EXTRA    {p.category.value:<18} {p.text!r} ({pos})")
    return "\n".join(lines) if lines else "No errors."


def build_report(metrics: SpanMetrics, matches: list[DocMatch], n_docs: int,
                 model: str = "<model>", title: str = "Agent 1 - Metrics Report") -> str:
    n_fn = sum(len(m.fn) for m in matches)
    n_fp = sum(len(m.fp) for m in matches)
    n_tp = sum(len(m.tp) for m in matches)
    return f"""# {title}

**Model:** {model}  |  **Documents:** {n_docs}  |  **Gold spans:** {metrics.support}

## Headline

{format_demo_line(metrics, n_docs)}

Recall is the primary figure: a missed span is a leak, an extra span only costs some utility.

| | Count |
|---|---:|
| True positives | {n_tp} |
| False negatives (leaks) | {n_fn} |
| False positives | {n_fp} |

## Per category

{format_table(metrics)}

## Match rules used

- Explicit spans: character ranges overlap and category equal (partial overlap counts).
- Quasi-identifiers (no offsets): matched on category alone within a document.
- One-to-one matching: a gold span is credited at most once, a prediction credits at most once.
- Micro-averaged across documents. Offsets are Python indices, end-exclusive.

## Limitations (state these in the write-up)

- Small gold set ({n_docs} documents): per-category figures rest on few spans and move a lot
  with a single error. Treat categories with support under 10 as indicative only.
- Quasi-identifiers are scored on category only, so a QUASI_COMBO prediction that is
  about the wrong combination still counts. Quasi recall is therefore an upper bound.
- The age position (AGE treated as sensitive) is stricter than HIPAA Safe Harbor.

## Errors (misses first)

```
{format_errors(matches)}
```
"""


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="score", description="Score Agent 1 output against gold.")
    ap.add_argument("--gold", required=True, type=Path, help="directory of gold JSON files")
    ap.add_argument("--pred", required=True, type=Path, help="directory of predicted JSON files")
    ap.add_argument("--errors", action="store_true", help="print misses and false positives")
    ap.add_argument("--table", action="store_true", help="print per-category table")
    ap.add_argument("--json", type=Path, help="write SpanMetrics JSON to this path")
    ap.add_argument("--report", type=Path, help="write a markdown metrics report to this path")
    ap.add_argument("--model", default="<model>", help="model name for the report header")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    gold = load_dir(args.gold, gold=True)
    pred = load_dir(args.pred)
    matches = match_all(pred, gold)
    metrics = score_matches(matches)

    print(format_demo_line(metrics, len(gold)))
    if args.table:
        print()
        print(format_table(metrics))
    if args.errors:
        print()
        names = {}
        for f in sorted(args.gold.glob("*.json")):
            names[json.loads(f.read_text(encoding="utf-8"))["doc_id"]] = f.stem
        print(format_errors(matches, names))
    if args.json:
        args.json.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(build_report(metrics, matches, len(gold), args.model),
                               encoding="utf-8")
        print(f"\nreport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
