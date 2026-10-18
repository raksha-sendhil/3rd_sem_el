"""
validate_gold.py - catch silent annotation errors before they corrupt the metrics.

Checks, per gold file:
  - valid JSON with doc_id, text, spans
  - doc_id == first 16 hex chars of SHA-256 of text (matches what detect() will produce)
  - text length <= 2000 (the Phase 2 limit)
  - every category is a valid PIICategory
  - explicit spans: 0 <= start < end <= len(text) and text[start:end] == span text
  - QUASI_COMBO spans have start = end = null; other spans never do
  - no two explicit spans overlap (annotation guide rule: no nested/overlapping spans)
  - no duplicate spans
  - reviewed_by is set (warning only until review is finished)
And across the directory:
  - doc_ids are unique; a matching fixture exists with identical text (if --fixtures given)

Usage:  python -m evaluation.validate_gold --gold data/gold [--fixtures data/fixtures]
Exit code 1 if any ERROR is found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from schemas import PIICategory

VALID = {c.value for c in PIICategory}
MAX_LEN = 2000


def validate_file(path: Path) -> tuple[list[str], list[str], dict]:
    """Returns (errors, warnings, parsed_data). parsed_data is {} if unreadable."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"], warnings, {}

    for key in ("doc_id", "text", "spans"):
        if key not in data:
            errors.append(f"missing key '{key}'")
    if errors:
        return errors, warnings, data

    text: str = data["text"]
    expected_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if data["doc_id"] != expected_id:
        errors.append(f"doc_id {data['doc_id']} != sha256(text)[:16] = {expected_id}")
    if len(text) > MAX_LEN:
        errors.append(f"text is {len(text)} chars, limit is {MAX_LEN}")
    if not data.get("reviewed_by"):
        warnings.append("not reviewed yet (reviewed_by is empty)")

    explicit: list[tuple[int, int, str]] = []
    seen: set[tuple] = set()
    for i, s in enumerate(data["spans"]):
        tag = f"spans[{i}]"
        cat, stext = s.get("category"), s.get("text")
        start, end = s.get("start"), s.get("end")
        if cat not in VALID:
            errors.append(f"{tag}: invalid category {cat!r}")
            continue
        key = (cat, stext, start, end)
        if key in seen:
            errors.append(f"{tag}: duplicate span {cat} {stext!r}")
        seen.add(key)

        if cat == "QUASI_COMBO":
            if start is not None or end is not None:
                errors.append(f"{tag}: QUASI_COMBO must have start/end = null")
            continue
        if start is None or end is None:
            errors.append(f"{tag}: {cat} needs start and end (only QUASI_COMBO may be null)")
            continue
        if not (0 <= start < end <= len(text)):
            errors.append(f"{tag}: bad range {start}-{end} for text of length {len(text)}")
            continue
        if text[start:end] != stext:
            errors.append(f"{tag}: text[{start}:{end}] = {text[start:end]!r} but span says {stext!r}")
        if stext != stext.strip():
            warnings.append(f"{tag}: span text has leading/trailing whitespace {stext!r}")
        explicit.append((start, end, f"{cat} {stext!r}"))

    explicit.sort()
    for (s1, e1, l1), (s2, e2, l2) in zip(explicit, explicit[1:]):
        if s2 < e1:
            errors.append(f"overlapping spans: {l1} ({s1}-{e1}) and {l2} ({s2}-{e2})")
    return errors, warnings, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="validate_gold")
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--fixtures", type=Path, help="also check gold text matches fixtures")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args(argv)

    files = sorted(args.gold.glob("*.json"))
    if not files:
        print(f"no .json files in {args.gold}")
        return 1

    total_err = total_warn = total_spans = 0
    ids: dict[str, str] = {}
    for f in files:
        errs, warns, data = validate_file(f)
        total_spans += len(data.get("spans", []))

        did = data.get("doc_id")
        if did:
            if did in ids:
                errs.append(f"duplicate doc_id, also used by {ids[did]}")
            ids[did] = f.name

        if args.fixtures and "text" in data:
            fx = args.fixtures / f.name
            if not fx.exists():
                errs.append(f"no matching fixture {fx}")
            else:
                fx_data = json.loads(fx.read_text(encoding="utf-8"))
                if fx_data.get("text") != data["text"]:
                    errs.append("gold text differs from fixture text")
                if fx_data.get("doc_id") != data["doc_id"]:
                    errs.append("gold doc_id differs from fixture doc_id")

        status = "FAIL" if errs else ("warn" if warns else "ok")
        print(f"{status:<5} {f.name}")
        for e in errs:
            print(f"        ERROR   {e}")
        for w in warns:
            print(f"        warning {w}")
        total_err += len(errs)
        total_warn += len(warns)

    print(f"\n{len(files)} files, {total_spans} spans, {total_err} errors, {total_warn} warnings")
    if total_err or (args.strict and total_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
