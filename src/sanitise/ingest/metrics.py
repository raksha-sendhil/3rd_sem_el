"""How well a file was read: compare the extracted text with the known (gold) text.

Headline metric: **CER, the character error rate** = (characters inserted + deleted +
substituted to turn the extracted text into the gold text) / characters in the gold text.
0 is perfect; 0.02 means 2 wrong characters in every 100. Both texts are normalised first so
only reading errors count, not formatting: Markdown marks (table pipes, bold, headings) and the
model's ``[signature]`` / ``[stamp: ...]`` / ``[illegible]`` tags are removed, case and spacing
are ignored. An ``[illegible]`` word therefore counts as missing, as it should.

Also reported, because they are free once the gold is known:

* WER, the same on words;
* word F1, order-free: the words read correctly whatever order they came in. Real test sources
  whose gold text was rebuilt from word boxes (receipts, forms) have only an approximate reading
  order, which inflates CER without any word being misread; word F1 is the fair number there;
* detail recall: the share of the document's private details (names, phones, ID numbers,
  addresses) that appear exactly in the extracted text. A detail read wrongly cannot be found by
  Agent 1, and a near-miss ("Ravl Kumar") still points at the person;
* route accuracy: pages sent the right way (text layer vs vision model);
* header/footer removal: recall (repeated copies removed) and lines lost (removed lines whose
  content is nowhere else in the text; should be 0).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_TAGS = re.compile(r"\[(?:signature|illegible|stamp[^\]]*|seal[^\]]*)\]", re.I)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}[-|: ]*$", re.M)
_MARKS = re.compile(r"[|*#_`>]")
_DASHES = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_DASHES).replace("­", "")
    # a word broken across lines ("consti-\ntutional") is one word, however it was read
    t = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", t)
    t = _TAGS.sub(" ", t)
    t = _TABLE_RULE.sub(" ", t)
    t = _MARKS.sub(" ", t)
    t = re.sub(r"\s+([.,;:!?)])", r"\1", t)  # "word ." and "word." are the same reading
    t = re.sub(r"([(])\s+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass
class TextScore:
    cer: float
    wer: float
    char_errors: int
    gold_chars: int
    word_errors: int
    gold_words: int
    # order-free: words read correctly, whatever order they came in (multiset overlap)
    words_matched: int = 0
    pred_words: int = 0

    @property
    def word_f1(self) -> float:
        total = self.pred_words + self.gold_words
        return 2 * self.words_matched / total if total else 1.0


def _bag(words: list[str]) -> Counter[str]:
    return Counter(w.strip(".,;:!?()[]\"'") for w in words if w.strip(".,;:!?()[]\"'"))


def score_text(pred: str, gold: str) -> TextScore:
    p, g = normalise(pred), normalise(gold)
    ce = Levenshtein.distance(p, g)
    pw, gw = p.split(), g.split()
    we = Levenshtein.distance(pw, gw)
    pb, gb = _bag(pw), _bag(gw)
    return TextScore(
        cer=ce / max(len(g), 1),
        wer=we / max(len(gw), 1),
        char_errors=ce,
        gold_chars=len(g),
        word_errors=we,
        gold_words=len(gw),
        words_matched=sum((pb & gb).values()),
        pred_words=sum(pb.values()),
    )


def detail_hits(pred: str, details: list[str]) -> list[bool]:
    """For each private detail: is it in the extracted text exactly (after normalising)?"""
    p = normalise(pred)
    p_tight = p.replace(" ", "")
    out = []
    for d in details:
        n = normalise(d)
        # "+91 98450 12345" vs "+91 9845012345": spacing inside a number is not a misreading
        out.append(n in p or (any(c.isdigit() for c in n) and n.replace(" ", "") in p_tight))
    return out


def strip_scores(removed: list[dict], expected: list[dict], kept_text: str) -> tuple[int, int, int]:
    """(expected lines removed, removals that lost information, expected lines).

    A removal is safe when it is an expected header/footer copy, a page number, or a line whose
    content is still in the kept text (a duplicate); otherwise information was lost. Readers
    merge or split lines ("Page 2 of 2 This is a computer generated document."), so a removed
    line matches an expected line on the same page when one contains the other or they are
    nearly identical."""
    from difflib import SequenceMatcher

    from sanitise.ingest.headers import _key

    kept = normalise(kept_text)
    exp = [(e["page"], normalise(e["line"])) for e in expected]
    covered: set[int] = set()
    lost = 0
    for r in removed:
        line = normalise(r["line"])
        hits = [
            i
            for i, (page, e) in enumerate(exp)
            if page == r["page"]
            and e
            and (e in line or line in e or SequenceMatcher(None, e, line).ratio() >= 0.85)
        ]
        covered.update(hits)
        if not hits and line not in kept and _key(r["line"]) != "<page-number>":
            lost += 1
    return len(covered), lost, len(expected)
