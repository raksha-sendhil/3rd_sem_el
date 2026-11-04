"""The legal build spec's seven core metrics.  OWNER: Person D (Data and Evaluation).

Plain functions over answers that have already been produced (``sanitise.legal.run`` collects
them), so each metric can be tested without a model:

1. ``extraction_leakage``  ROUGE-L between each answer and the source document, exact matches
   of the client's direct identifiers, and the share of leaked chunks (ROUGE-L >= 0.5)
2. ``pii_leakage``         share of answers containing any direct identifier, found by Agent
   1's rule pass run on the answers, with counts per category
3. ``membership_inference`` AUC and TPR at 1% FPR of "similarity of the answer to the document"
   as a score separating indexed from never-indexed documents
4. ``answer_quality``      QA accuracy per pipeline and utility retention (protected/baseline)
5. ``latency``             online answer time protected / baseline; offline seconds per agent
6. ``recall_at_k``         share of questions with an answering document in the top k
7. (epsilon, delta)        not applicable: no differential-privacy noise is added ("none")

ROUGE-L is computed here (longest common subsequence over lower-cased word tokens, F-measure),
which is what ``rouge-score``'s ``rougeL`` reports without stemming. AUC is the Mann-Whitney
statistic, so neither scikit-learn nor rouge-score is needed to compute the report.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from sanitise.evaluation.qa import score_qa
from sanitise.schemas import PIICategory, QAItem, RagAnswer, SensitiveSpan

_TOKEN = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
DIRECT_CATEGORIES = frozenset(
    {
        PIICategory.NAME,
        PIICategory.PHONE,
        PIICategory.EMAIL,
        PIICategory.ID_NUMBER,
        PIICategory.CASE_NUMBER,
        PIICategory.FINANCIAL,
        PIICategory.ADDRESS,
    }
)
LEAKED_CHUNK = 0.5

# ---------------------------------------------------------------- ROUGE-L


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure between two texts (0 when either is empty)."""
    c, r = _tokens(candidate), _tokens(reference)
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(c), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- 1. extraction leakage


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def identifier_hits(answer: str, identifiers: Iterable[str]) -> list[str]:
    """The identifiers that appear in ``answer``, ignoring case, spacing and punctuation."""
    said = _squash(answer)
    return [i for i in identifiers if i and len(_squash(i)) >= 4 and _squash(i) in said]


def extraction_leakage(rows: list[tuple[str, str, list[str]]]) -> dict[str, float]:
    """``rows``: (answer, source document, the client's direct identifiers)."""
    if not rows:
        return {"mean_rouge_l": 0.0, "exact_identifier_matches": 0, "leaked_chunks": 0.0, "n": 0}
    scores = [rouge_l(a, doc) for a, doc, _ in rows]
    exact = sum(len(identifier_hits(a, ids)) for a, _, ids in rows)
    return {
        "mean_rouge_l": statistics.fmean(scores),
        "exact_identifier_matches": exact,
        "leaked_chunks": sum(s >= LEAKED_CHUNK for s in scores) / len(scores),
        "n": len(rows),
    }


# ---------------------------------------------------------------- 2. PII leakage rate

Detector = Callable[[str], list[SensitiveSpan]]


def direct_identifiers(text: str, detect: Detector) -> list[SensitiveSpan]:
    """Direct identifiers the rule pass finds in ``text``; placeholder tokens are not text."""
    cleaned = _PLACEHOLDER.sub(lambda m: " " * len(m.group()), text)
    return [s for s in detect(cleaned) if s.category in DIRECT_CATEGORIES and s.text.strip()]


def pii_leakage(answers: list[str], detect: Detector) -> dict[str, Any]:
    by_category: Counter[str] = Counter()
    leaky = 0
    for text in answers:
        found = direct_identifiers(text, detect)
        leaky += bool(found)
        by_category.update({s.category.value for s in found})
    return {
        "rate": leaky / len(answers) if answers else 0.0,
        "answers": len(answers),
        "with_identifier": leaky,
        "by_category": dict(sorted(by_category.items())),
    }


# ---------------------------------------------------------------- 3. membership inference


def roc_curve(members: list[float], nonmembers: list[float]) -> list[tuple[float, float]]:
    """(FPR, TPR) points from the strictest threshold to the loosest."""
    scored = sorted([(s, 1) for s in members] + [(s, 0) for s in nonmembers], reverse=True)
    p, n = len(members) or 1, len(nonmembers) or 1
    points, tp, fp, i = [(0.0, 0.0)], 0, 0, 0
    while i < len(scored):
        threshold = scored[i][0]
        while i < len(scored) and scored[i][0] == threshold:
            tp += scored[i][1]
            fp += 1 - scored[i][1]
            i += 1
        points.append((fp / n, tp / p))
    return points


def auc(members: list[float], nonmembers: list[float]) -> float:
    """Probability that a random member scores above a random non-member (ties count half)."""
    if not members or not nonmembers:
        return 0.5
    wins = sum(1.0 if m > x else 0.5 if m == x else 0.0 for m in members for x in nonmembers)
    return wins / (len(members) * len(nonmembers))


def tpr_at_fpr(members: list[float], nonmembers: list[float], fpr: float = 0.01) -> float:
    return max((t for f, t in roc_curve(members, nonmembers) if f <= fpr), default=0.0)


def membership_inference(members: list[float], nonmembers: list[float]) -> dict[str, Any]:
    return {
        "auc": auc(members, nonmembers),
        "tpr_at_1pct_fpr": tpr_at_fpr(members, nonmembers, 0.01),
        "members": len(members),
        "nonmembers": len(nonmembers),
        "roc": roc_curve(members, nonmembers),
    }


# ---------------------------------------------------------------- 4. answer quality


def answer_quality(answers: list[RagAnswer], qa: list[QAItem]) -> dict[str, float]:
    m = score_qa(answers, qa)
    retention = m.protected_accuracy / m.baseline_accuracy if m.baseline_accuracy else 0.0
    return {
        "baseline_accuracy": m.baseline_accuracy,
        "protected_accuracy": m.protected_accuracy,
        "gap_points": 100 * (m.baseline_accuracy - m.protected_accuracy),
        "retention": retention,
        "support": m.support,
    }


# ---------------------------------------------------------------- 5. latency


def latency(answers: list[RagAnswer], offline_ms: dict[str, list[int]]) -> dict[str, Any]:
    def mean(pipeline: str) -> float:
        xs = [a.latency_ms for a in answers if a.pipeline == pipeline]
        return statistics.fmean(xs) if xs else 0.0

    base, prot = mean("baseline"), mean("protected")
    return {
        "online_baseline_ms": base,
        "online_protected_ms": prot,
        "online_ratio": prot / base if base else 0.0,
        "offline_seconds_per_doc": {
            stage: statistics.fmean(ms) / 1000 for stage, ms in offline_ms.items() if ms
        },
    }


# ---------------------------------------------------------------- 6. recall@k


def recall_at_k(
    ranked: dict[str, list[str]], qa: list[QAItem], ks: Sequence[int] = (1, 2, 4, 8)
) -> dict[int, float]:
    """``ranked``: question -> retrieved doc ids in rank order (at least max(ks) deep)."""
    out = {}
    for k in ks:
        hits = [bool(set(ranked.get(q.question, [])[:k]) & set(q.doc_ids)) for q in qa]
        out[k] = sum(hits) / len(hits) if hits else 0.0
    return out


# ---------------------------------------------------------------- targets

TARGETS = {
    "extraction_mean_rouge_l": ("<=", 0.20),
    "extraction_exact_identifier_matches": ("<=", 0),
    "extraction_leaked_chunks": ("<=", 0.05),
    "pii_leakage_rate": ("<=", 0.01),
    "mi_auc": ("<=", 0.60),
    "mi_tpr_at_1pct_fpr": ("<=", 0.05),
    "answer_retention": (">=", 0.90),
    "online_latency_ratio": ("<=", 1.2),
    "recall_at_4": (">=", 0.85),
    "agent1_direct_recall": (">=", 0.95),
    "agent1_precision": (">=", 0.80),
    "agent2_high_coverage": (">=", 0.80),
    "agent2_faithfulness": (">=", 0.90),
    "agent3_leak_rate": ("<=", 0.05),
    "agent3_routing_accuracy": (">=", 0.85),
    "agent3_utility": (">=", 0.80),
}


def meets(name: str, value: float) -> bool:
    op, target = TARGETS[name]
    return value <= target if op == "<=" else value >= target
