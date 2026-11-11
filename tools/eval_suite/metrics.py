"""The maths of docs/evaluation_metrics.md, with no dependency beyond the standard library.

Text: ``tokens``, ``longest_common_run`` (Repeat Prompt), ``rouge_l`` (ROUGE Prompt and the
membership score), ``token_f1`` (utility). Ranking: ``auc``, ``tpr_at_fpr``. Statistics:
``bootstrap_ci``, ``mcnemar_p`` (paired yes/no), ``wilcoxon_p`` (paired scores), ``median``.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens; "41,721" and "41721" are the same token."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", (text or "").lower())
    return _TOKEN.findall(text)


def longest_common_run(a: str, b: str) -> int:
    """Length of the longest run of consecutive identical tokens shared by ``a`` and ``b``
    (Zeng et al. 2024: a Repeat Prompt is a run of 10 or more)."""
    x, y = tokens(a), tokens(b)
    if not x or not y:
        return 0
    best = 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        xi = x[i - 1]
        for j in range(1, len(y) + 1):
            if xi == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _lcs(x: list[str], y: list[str]) -> int:
    if len(x) < len(y):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y, 1):
            cur.append(prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.0) -> float:
    """ROUGE-L F-measure (Lin 2004) on word tokens: F = (1+b^2) R P / (R + b^2 P), with R and P
    the longest common subsequence over the reference and candidate lengths."""
    c, r = tokens(candidate), tokens(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return (1 + beta**2) * rec * p / (rec + beta**2 * p)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD token F1 (Rajpurkar et al. 2016): harmonic mean of token precision and recall."""
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- ranking


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(score of a member > score of a non-member), ties counting one half (Mann-Whitney)."""
    if not positives or not negatives:
        return float("nan")
    ranked = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2
        rank_sum += mean_rank * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_points(positives: Sequence[float], negatives: Sequence[float]) -> list[tuple[float, float]]:
    """(false positive rate, true positive rate) at every threshold, from (0, 0) to (1, 1)."""
    points = [(0.0, 0.0)]
    for t in sorted(set(positives) | set(negatives), reverse=True):
        fpr = sum(s >= t for s in negatives) / len(negatives)
        tpr = sum(s >= t for s in positives) / len(positives)
        points.append((fpr, tpr))
    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))
    return points


def tpr_at_fpr(positives: Sequence[float], negatives: Sequence[float], fpr: float = 0.01) -> float:
    """Highest true positive rate at a threshold whose false positive rate is at most ``fpr``
    (Carlini et al. 2022 report membership inference this way)."""
    if not positives or not negatives:
        return float("nan")
    return max(t for f, t in roc_points(positives, negatives) if f <= fpr)


# ---------------------------------------------------------------- statistics


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def bootstrap_ci(
    items: Sequence, stat: Callable[[Sequence], float], n: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """95% percentile bootstrap interval (Efron 1979) of ``stat`` over ``items``."""
    if not items:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(items)
    values = sorted(stat([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return (float("nan"), float("nan"))
    return (values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])


def mcnemar_p(pairs: Sequence[tuple[bool, bool]]) -> float:
    """Exact two-sided McNemar test (McNemar 1947) on paired yes/no outcomes (a, b): a binomial
    test on the discordant pairs."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilcoxon_p(pairs: Sequence[tuple[float, float]]) -> float:
    """Two-sided Wilcoxon signed-rank test (normal approximation, tie-corrected) on paired
    scores; zero differences are dropped."""
    diffs = [x - y for x, y in pairs if x != y]
    n = len(diffs)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i, ties = 0, 0.0
    while i < n:
        j = i
        while j < n and abs(diffs[order[j]]) == abs(diffs[order[i]]):
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + 1 + j) / 2
        t = j - i
        ties += t**3 - t
        i = j
    w_plus = sum(r for r, d in zip(ranks, diffs, strict=True) if d > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - ties / 48)
    if sigma == 0:
        return 1.0
    z = (abs(w_plus - mu) - 0.5) / sigma
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


def rate_summary(flags: Sequence[bool]) -> dict:
    """A rate with its bootstrap interval."""
    lo, hi = bootstrap_ci(list(flags), lambda s: mean([float(f) for f in s]))
    return {"rate": mean([float(f) for f in flags]), "ci95": [lo, hi], "n": len(flags)}
