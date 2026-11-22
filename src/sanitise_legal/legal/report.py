"""The metrics report and charts from ``results.json``.  OWNER: Person D.

``write_report(results, out_dir)`` writes ``REPORT.md`` (every metric of the legal build
spec next to its target, both pipelines side by side, the baseline reported but not given a
target) and, when matplotlib is installed, the charts the spec lists:

==========================  ===========================================
extraction.png              grouped bars, baseline vs protected
pii_leakage.png             stacked bars by identifier category
membership_roc.png          ROC curves for both pipelines, with the diagonal
answer_quality.png          grouped bars
latency.png                 offline seconds per document, stacked by agent
recall_at_k.png             line chart over k = 1, 2, 4, 8
privacy_utility.png         leakage (x) vs answer quality (y), one dot per system
==========================  ===========================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sanitise_legal.evaluation.legal import TARGETS, meets

PIPELINES = ("baseline", "protected")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _row(name: str, label: str, values: dict[str, Any]) -> str:
    op, target = TARGETS[name]
    protected = values["protected"]
    ok = "yes" if meets(name, float(protected)) else "**no**"
    return f"| {label} | {_fmt(values['baseline'])} | {_fmt(protected)} | {op} {target} | {ok} |"


def core_values(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    p = results["pipelines"]
    q = results["answer_quality"]
    return {
        "extraction_mean_rouge_l": {k: p[k]["extraction"]["mean_rouge_l"] for k in PIPELINES},
        "extraction_exact_identifier_matches": {
            k: p[k]["extraction"]["exact_identifier_matches"] for k in PIPELINES
        },
        "extraction_leaked_chunks": {k: p[k]["extraction"]["leaked_chunks"] for k in PIPELINES},
        "pii_leakage_rate": {k: p[k]["pii_leakage"]["rate"] for k in PIPELINES},
        "mi_auc": {k: p[k]["membership"]["auc"] for k in PIPELINES},
        "mi_tpr_at_1pct_fpr": {k: p[k]["membership"]["tpr_at_1pct_fpr"] for k in PIPELINES},
        "answer_retention": {"baseline": 1.0, "protected": q["retention"]},
        "online_latency_ratio": {"baseline": 1.0, "protected": results["latency"]["online_ratio"]},
    }


def render(results: dict[str, Any]) -> str:
    lines = [
        "# Legal build spec: metrics report",
        "",
        f"Answer model: `{results.get('answer_model')}`. Documents indexed: "
        f"{results.get('documents_indexed')}. Run time {results.get('seconds', 0):.0f} s.",
        "",
        "## Core metrics",
        "",
        "| Metric | Baseline | Protected | Target (protected) | Met |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "extraction_mean_rouge_l": "1. Extraction: mean ROUGE-L",
        "extraction_exact_identifier_matches": "1. Extraction: exact identifier matches",
        "extraction_leaked_chunks": "1. Extraction: leaked chunks (ROUGE-L >= 0.5)",
        "pii_leakage_rate": "2. PII leakage rate",
        "mi_auc": "3. Membership inference AUC",
        "mi_tpr_at_1pct_fpr": "3. Membership inference TPR at 1% FPR",
        "answer_retention": "4. Utility retention (protected / baseline)",
        "online_latency_ratio": "5. Online latency (protected / baseline)",
    }
    for name, values in core_values(results).items():
        lines.append(_row(name, labels[name], values))
    q = results["answer_quality"]
    r4 = results["recall_at_k"].get("4", results["recall_at_k"].get(4, 0.0))
    lines += [
        f"| 4. QA accuracy | {q['baseline_accuracy']:.3f} | {q['protected_accuracy']:.3f} | "
        f"within 10 points | {'yes' if q['gap_points'] <= 10 else '**no**'} |",
        f"| 6. Recall@4 (same retrieval for both) | {r4:.3f} | {r4:.3f} | >= 0.85 | "
        f"{'yes' if r4 >= 0.85 else '**no**'} |",
        "| 7. (epsilon, delta) | none | none | not applicable | - |",
        "",
        "Recall@k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in results["recall_at_k"].items()),
        "",
        "## Agent-level metrics",
        "",
        "| Agent | Metric | Value | Target | Met |",
        "|---|---|---|---|---|",
    ]
    a = results["agents"]
    agent_rows = [
        (
            "Agent 1",
            "Recall on direct identifiers",
            a["agent1"]["direct_recall"],
            "agent1_direct_recall",
        ),
        ("Agent 1", "Precision, all categories", a["agent1"]["precision"], "agent1_precision"),
        (
            "Agent 2",
            "Coverage of high-importance facts",
            a["agent2"]["high_coverage"],
            "agent2_high_coverage",
        ),
        ("Agent 2", "Faithfulness", a["agent2"]["faithfulness"], "agent2_faithfulness"),
        ("Agent 3", "Leak rate", a["agent3"]["leak_rate"], "agent3_leak_rate"),
        ("Agent 3", "Routing accuracy", a["agent3"]["routing_accuracy"], "agent3_routing_accuracy"),
        ("Agent 3", "Utility", a["agent3"]["utility"], "agent3_utility"),
    ]
    for agent, label, value, key in agent_rows:
        op, target = TARGETS[key]
        lines.append(
            f"| {agent} | {label} | {value:.3f} | {op} {target} | "
            f"{'yes' if meets(key, value) else '**no**'} |"
        )
    lines += [
        f"| Agent 3 | Documents with a direct identifier left after the mask | "
        f"{a['agent3']['documents_with_direct_leaks']} | 0 | "
        f"{'yes' if a['agent3']['documents_with_direct_leaks'] == 0 else '**no**'} |",
        "",
        "Documents where a model step fell back after broken output: "
        + (
            ", ".join(f"{k} {v}" for k, v in sorted(a["agent3"].get("fallbacks", {}).items()))
            or "none"
        )
        + ".",
        "",
        "## Attack outcomes",
        "",
        "| Attack | Monitor | Baseline identified at | Protected identified at | "
        "Smallest protected pool | Blocked rounds |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["attack_outcomes"]:
        lines.append(
            f"| {row['attack_id']} | {row['monitor']} | {row['baseline_identified_at'] or 'never'}"
            f" | {row['protected_identified_at'] or 'never'} | {row['protected_min_pool']} | "
            f"{', '.join(map(str, row['blocked_rounds'])) or '-'} |"
        )
    lines += [
        "",
        "Targets: direct and linkage attacks never identify the target on the protected "
        "pipeline; the stepwise attack with the monitor never leaves fewer than 5 candidates.",
        "",
    ]
    return "\n".join(lines)


def charts(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """The spec's charts as PNG files; [] when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    colours = {"baseline": "#c0504d", "protected": "#4f81bd"}
    p = results["pipelines"]
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["mean ROUGE-L", "leaked chunks"]
    for i, pipe in enumerate(PIPELINES):
        e = p[pipe]["extraction"]
        ax.bar(
            [x + i * 0.38 for x in range(2)],
            [e["mean_rouge_l"], e["leaked_chunks"]],
            0.38,
            label=pipe,
            color=colours[pipe],
        )
    ax.set_xticks([x + 0.19 for x in range(2)], names)
    ax.set_title("Extraction leakage")
    ax.legend()
    save(fig, "extraction.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    cats = sorted({c for pipe in PIPELINES for c in p[pipe]["pii_leakage"]["by_category"]})
    for i, pipe in enumerate(PIPELINES):
        bottom = 0
        for c in cats:
            n = p[pipe]["pii_leakage"]["by_category"].get(c, 0)
            ax.bar(i, n, bottom=bottom, label=c if i == 0 else None)
            bottom += n
    ax.set_xticks(range(2), PIPELINES)
    ax.set_ylabel("answers with this identifier")
    ax.set_title("PII leakage by category")
    if cats:
        ax.legend(fontsize=7)
    save(fig, "pii_leakage.png")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for pipe in PIPELINES:
        roc = p[pipe]["membership"]["roc"]
        ax.plot(
            [x for x, _ in roc],
            [y for _, y in roc],
            color=colours[pipe],
            label=f"{pipe} (AUC {p[pipe]['membership']['auc']:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Membership inference")
    ax.legend()
    save(fig, "membership_roc.png")

    q = results["answer_quality"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(
        PIPELINES,
        [q["baseline_accuracy"], q["protected_accuracy"]],
        color=[colours[x] for x in PIPELINES],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Answer quality (QA accuracy)")
    save(fig, "answer_quality.png")

    offline = results["latency"]["offline_seconds_per_doc"]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    left = 0.0
    for stage, secs in offline.items():
        ax.barh(["offline"], [secs], left=left, label=stage)
        left += secs
    ax.set_xlabel("seconds per document")
    ax.set_title(f"Latency (online ratio {results['latency']['online_ratio']:.2f}x)")
    ax.legend(fontsize=7)
    save(fig, "latency.png")

    rk = results["recall_at_k"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([int(k) for k in rk], list(rk.values()), marker="o")
    ax.axhline(0.85, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("k")
    ax.set_ylim(0, 1.02)
    ax.set_title("Recall@k")
    save(fig, "recall_at_k.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for pipe in PIPELINES:
        acc = q[f"{pipe}_accuracy"]
        ax.scatter(p[pipe]["pii_leakage"]["rate"], acc, s=80, color=colours[pipe], label=pipe)
    ax.set_xlabel("PII leakage rate")
    ax.set_ylabel("QA accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Privacy-utility")
    ax.legend()
    save(fig, "privacy_utility.png")
    return written


def write_report(results: dict[str, Any] | Path, out_dir: Path) -> Path:
    if isinstance(results, Path):
        results = json.loads(results.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "REPORT.md"
    path.write_text(render(results), encoding="utf-8")
    charts(results, out_dir / "charts")
    return path
