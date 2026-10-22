"""Score the rule pass against a gold directory (spec section 3.2 match rule).

A predicted span is correct if it overlaps a gold span of the same category; a
gold span is recalled if some same-category prediction overlaps it. "Leak
coverage" ignores the category: was the gold text flagged at all?

    python eval/score_rules.py [gold_dir] [--show] [--layers core,spec,...]
    python eval/score_rules.py [gold_dir] --ablation   # adds one layer at a time
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sanitise.rules import ALL_LAYERS, rule_detect

SKIP = {"QUASI_COMBO", "MEDICAL_CONDITION"}  # LLM-pass scope


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def prf(tp_p, n_p, tp_g, n_g):
    p = tp_p / n_p if n_p else 0.0
    r = tp_g / n_g if n_g else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(docs, gold_dir, layers=ALL_LAYERS, show=False, quiet=False):
    rule_detect("warm up", layers=layers)

    stats = defaultdict(lambda: [0, 0, 0, 0])  # tp_pred, n_pred, tp_gold, n_gold
    leak_hit = leak_n = 0
    misses, fps = [], []
    t0 = time.perf_counter()
    for d in docs:
        gold = [g for g in d["spans"] if g["category"] not in SKIP and g["start"] is not None]
        pred = [s.model_dump(mode="json") for s in rule_detect(d["text"], layers=layers)]
        for p in pred:
            ok = any(overlaps(p, g) and g["category"] == p["category"] for g in gold)
            stats[p["category"]][0] += ok
            stats[p["category"]][1] += 1
            if not ok:
                near = next((g for g in gold if overlaps(p, g)), None)
                fps.append((d["name"], p["category"], p["text"], near and near["category"]))
        for g in gold:
            ok = any(overlaps(p, g) and p["category"] == g["category"] for p in pred)
            any_ok = any(overlaps(p, g) for p in pred)
            stats[g["category"]][2] += ok
            stats[g["category"]][3] += 1
            leak_hit += any_ok
            leak_n += 1
            if not ok:
                got = next((p["category"] for p in pred if overlaps(p, g)), None)
                misses.append((d["name"], g["category"], g["text"], got))
    elapsed = time.perf_counter() - t0

    tot = [sum(v[i] for v in stats.values()) for i in range(4)]
    p, r, f = prf(*tot)
    if quiet:
        return {"precision": p, "recall": r, "f1": f, "leak_coverage": leak_hit / leak_n}
    print(f"\nlayers: {', '.join(layers)}")
    print(
        f"\nRULE PASS vs {gold_dir.name}: {len(docs)} documents, {tot[3]} gold spans, {tot[1]} predicted, "
        f"{elapsed / len(docs):.2f} s/doc\n"
    )
    print(f"  OVERALL   precision {p:.2f}   recall {r:.2f}   F1 {f:.2f}")
    print(
        f"  Leak coverage (flagged under any category): {leak_hit}/{leak_n} = {leak_hit / leak_n:.2f}\n"
    )
    print(f"  {'category':12} {'prec':>5} {'rec':>5} {'F1':>5}  {'gold':>4} {'pred':>4}")
    for cat in sorted(stats, key=lambda c: -stats[c][3]):
        cp, cr, cf = prf(*stats[cat])
        print(f"  {cat:12} {cp:5.2f} {cr:5.2f} {cf:5.2f}  {stats[cat][3]:4} {stats[cat][1]:4}")
    if show:
        print(f"\n  MISSED gold spans ({len(misses)}):")
        for m in misses:
            print(f"    {m[0]}  {m[1]:11} {m[2]!r}" + (f"   (flagged as {m[3]})" if m[3] else ""))
        print(f"\n  FALSE POSITIVES ({len(fps)}):")
        for x in fps:
            print(f"    {x[0]}  {x[1]:11} {x[2]!r}" + (f"   (gold says {x[3]})" if x[3] else ""))
    return {"precision": p, "recall": r, "f1": f, "leak_coverage": leak_hit / leak_n}


def main():
    args = sys.argv[1:]
    pos = [a for a in args if not a.startswith("--")]
    layers = ALL_LAYERS
    if "--layers" in args:
        layers = tuple(args[args.index("--layers") + 1].split(","))
        pos = [a for a in pos if a != args[args.index("--layers") + 1]]
    gold_dir = (
        Path(pos[0]) if pos else Path(__file__).resolve().parent.parent / "data" / "gold_draft"
    )
    docs = [json.loads(p.read_text()) for p in sorted(gold_dir.glob("*.json"))]
    if "--ablation" not in args:
        evaluate(docs, gold_dir, layers, show="--show" in args)
        return
    print(f"\nLAYER ABLATION vs {gold_dir.name} ({len(docs)} documents)\n")
    print(f"  {'layers':40} {'prec':>5} {'rec':>5} {'F1':>5} {'leak cov':>8}")
    for i in range(1, len(ALL_LAYERS) + 1):
        m = evaluate(docs, gold_dir, ALL_LAYERS[:i], quiet=True)
        label = " + ".join(ALL_LAYERS[:i])
        print(
            f"  {label:40} {m['precision']:5.2f} {m['recall']:5.2f} {m['f1']:5.2f} {m['leak_coverage']:8.2f}"
        )


if __name__ == "__main__":
    main()
