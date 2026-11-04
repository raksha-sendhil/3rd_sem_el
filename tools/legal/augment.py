"""train_aug: Gemini rewords training documents (legal build spec, "Data", optional).

OWNER: Person D. Gemini sees only fictional training documents. Each document is reworded
with every entity string kept word for word; the spans are then re-found by text search
(``generator.relocate_spans``) and any document with a missing entity is dropped. The output
is a folder of gold-format files and an ``sft.jsonl`` that ``finetune_agent1.py train --extra``
adds to the training set.

    GEMINI_API_KEY=... python tools/legal/augment.py --n 150 --out data/legal/train_aug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from sanitise.config import get_settings
from sanitise.legal.generator import relocate_spans
from sanitise.llm.gemini import GeminiClient
from sanitise.loader import compute_doc_id

SYSTEM = (
    "You reword short legal documents from a fictional law firm so that a model trained on "
    "them sees varied writing. Keep the meaning and the document type. Every string listed "
    "under KEEP must appear in your version exactly as written, character for character. "
    "Return only the reworded document."
)


def user_prompt(text: str, spans: list[dict]) -> str:
    keep = "\n".join(f"- {s['text']}" for s in spans if s["start"] is not None)
    return f"DOCUMENT:\n{text}\n\nKEEP:\n{keep}\n\nREWORDED DOCUMENT:"


async def augment(n: int, src: Path, out: Path, seed: int) -> tuple[int, int]:
    from sanitise.agents.detector.quasi import (
        LEGAL_LLM_CATEGORIES,
        LEGAL_SYSTEM_PROMPT,
        legal_user_prompt,
    )

    files = sorted((src / "gold").glob("*.json"))
    random.Random(seed).shuffle(files)
    client = GeminiClient.from_settings(get_settings())
    kept = dropped = 0
    (out / "gold").mkdir(parents=True, exist_ok=True)
    with (out / "sft.jsonl").open("w", encoding="utf-8") as sft:
        for path in files[:n]:
            gold = json.loads(path.read_text(encoding="utf-8"))
            text = await client.text_call(
                SYSTEM, user_prompt(gold["text"], gold["spans"]), temperature=0.9
            )
            spans = relocate_spans(text, gold["spans"])
            if spans is None:
                dropped += 1
                continue
            doc = {
                "doc_id": compute_doc_id(text),
                "text": text,
                "spans": spans,
                "annotator": "generator+gemini",
                "reviewed_by": None,
            }
            (out / "gold" / f"aug_{path.stem}.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            target = [
                {"text": s["text"], "category": s["category"]}
                for s in spans
                if s["category"] in LEGAL_LLM_CATEGORIES
            ]
            sft.write(
                json.dumps(
                    {
                        "doc_id": doc["doc_id"],
                        "system": LEGAL_SYSTEM_PROMPT,
                        "user": legal_user_prompt(text),
                        "target": json.dumps({"spans": target}, ensure_ascii=False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    return kept, dropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--src", type=Path, default=Path("data/legal/train"))
    p.add_argument("--out", type=Path, default=Path("data/legal/train_aug"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    kept, dropped = asyncio.run(augment(args.n, args.src, args.out, args.seed))
    print(f"{kept} reworded documents kept, {dropped} dropped (an entity went missing)")


if __name__ == "__main__":
    main()
