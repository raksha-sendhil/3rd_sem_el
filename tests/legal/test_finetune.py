"""The fine-tuning script's data handling (no GPU, torch or transformers needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "finetune_agent1", REPO / "tools" / "legal" / "finetune_agent1.py"
)
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)


def _encode(prompt_words: int, target_words: int, max_length: int) -> dict:
    def tokenizer(text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def template(messages, add_generation_prompt=False, tokenize=False):
        return " ".join(m["content"] for m in messages)

    example = {"system": "s " * prompt_words, "user": "u", "target": "t " * target_words}
    return finetune.encode(example, tokenizer, template, max_length)


def test_prompt_tokens_are_masked_and_the_target_is_kept() -> None:
    e = _encode(10, 5, 100)
    assert e["labels"][:11] == [-100] * 11
    assert all(label != -100 for label in e["labels"][11:])
    assert len(e["labels"]) == len(e["input_ids"]) == 16


def test_examples_with_no_target_left_are_dropped_and_cuts_are_counted() -> None:
    whole, cut, gone = _encode(10, 5, 100), _encode(10, 50, 30), _encode(40, 5, 30)
    kept, n_cut, n_dropped = finetune.usable([whole, cut, gone], 30)
    assert kept == [whole, cut] and (n_cut, n_dropped) == (1, 1)


def test_warmup_is_three_percent_of_the_optimiser_steps() -> None:
    assert finetune.warmup_steps(400, 1.0) == 2  # 50 steps
    assert finetune.warmup_steps(8, 1.0) == 1  # never zero
