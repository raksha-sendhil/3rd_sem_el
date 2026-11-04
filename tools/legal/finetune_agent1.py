"""Fine-tune Gemma 4 E4B for Agent 1's legal LLM pass with QLoRA (legal build spec, Part 3).

OWNER: Person C. Runs on one Kaggle T4. Three sub-commands, in the spec's order:

    python tools/legal/finetune_agent1.py eval  --out before.json           # 1. base model
    python tools/legal/finetune_agent1.py train --adapter /kaggle/working/agent1_lora
    python tools/legal/finetune_agent1.py eval  --adapter /kaggle/working/agent1_lora \\
        --out after.json                                                    # 3. tuned model

Training data is ``data/legal/train/sft.jsonl`` (written by the generator): the legal
detection prompt plus one document, and the exact target ``{"spans": [...]}``. The loss is on
the JSON answer only (prompt tokens are masked with -100). Settings are the spec's table:

    4-bit NF4 + double quantisation, float16 compute (T4: no bfloat16; --fp32 if the loss is nan)
    LoRA r=16, alpha=32, dropout=0.05 on attention and MLP projections of the language model
    1 epoch, lr 2e-4, batch 1 x grad accumulation 8, max length 1024, cosine, 3% warm-up,
    gradient checkpointing, paged_adamw_8bit, loss logged every 10 steps

``eval`` runs the model (``sanitise.llm.hf.TransformersClient``, same prompt as the pipeline)
on the 40 held-out documents and scores its spans against the held-out gold with the project's
span scorer (overlap + same category), restricted to the LLM-pass categories.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

DATA = Path("data/legal/train")
BASE = "google/gemma-4-E4B-it"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- data


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def encode(example: dict, tokenizer, apply_template, max_length: int) -> dict:
    """input_ids and labels with the prompt masked out (-100), truncated to max_length."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    prompt = apply_template(messages, add_generation_prompt=True, tokenize=False)
    full = apply_template(
        [*messages, {"role": "assistant", "content": example["target"]}], tokenize=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels[: len(full_ids)]}


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids = [b["input_ids"] + [pad_id] * (width - len(b["input_ids"])) for b in batch]
    labels = [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch]
    mask = [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


# ---------------------------------------------------------------- train


def train(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

    dtype = torch.float32 if args.fp32 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(args.base)
    tokenizer = getattr(processor, "tokenizer", processor)
    model_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or (
        transformers.AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        args.base, quantization_config=quant, device_map={"": 0}, torch_dtype=dtype
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # language model only: leave the vision and audio towers frozen
        target_modules=r".*language_model.*\.(" + "|".join(TARGET_MODULES) + r")$",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    examples = load_examples(args.data / "sft.jsonl")
    for extra in args.extra or []:  # e.g. the Gemini-reworded train_aug set
        examples += load_examples(Path(extra))
    encoded = [
        encode(e, tokenizer, processor.apply_chat_template, args.max_length) for e in examples
    ]
    print(f"{len(encoded)} training examples")

    trainer = Trainer(
        model=model,
        train_dataset=encoded,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id or 0),
        args=TrainingArguments(
            output_dir=str(Path(args.adapter).parent / "agent1_checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            fp16=not args.fp32,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(args.adapter)
    processor.save_pretrained(args.adapter)
    print(f"adapter saved to {args.adapter}")


# ---------------------------------------------------------------- evaluate


async def _detect_all(client, docs: list[dict]) -> list:
    from sanitise.agents.detector.quasi import infer_quasi
    from sanitise.domain import use_domain
    from sanitise.schemas import DetectorOutput

    out = []
    with use_domain("legal"):
        for i, d in enumerate(docs, 1):
            try:
                spans = await infer_quasi(d["text"], [], client)
            except Exception as exc:  # broken JSON after retries counts as nothing found
                print(f"  {d['doc_id']}: {exc}")
                spans = []
            out.append(DetectorOutput(doc_id=d["doc_id"], spans=spans))
            print(f"  [{i}/{len(docs)}] {len(spans)} spans", flush=True)
    return out


def evaluate(args: argparse.Namespace) -> None:
    from sanitise.agents.detector.quasi import LEGAL_LLM_CATEGORIES
    from sanitise.evaluation.scoring import load_dir, score
    from sanitise.llm.hf import TransformersClient

    docs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((args.data / "heldout").glob("*.json"))
    ]
    gold = load_dir(args.data / "heldout_gold", gold=True)
    for g in gold:
        g.spans = [s for s in g.spans if s.category.value in LEGAL_LLM_CATEGORIES]
    client = TransformersClient(model=args.base, adapter=args.adapter, device="cuda:0", retries=1)
    pred = asyncio.run(_detect_all(client, docs))
    m = score(pred, gold)
    result = {
        "model": client.model,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "documents": len(docs),
        "per_category": m.per_category,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{client.model}: P {m.precision:.3f} R {m.recall:.3f} F1 {m.f1:.3f} -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("train", "eval"))
    p.add_argument("--base", default=BASE)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--adapter", default=None, help="train: where to save; eval: adapter to load")
    p.add_argument("--extra", nargs="*", help="more sft.jsonl files (train_aug)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--fp32", action="store_true", help="if the float16 loss shows nan")
    p.add_argument("--resume", default=None, help="checkpoint folder to resume from")
    p.add_argument("--out", default="agent1_eval.json")
    args = p.parse_args()
    if args.command == "train":
        if not args.adapter:
            p.error("train needs --adapter (e.g. /kaggle/working/agent1_lora)")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
