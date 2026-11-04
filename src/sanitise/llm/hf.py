"""Gemma 4 through ``transformers``, optionally with the fine-tuned LoRA adapter.  OWNER: C.

Legal build spec, Part 3, step 5: "load the base model plus this adapter on GPU 1 for step 3.
The rest of the agents use the base model through Ollama." This client is that step 3 model:
``google/gemma-4-E4B-it`` loaded in 4-bit NF4 (as in training), plus ``agent1_lora`` when an
adapter path is given. Without an adapter it is the "before" model of the before/after
comparison, prompted identically.

There is no grammar-constrained decoding here, so ``json_call`` relies on the shared
validate-and-repair loop of ``BaseJSONClient``. Generation runs in a worker thread so the
event loop stays free. torch, transformers, peft and bitsandbytes are imported only when the
model is loaded (Kaggle GPU), never by the test suite.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sanitise.config import Settings
from sanitise.llm.base import BaseJSONClient, Message

log = logging.getLogger(__name__)


def load_gemma(base_model: str, adapter: str | None = None, device: str = "cuda:0") -> tuple:
    """(processor, model) in 4-bit NF4 with double quantisation and float16 compute."""
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 GPUs have no bfloat16
    )
    model_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or (
        transformers.AutoModelForCausalLM
    )
    processor = AutoProcessor.from_pretrained(base_model)
    model = model_cls.from_pretrained(
        base_model, quantization_config=quant, device_map={"": device}, torch_dtype=torch.float16
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


class TransformersClient(BaseJSONClient):
    def __init__(
        self,
        *,
        model: str = "google/gemma-4-E4B-it",
        adapter: str | None = None,
        device: str = "cuda:1",
        max_new_tokens: int = 1024,
        loaded: tuple | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model + (f"+{adapter}" if adapter else ""), **kwargs)
        self.base_model, self.adapter, self.device = model, adapter, device
        self.max_new_tokens = max_new_tokens
        self._loaded = loaded

    @classmethod
    def from_settings(cls, settings: Settings) -> TransformersClient:
        return cls(
            model=settings.agent1_base_model,
            adapter=settings.agent1_adapter,
            device=settings.agent1_device,
            max_new_tokens=settings.llm_num_predict,
            retries=settings.llm_retries,
            temperature=settings.llm_temperature,
            temperature_step=settings.llm_temperature_step,
            temperature_max=settings.llm_temperature_max,
        )

    def _model(self) -> tuple:
        if self._loaded is None:
            log.info("loading %s (adapter %s) on %s", self.base_model, self.adapter, self.device)
            self._loaded = load_gemma(self.base_model, self.adapter, self.device)
        return self._loaded

    def _run(self, messages: list[Message], temperature: float, max_tokens: int | None) -> str:
        import torch

        processor, model = self._model()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        sampling: dict[str, Any] = {"do_sample": False}
        if temperature > 0:  # a repair retry at a raised temperature
            sampling = {"do_sample": True, "temperature": temperature}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens or self.max_new_tokens, **sampling
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return processor.decode(new_tokens, skip_special_tokens=True)

    async def _generate(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float,
        *,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, messages, temperature, max_tokens)
