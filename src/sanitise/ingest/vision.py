"""Transcribe a page image with a vision model: on Ollama (default ``qwen3-vl:8b-instruct``,
the page never leaves the machine) or, with ``ocr_backend="gemini"``, on the Gemini API.

The prompt is domain-neutral (hospital, legal and finance papers alike): copy, don't correct;
labelled fields on their own lines; tables as Markdown; ``[illegible]`` instead of a guess;
``[signature]`` and ``[stamp: ...]`` for marks. The model's output is then tidied: code fences
and reasoning removed, and runaway repetition (a known failure of vision models) cut.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

import httpx

from sanitise.config import Settings
from sanitise.errors import LLMTimeout, LLMUnreachable

PROMPT = """Transcribe all the text in this document image exactly as it is written, in reading order (top to bottom; left column before right column).

Rules:
- Copy every word, name, number, date, code, dose and abbreviation exactly as written. Do not correct spelling, expand abbreviations, translate, summarise or add anything.
- Put each "Label: value" pair on its own line, for example "Name: ..." or "Date: ...".
- Write tables as Markdown tables, one table row per line, keeping every cell.
- Transcribe handwritten text too. If a word cannot be read, write [illegible] instead of guessing.
- For a signature write [signature]. For a stamp or seal write [stamp: its readable text]. Skip logos, photos and barcodes.
- Output only the transcription, with no comments before or after it."""

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def tidy(text: str) -> str:
    """Remove fences/reasoning, collapse runs of the same line (more than 2) to one."""
    text = _THINK.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    out: list[str] = []
    for line in text.split("\n"):
        if line.strip() and len(out) >= 2 and out[-1] == line and out[-2] == line:
            continue
        out.append(line)
    return "\n".join(out).strip()


class VisionReader:
    """Async client for one vision model on Ollama."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 600.0,
        num_ctx: int = 16384,
        num_predict: int = 4096,
        seed: int | None = 42,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None) -> VisionReader:
        return cls(
            model=model or settings.ocr_model,
            base_url=settings.ollama_base_url,
            timeout_s=settings.ocr_timeout_s,
            num_ctx=settings.ocr_num_ctx,
            num_predict=settings.ocr_num_predict,
            seed=settings.llm_seed,
        )

    async def read(self, image: bytes, prompt: str = PROMPT) -> str:
        options: dict[str, Any] = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": 1.05,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
            "stream": False,
            "keep_alive": "1h",
            "options": options,
        }
        try:
            response = await self._http.post("/api/chat", json=payload, timeout=self.timeout_s)
        except httpx.TimeoutException:
            raise LLMTimeout(
                f"{self.model} did not read the page within {self.timeout_s:.0f}s", stage="ingest"
            ) from None
        except httpx.TransportError as exc:
            raise LLMUnreachable(
                f"cannot reach Ollama at {self.base_url}: {exc}", stage="ingest"
            ) from None
        if response.status_code == 404:
            raise LLMUnreachable(
                f"vision model {self.model!r} not found - run `ollama pull {self.model}`",
                stage="ingest",
            )
        if response.status_code >= 400:
            raise LLMUnreachable(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}",
                stage="ingest",
            )
        try:
            return tidy(response.json()["message"]["content"])
        except (KeyError, TypeError, ValueError):
            raise LLMUnreachable(
                f"unexpected response from Ollama: {response.text[:200]}", stage="ingest"
            ) from None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> VisionReader:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class MockVisionReader:
    """Mock mode and tests: no model; returns a fixed text (or the one given)."""

    model = "mock"

    def __init__(self, text: str = "[scanned page: no vision model in mock mode]") -> None:
        self.text = text
        self.calls = 0

    async def read(self, image: bytes, prompt: str = PROMPT) -> str:
        self.calls += 1
        return self.text

    async def aclose(self) -> None:
        pass


class GeminiVisionReader:
    """The same transcription on the Gemini API instead of a local model: for machines without a
    GPU. The page image leaves the machine (Google's servers), which the local model avoids.
    Several comma-separated keys are tried in turn when one is out of quota."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_s: float = 120.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.keys = [k.strip() for k in api_key.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("GeminiVisionReader needs an API key (SANITISE_GEMINI_API_KEY)")
        self.timeout_s = timeout_s
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_s)

    async def read(self, image: bytes, prompt: str = PROMPT) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image).decode("ascii"),
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        last = ""
        for key in self.keys:
            try:
                response = await self._http.post(
                    url, params={"key": key}, json=payload, timeout=self.timeout_s
                )
            except httpx.TimeoutException:
                raise LLMTimeout(
                    f"Gemini {self.model} did not read the page in time", stage="ingest"
                ) from None
            except httpx.TransportError as exc:
                raise LLMUnreachable(
                    f"cannot reach the Gemini API: {exc}", stage="ingest"
                ) from None
            if response.status_code in (429, 403):  # this key is out of quota: try the next one
                last = response.text[:200]
                continue
            if response.status_code >= 400:
                raise LLMUnreachable(
                    f"Gemini returned HTTP {response.status_code}: {response.text[:300]}",
                    stage="ingest",
                )
            parts = (
                (response.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            )
            return tidy("".join(p.get("text", "") for p in parts))
        raise LLMUnreachable(f"every Gemini key is out of quota: {last}", stage="ingest")

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def make_reader(settings: Settings, model: str | None = None):
    """The reader the settings ask for: mock (mock mode), Gemini (``ocr_backend="gemini"``) or
    the local model on Ollama (default). ``model`` overrides the backend's model."""
    if settings.llm_backend == "mock":
        return MockVisionReader()
    if settings.ocr_backend == "gemini":
        key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        return GeminiVisionReader(model=model or settings.ocr_gemini_model, api_key=key)
    return VisionReader.from_settings(settings, model=model)
