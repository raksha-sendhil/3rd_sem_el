"""The LLM seam (spec section 8.2): ``LLMClient`` and the shared retry/repair loop.

Everything that needs a model depends on the ``LLMClient`` protocol only, never
on Ollama directly. That is what lets Person B develop against ``MockLLM`` and
lets tests run without a model (dependency inversion).

``BaseJSONClient`` implements ``json_call`` once for every backend:

    generate -> parse -> validate against the Pydantic schema
       on failure: retry with a higher temperature and a repair instruction
       that quotes the validation error, up to ``retries`` extra attempts
       after the last failure: raise ``SchemaValidationFailed``

A backend only implements ``_generate`` (messages in, raw text out). The text of
the repair instruction and the lenient parser are pluggable hooks so Person B
can tune them for small models without touching transport code.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from sanitise.errors import SchemaValidationFailed

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}
OutputParser = Callable[[str], Any]
"""Raw model text -> a JSON value. Raises ``ValueError`` when no JSON can be recovered."""
RepairBuilder = Callable[[str, str, dict[str, Any]], str]
"""(validation_error, previous_raw_output, json_schema) -> the next user message."""


# ------------------------------------------------------------------ protocol


@runtime_checkable
class LLMClient(Protocol):
    """Spec 8.2. Implementations may widen ``retries`` to ``int | None``
    (None = the client's configured default)."""

    async def json_call(self, system: str, user: str, schema: type[M], retries: int = 3) -> M: ...


# ------------------------------------------------------------------ call accounting
#
# ``json_call`` returns only the validated model, as the contract requires. The
# orchestrator still needs retry counts for ``DetectorMeta``, so clients report
# them into a context-local collector. Each asyncio task has its own context, so
# concurrent ``detect()`` calls never see each other's numbers.


@dataclass
class CallStats:
    calls: int = 0
    attempts: int = 0
    retries: int = 0
    failures: int = 0
    models: list[str] = field(default_factory=list)


_current_stats: ContextVar[CallStats | None] = ContextVar("sanitise_llm_call_stats", default=None)


@contextmanager
def track_calls() -> Iterator[CallStats]:
    """Collect stats for every ``json_call`` made inside the ``with`` block."""
    stats = CallStats()
    token = _current_stats.set(stats)
    try:
        yield stats
    finally:
        _current_stats.reset(token)


def _record(attempts: int, failed: bool, model: str) -> None:
    stats = _current_stats.get()
    if stats is None:
        return
    stats.calls += 1
    stats.attempts += attempts
    stats.retries += attempts - 1
    stats.failures += int(failed)
    stats.models.append(model)


# ------------------------------------------------------------------ default hooks

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_lenient(raw: str) -> Any:
    """Parse model output as JSON, tolerating code fences and chatter around one object."""
    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"output is not valid JSON: {first_error.msg} at char {first_error.pos}"
        ) from None


def default_repair_instruction(error: str, raw_output: str, json_schema: dict[str, Any]) -> str:
    return (
        "Your previous reply could not be accepted because it failed validation:\n"
        f"{error}\n\n"
        "Reply again with ONLY a single JSON object that matches this JSON schema exactly. "
        "Use only the allowed enum values. No prose, no markdown.\n"
        f"{json.dumps(json_schema, separators=(',', ':'))}"
    )


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable error text, e.g.
    ``quasi_identifiers[1].category: Input should be 'NAME', 'AGE', ... (got 'DIAGNOSIS')``."""
    parts = []
    for err in exc.errors(include_url=False):
        path = ""
        for item in err["loc"]:
            path += f"[{item}]" if isinstance(item, int) else (f".{item}" if path else str(item))
        msg = err["msg"]
        if err["type"] not in {"missing", "model_type"} and "input" in err:
            shown = repr(err["input"])
            msg += f" (got {shown[:80] + '...' if len(shown) > 80 else shown})"
        parts.append(f"{path or '<root>'}: {msg}")
    return "; ".join(parts)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every ``{"$ref": "#/$defs/X"}`` with the definition itself.

    Pydantic emits nested models as ``$defs`` + ``$ref``. A flat schema is the
    safest thing to hand to grammar-constrained decoding and is easier for a
    small model to read inside a repair instruction.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name in seen:  # recursive model: leave the reference in place
                    return node
                merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged, seen | {name})
            return {k: resolve(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema, frozenset())


# ------------------------------------------------------------------ base client


class _InvalidOutput(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class BaseJSONClient(ABC):
    """Shared ``json_call`` implementation. Subclasses implement ``_generate``."""

    def __init__(
        self,
        *,
        model: str,
        retries: int = 3,
        temperature: float = 0.0,
        temperature_step: float = 0.2,
        temperature_max: float = 1.0,
        parser: OutputParser = parse_json_lenient,
        repair_builder: RepairBuilder = default_repair_instruction,
    ) -> None:
        self.model = model
        self.default_retries = retries
        self.temperature = temperature
        self.temperature_step = temperature_step
        self.temperature_max = temperature_max
        self.parser = parser
        self.repair_builder = repair_builder

    def with_options(self, *, model: str | None = None, retries: int | None = None) -> Self:
        """A shallow copy with a different model name or default retry budget."""
        clone = copy.copy(self)
        if model is not None:
            clone.model = model
        if retries is not None:
            clone.default_retries = retries
        return clone

    @abstractmethod
    async def _generate(
        self, messages: list[Message], json_schema: dict[str, Any], temperature: float
    ) -> str:
        """Return the model's raw text for ``messages``. Raise ``LLMTimeout``/``LLMUnreachable``."""

    def temperature_for(self, attempt: int) -> float:
        """Attempt 0 uses the base temperature; each retry escalates by one step."""
        return min(self.temperature + attempt * self.temperature_step, self.temperature_max)

    def _validate(self, raw: str, schema: type[M]) -> M:
        try:
            data = self.parser(raw)
        except ValueError as exc:
            raise _InvalidOutput(str(exc)) from None
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise _InvalidOutput(format_validation_error(exc)) from None

    async def json_call(
        self, system: str, user: str, schema: type[M], retries: int | None = None
    ) -> M:
        retries = self.default_retries if retries is None else retries
        if retries < 0:
            raise ValueError("retries must be >= 0")
        json_schema = inline_refs(schema.model_json_schema())
        base: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        messages = base
        last_raw: str | None = None
        last_error = ""

        for attempt in range(retries + 1):
            temperature = self.temperature_for(attempt)
            try:
                raw = await self._generate(messages, json_schema, temperature)
            except BaseException:
                _record(attempt + 1, failed=True, model=self.model)
                raise
            try:
                result = self._validate(raw, schema)
            except _InvalidOutput as exc:
                last_raw, last_error = raw, exc.detail
                log.warning(
                    "%s output failed %s validation (attempt %d/%d, T=%.2f): %s",
                    self.model,
                    schema.__name__,
                    attempt + 1,
                    retries + 1,
                    temperature,
                    exc.detail,
                )
                # Only the latest failure is shown to the model: short context suits 8B models.
                messages = [
                    *base,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": self.repair_builder(exc.detail, raw, json_schema)},
                ]
                continue
            _record(attempt + 1, failed=False, model=self.model)
            if attempt:
                log.info("%s output validated after %d retries", self.model, attempt)
            return result

        _record(retries + 1, failed=True, model=self.model)
        raise SchemaValidationFailed(
            f"{schema.__name__}: {last_error}", attempts=retries + 1, raw_output=last_raw
        )
