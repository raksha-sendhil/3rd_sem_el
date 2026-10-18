"""MockLLM - a scriptable stand-in for Ollama.

It goes through exactly the same parse / validate / retry / repair loop as the
real client (both subclass ``BaseJSONClient``), so a malformed scripted reply
exercises the repair path just as a malformed reply from llama3 would.

Where each reply comes from, in order:

1. the per-schema queue (``by_schema={"QuasiList": [...]}``), keyed by the
   Pydantic model's class name;
2. the shared queue (``responses=[...]``);
3. ``handler(call)`` if given;
4. otherwise a minimal valid instance fabricated from the JSON schema
   (``QuasiList`` -> ``{"quasi_identifiers": []}``).

A scripted item may be a ``str`` (sent verbatim - use this for broken JSON), a
``dict``/``list`` (JSON-encoded), a ``BaseModel`` (dumped), an ``Exception``
(raised, e.g. ``LLMTimeout``) or a callable taking ``MockCall`` and returning
any of those.

Examples::

    llm = MockLLM(by_schema={"QuasiList": [
        '{"quasi_identifiers": [{"text": "x", "category": "DIAGNOSIS"}]}',  # bad enum
        QuasiList(quasi_identifiers=[QuasiHit(text="x", category="QUASI_COMBO",
                                               rationale="rare job + rare condition")]),
    ]})
    out = await llm.json_call(system, user, QuasiList)   # 1 retry, then validates
    assert llm.calls[1].temperature > llm.calls[0].temperature
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel

from sanitise.llm.base import BaseJSONClient, Message

Scripted = Union[str, dict, list, BaseModel, BaseException, Callable[["MockCall"], Any]]  # noqa: UP007


@dataclass(frozen=True)
class MockCall:
    """One generation attempt (a json_call with retries produces several)."""

    schema_name: str
    messages: list[Message]
    json_schema: dict[str, Any]
    temperature: float

    @property
    def system(self) -> str:
        return self.messages[0]["content"]

    @property
    def user(self) -> str:
        return self.messages[1]["content"]

    @property
    def is_repair(self) -> bool:
        return len(self.messages) > 2


class MockLLM(BaseJSONClient):
    def __init__(
        self,
        responses: Iterable[Scripted] = (),
        *,
        by_schema: Mapping[str, Iterable[Scripted]] | None = None,
        handler: Callable[[MockCall], Any] | None = None,
        model: str = "mock-llm",
        delay_s: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._queue: deque[Scripted] = deque(responses)
        self._by_schema: defaultdict[str, deque[Scripted]] = defaultdict(deque)
        for name, items in (by_schema or {}).items():
            self._by_schema[name].extend(items)
        self.handler = handler
        self.delay_s = delay_s
        self.calls: list[MockCall] = []

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> MockLLM:
        """Load ``{"<SchemaName>": [reply, ...], ...}`` from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected an object mapping schema names to reply lists")
        return cls(
            by_schema={k: (v if isinstance(v, list) else [v]) for k, v in data.items()}, **kwargs
        )

    def push(self, *responses: Scripted, schema: str | None = None) -> None:
        (self._by_schema[schema] if schema else self._queue).extend(responses)

    @property
    def pending(self) -> int:
        return len(self._queue) + sum(len(q) for q in self._by_schema.values())

    async def _generate(
        self, messages: list[Message], json_schema: dict[str, Any], temperature: float
    ) -> str:
        call = MockCall(
            schema_name=json_schema.get("title", ""),
            messages=[dict(m) for m in messages],
            json_schema=json_schema,
            temperature=temperature,
        )
        self.calls.append(call)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        per_schema = self._by_schema.get(call.schema_name)
        if per_schema:
            item: Any = per_schema.popleft()
        elif self._queue:
            item = self._queue.popleft()
        elif self.handler is not None:
            item = self.handler
        else:
            item = fabricate(json_schema)

        if callable(item) and not isinstance(item, (BaseModel, type)):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        if isinstance(item, str):
            return item
        return json.dumps(item)


def fabricate(schema: dict[str, Any]) -> Any:
    """Smallest JSON value that satisfies a (ref-inlined) JSON schema."""
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return fabricate(options[0])
    if "allOf" in schema:
        return fabricate(schema["allOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or "properties" in schema:
        props = schema.get("properties", {})
        return {
            name: fabricate(props[name]) for name in schema.get("required", []) if name in props
        }
    if kind == "array":
        return [fabricate(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "mock" + "x" * max(0, schema.get("minLength", 0) - 4)
    if kind == "integer":
        return int(schema.get("minimum", schema.get("exclusiveMinimum", -1) + 1))
    if kind == "number":
        return float(schema.get("minimum", 0.0))
    if kind == "boolean":
        return False
    return None
