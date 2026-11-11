"""Per-request domain selection.  OWNER: Person C.

The dashboard sends the chosen domain with every request, as ``?domain=legal`` (the attack
stream is an EventSource, which cannot set headers) or as an ``X-Sanitise-Domain`` header.
This pure ASGI middleware runs the whole request inside ``use_domain(...)``, so the prompts,
the data and the models the endpoint picks all belong to that domain. Without either, the
request uses ``SANITISE_DOMAIN``. An unknown domain is a 400 with the usual error envelope.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

from sanitise_legal.config import Settings
from sanitise_legal.domain import profiles, use_domain

HEADER = b"x-sanitise-domain"


def requested_domain(scope: dict[str, Any]) -> str | None:
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if query.get("domain"):
        return query["domain"][0]
    for name, value in scope.get("headers", []):
        if name.lower() == HEADER:
            return value.decode("latin-1")
    return None


class DomainMiddleware:
    def __init__(self, app: Any, settings: Settings) -> None:
        self.app, self.settings = app, settings

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or (domain := requested_domain(scope)) is None:
            await self.app(scope, receive, send)
            return
        known = profiles(self.settings)
        if domain not in known:
            body = json.dumps(
                {
                    "error": {
                        "code": "UNKNOWN_DOMAIN",
                        "stage": "request",
                        "detail": f"unknown domain {domain!r}; choose from {sorted(known)}",
                    }
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        with use_domain(domain):
            await self.app(scope, receive, send)
