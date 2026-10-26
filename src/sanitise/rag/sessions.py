"""Session traces for the dashboard.  OWNER: Person C.

In memory: a session lives as long as the API process. Every ``/v1/compare`` question and
every attack round is appended, which is what the dashboard's session trace table shows.
"""

from __future__ import annotations

import uuid

from sanitise.errors import SessionNotFound
from sanitise.schemas import AttackRound, SessionTrace


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[AttackRound]] = {}

    def create(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = []
        return session_id

    def ensure(self, session_id: str) -> None:
        self._sessions.setdefault(session_id, [])

    def append(self, session_id: str, round_: AttackRound) -> None:
        self.ensure(session_id)
        self._sessions[session_id].append(round_)

    def get(self, session_id: str) -> SessionTrace:
        if session_id not in self._sessions:
            raise SessionNotFound(f"no session {session_id!r}", stage="session")
        return SessionTrace(session_id=session_id, rounds=list(self._sessions[session_id]))

    def next_round(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, [])) + 1
