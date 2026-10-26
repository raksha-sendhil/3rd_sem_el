"""Attack runner.  OWNER: Person C (Platform and Integration).

Plays one attack script (data/attacks/*.json, written by Person D) against both pipelines,
one question per round, in a single session, and scores every round with Person D's
``score_round`` against the person table.
"""

from __future__ import annotations

import json
from pathlib import Path

from sanitise.errors import AttackNotFound
from sanitise.evaluation.attacks import score_round
from sanitise.llm.base import LLMClient
from sanitise.rag.answer import RagContext, answer
from sanitise.schemas import AttackDefinition, AttackRound, AttackRun, Person, RagAnswer


def load_attacks(directory: str | Path) -> dict[str, AttackDefinition]:
    attacks: dict[str, AttackDefinition] = {}
    for path in sorted(Path(directory).glob("*.json")):
        attack = AttackDefinition.model_validate_json(path.read_text(encoding="utf-8"))
        attacks[attack.attack_id] = attack
    return attacks


def load_people(path: str | Path) -> list[Person]:
    path = Path(path)
    if not path.is_file():
        return []
    return [Person.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))]


def get_attack(attacks: dict[str, AttackDefinition], attack_id: str) -> AttackDefinition:
    try:
        return attacks[attack_id]
    except KeyError:
        raise AttackNotFound(f"no attack {attack_id!r}", stage="attack") from None


async def run_attack(
    ctx: RagContext,
    attack: AttackDefinition,
    people: list[Person],
    llm: LLMClient,
    session_id: str | None = None,
) -> AttackRun:
    session_id = session_id or ctx.sessions.create()
    ctx.sessions.ensure(session_id)
    seen: dict[str, list[RagAnswer]] = {"baseline": [], "protected": []}
    rounds: list[AttackRound] = []
    identified_at: dict[str, int | None] = {"baseline": None, "protected": None}
    for n, query in enumerate(attack.queries, start=1):
        base = await answer(ctx, query, "baseline", session_id, llm)
        prot = await answer(ctx, query, "protected", session_id, llm)
        seen["baseline"].append(base)
        seen["protected"].append(prot)
        base_leak = score_round(people, attack.target_person_id, seen["baseline"])
        prot_leak = score_round(people, attack.target_person_id, seen["protected"])
        for name, leak in (("baseline", base_leak), ("protected", prot_leak)):
            if leak.identified and identified_at[name] is None:
                identified_at[name] = n
        round_ = AttackRound(
            round=n,
            query=query,
            baseline=base,
            protected=prot,
            baseline_leakage=base_leak,
            protected_leakage=prot_leak,
        )
        ctx.sessions.append(session_id, round_)
        rounds.append(round_)
    return AttackRun(
        attack_id=attack.attack_id,
        session_id=session_id,
        rounds=rounds,
        baseline_identified_at=identified_at["baseline"],
        protected_identified_at=identified_at["protected"],
    )
