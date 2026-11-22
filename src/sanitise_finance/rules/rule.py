"""One regex rule of Agent 1's rule pass."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    """``check`` (optional) gets the matched value and says whether it counts (a Luhn checksum)."""

    name: str
    pattern: str
    score: float
    flags: int = re.IGNORECASE
    check: Callable[[str], bool] | None = field(default=None, repr=False, compare=False)
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile(self.pattern, self.flags))
