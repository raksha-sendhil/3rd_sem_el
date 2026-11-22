"""Person A - deterministic logic. Everything in Agent 1 that runs without a model."""

from sanitise_legal.rules.categories import CATEGORY_PRECEDENCE, ENTITY_TO_CATEGORY, precedence
from sanitise_legal.rules.engine import build_analyzer, get_analyzer, rule_detect
from sanitise_legal.rules.merge import merge_spans
from sanitise_legal.rules.normalise import OffsetOutOfRangeError, normalise_spans
from sanitise_legal.rules.recognizers import ALL_LAYERS

__all__ = [
    "ALL_LAYERS",
    "CATEGORY_PRECEDENCE",
    "ENTITY_TO_CATEGORY",
    "OffsetOutOfRangeError",
    "build_analyzer",
    "get_analyzer",
    "merge_spans",
    "normalise_spans",
    "precedence",
    "rule_detect",
]
