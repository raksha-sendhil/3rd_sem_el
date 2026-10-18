"""The deterministic pass: Presidio + spaCy + custom regex -> validated rule spans."""

import logging
import re
from functools import lru_cache

import tldextract
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer

from sanitise.rules.categories import ENTITY_TO_CATEGORY, REQUESTED_ENTITIES
from sanitise.rules.normalise import normalise_spans
from sanitise.rules.recognizers import ALL_LAYERS, custom_recognizers
from sanitise.schemas import PIICategory, SensitiveSpan

C = PIICategory

DEFAULT_SPACY_MODEL = "en_core_web_lg"
# Low on purpose: a false negative is a leak. Still high enough to drop Presidio's
# near-zero-confidence guesses (US_BANK_NUMBER 0.05, US_DRIVER_LICENSE 0.01 on any
# digit run) unless context words lift them.
DEFAULT_SCORE_THRESHOLD = 0.35
PHONE_REGIONS = ("US", "GB", "IN", "CA", "AU", "DE", "FR", "IL", "BR")

# EmailRecognizer validates domains with tldextract, which by default tries to
# download the public suffix list. The system must run fully offline, so use the
# snapshot bundled with the package.
tldextract.tldextract.TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


@lru_cache(maxsize=2)
def _nlp_engine(spacy_model: str):
    # Shared across analyzers so switching layers does not reload spaCy.
    return NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    ).create_engine()


def build_analyzer(
    spacy_model: str = DEFAULT_SPACY_MODEL, layers: tuple[str, ...] = ALL_LAYERS
) -> AnalyzerEngine:
    """Presidio core plus the requested custom layers (see recognizers.py)."""
    analyzer = AnalyzerEngine(nlp_engine=_nlp_engine(spacy_model), supported_languages=["en"])
    analyzer.registry.remove_recognizer("PhoneRecognizer")
    analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=PHONE_REGIONS))
    for rec in custom_recognizers(layers):
        analyzer.registry.add_recognizer(rec)
    return analyzer


@lru_cache(maxsize=8)
def get_analyzer(
    spacy_model: str = DEFAULT_SPACY_MODEL, layers: tuple[str, ...] = ALL_LAYERS
) -> AnalyzerEngine:
    """Shared analyzer; loading spaCy takes several seconds so do it once per process."""
    return build_analyzer(spacy_model, tuple(layers))


# ---------------------------------------------------------------- post-filters

# spaCy tags durations and relative expressions as DATE. Those don't identify
# anyone on their own ("for 2 days", "yesterday") and would swamp the output, so
# they are dropped. Anchored dates ("March 3, 2021", "3 years ago") are kept.
_VAGUE_DATE = re.compile(
    r"^(?:(?:for|about|around|approximately|approx\.?|almost|nearly|over|under|past|last|next|the|a|an|"
    r"every|few|couple|several|of|some|more|than|less|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"\d+(?:\.\d+)?|\d+-\d+|half|and|or|-)\s*)*"
    r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|nights?|weeks?|wks?|months?|years?|yrs?|decades?)$"
    r"|^(?:today|tonight|yesterday|tomorrow|now|currently|recently|daily|weekly|monthly|yearly|annually"
    r"|morning|evening|afternoon|night|this\s+(?:morning|evening|afternoon|week|month|year)"
    r"|last\s+night|every\s+(?:day|night|morning|week)|at\s+night|the\s+(?:morning|evening|night)"
    r"|(?:this|last|next|the|in|in\s+the|over\s+the)?\s*(?:summer|winter|spring|autumn|fall|weekend)"
    r"|(?:around|about|at|by)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.?|p\.m\.?)|\d{1,2}:\d{2})$",
    re.IGNORECASE,
)

# spaCy regularly mislabels salutations and platform names in ChatDoctor as PERSON.
_NAME_STOPLIST = {
    "doctor",
    "dr",
    "doc",
    "sir",
    "madam",
    "maam",
    "ma'am",
    "hi",
    "hello",
    "dear",
    "thanks",
    "thank you",
    "regards",
    "chatdoctor",
    "chat doctor",
    "healthcaremagic",
    "hcm",
    "patient",
    "mr",
    "mrs",
    "ms",
    "miss",
    # kinship terms are RELATION, not NAME
    "mom",
    "mum",
    "mommy",
    "mummy",
    "mama",
    "dad",
    "daddy",
    "papa",
    "grandma",
    "grandpa",
    "granny",
    "nana",
    "uncle",
    "aunt",
    "auntie",
}
# "Peyronie's disease", "Guillain Barre syndrome" - eponyms, not people
_EPONYM_TAIL = re.compile(
    r"(?:'s|’s)?\s+(?:disease|syndrome|palsy|sign|test|reflex|procedure|disorder)\b", re.IGNORECASE
)


def _keep(category: PIICategory, fragment: str) -> bool:
    f = fragment.strip(" .,;:!?\"'").lower()
    if category is C.DATE and _VAGUE_DATE.match(f):
        return False
    return not (category is C.NAME and f in _NAME_STOPLIST)


def _to_span(text: str, r: RecognizerResult) -> SensitiveSpan | None:
    category = ENTITY_TO_CATEGORY.get(r.entity_type)
    if category is None:
        return None
    fragment = text[r.start : r.end]
    if not _keep(category, fragment):
        return None
    if category is C.NAME and _EPONYM_TAIL.match(text, r.end):
        return None
    meta = r.recognition_metadata or {}
    source = meta.get(RecognizerResult.RECOGNIZER_NAME_KEY, "presidio")
    detail = meta.get("rule") or r.entity_type
    return SensitiveSpan(
        text=fragment,
        category=category,
        start=r.start,
        end=r.end,
        method="rule",
        confidence=min(max(float(r.score), 0.0), 1.0),
        rationale=f"{source}:{detail}",
    )


def rule_detect(
    text: str,
    *,
    analyzer: AnalyzerEngine | None = None,
    layers: tuple[str, ...] = ALL_LAYERS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[SensitiveSpan]:
    """Presidio + spaCy + regex. Every span has exact start/end and method='rule'.

    ``layers`` selects which custom recognizer layers run on top of Presidio's
    core (default: all) - see recognizers.py.

    Output is normalised: no overlaps, ``text == doc[start:end]``, sorted by start.
    """
    if not text.strip():
        return []
    analyzer = analyzer or get_analyzer(layers=tuple(layers))
    results = analyzer.analyze(
        text=text, language="en", entities=REQUESTED_ENTITIES, score_threshold=score_threshold
    )
    spans = [s for r in results if (s := _to_span(text, r)) is not None]
    return normalise_spans(spans, text)
