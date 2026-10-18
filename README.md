# sanitise: Agent 1, Person A (Deterministic Logic)

This is everything in Agent 1 that runs without a model: Presidio, spaCy, the custom regex recognizers, span normalisation and the merge function. It covers stages 2, 3 and 6 of spec §3.1.

## Interfaces (spec §8.2)

```python
from sanitise.rules import rule_detect, merge_spans

rule_spans = rule_detect(text)                    # list[SensitiveSpan], method="rule"
final      = merge_spans(rule_spans, llm_spans)   # what Person C puts into DetectorOutput.spans
```

`rule_detect` guarantees the following for every span it returns:
- `text == doc[start:end]` holds exactly.
- The span has no surrounding whitespace or punctuation.
- Spans never overlap, and they come back sorted by `start`.
- `rationale` names the recognizer and rule that fired it, e.g. `AgeRecognizer:n_units_old` or `SpacyRecognizer:PERSON`. The document viewer can show it as the "agent decision".
- If the offsets are inconsistent it raises `OffsetOutOfRangeError`, which C maps to the `OFFSET_OUT_OF_RANGE` error code.

`merge_spans` produces this output:
- The rule spans come first, sorted by offset.
- The LLM spans follow, in the order the model returned them, deduplicated on category plus normalised text.
- An LLM span is dropped when every part of it (split on `+`, `,`, `and`, `with`) already appears in a rule span's text, because it only restates explicit identifiers.
- A combination that adds new information is kept, e.g. `32 years old + lighthouse keeper`.

## Pipeline: a general core with domain layers on top

The recognizers are built in layers: a general core first, then domain-specific additions. Each layer can be switched on or off:

```python
rule_detect(text)                                         # all layers (default)
rule_detect(text, layers=("core", "spec", "general"))     # e.g. no healthcare/email extras
```

| Layer | What it adds | Why |
|---|---|---|
| **core** | Presidio + spaCy `en_core_web_lg` built-ins. PERSON→NAME, PHONE_NUMBER→PHONE (regions US/GB/IN/CA/AU/DE/FR/IL/BR), EMAIL_ADDRESS→EMAIL, LOCATION, DATE_TIME→DATE, SSN/passport/credit card/IBAN/IP…→ID_NUMBER, NRP/URL→OTHER | These are the explicit identifiers in spec §3.1, and they're general to any domain. |
| **spec** | AGE (`32 years old`, `aged 54`, `my father is 67`), PHYSICAL (`6 ft 3 inches`, `approximately 220 pounds`, `BMI of 31`), RELATION (`my son`, `my mother-in-law`, `Hi Mom`) | These are spec categories Presidio has no recognizer for, and the §6 demo requires them. The patterns match phrasing; they don't encode medical knowledge. |
| **general** | ADDRESS (street addresses, `City, ST 12345`, P.O. boxes, UK postcodes), labelled ID_NUMBER (`MRN:`, `policy no.`, `CNIC`), OCCUPATION (`works as a …`, `is a retired teacher`) | Presidio misses these explicit identifiers in any domain. |
| **healthcare** | Named facilities → LOCATION (`Apollo Hospital`, `Cleveland Clinic`) | Specific to ChatDoctor. spaCy tags these ORGANIZATION. |
| **email** | Enron phone formats (`713/853-1234`, `x3-5761`, `ext. 5512`), signature title lines (`President, Enron Wholesale Services`) | Specific to Enron. Switch this layer off if the fixtures have no emails. |

Each layer adds measurable recall. Results are on the 10 held-out documents (`python eval/score_rules.py data/gold_heldout --ablation`):

| Layers | Precision | Recall | F1 |
|---|---|---|---|
| core (Presidio alone) | 0.79 | 0.48 | 0.60 |
| + spec | 0.91 | 0.74 | 0.82 |
| + general | 0.94 | 0.82 | 0.87 |
| + healthcare | 0.94 | 0.86 | 0.90 |
| + email | 0.95 | 0.88 | 0.91 |

Every layer then goes through the same post-processing:

- **Post-filters**:
  - Drop durations and relative times that spaCy calls DATE (`for 2 days`, `yesterday`, `this summer`, `6pm`). Anchored dates are kept.
  - Drop salutations and kinship words that spaCy mislabels as PERSON (`doctor`, `sir`, `Mom`).
  - Drop eponymous diseases (`Peyronie's disease`).
- **Normalisation** (`rules/normalise.py`): trim → union same-category overlaps → resolve cross-category overlaps by precedence → merge adjacent same-category spans.

### Category precedence (fixed Day 1)

When spans overlap, the earlier category in this list wins:

`EMAIL > ID_NUMBER > PHONE > ADDRESS > AGE > PHYSICAL > DATE > NAME > OCCUPATION > LOCATION > RELATION > MEDICAL_CONDITION > QUASI_COMBO > OTHER`

When a weaker span only partly overlaps a stronger one, the part nobody claimed is kept, as long as it still carries content. Recall comes first, so coverage is never silently lost.

## Positions the team should confirm (feed into D's annotation guide)

- **Age is treated as sensitive at any value.** This is stricter than HIPAA Safe Harbor, which only covers ages of 90 and over (spec §9).
- **Durations are not DATE.** `approximately 10 weeks` is left for the LLM pass, which matches the spec demo, where it appears as a QUASI_COMBO.
- **ORGANIZATION is not requested from Presidio.** In Enron it is almost always "Enron" itself. If an employer should count, it goes to the LLM pass.
- **MEDICAL_CONDITION is not produced by rules.** That is left to the LLM pass.
- **Score threshold is 0.35.** It drops Presidio's near-zero guesses (US_BANK_NUMBER 0.05 on any digit run) and keeps phone numbers (0.4).

## Running

```bash
uv sync                                   # or: pip install -e . pytest
python -m spacy download en_core_web_lg
pytest -q                                 # 79 tests, ~20 s (spaCy load dominates)
python -m sanitise.rules "My son is 32 years old and lives at 42 Elm Street."
```

`/v1/detect/rules` (Person C) is just `DetectorOutput(doc_id=..., spans=rule_detect(text))`.

The rule pass takes about 0.7 s on a 2000-character document once spaCy is loaded. Loading takes about 8 s once per process, and `get_analyzer()` caches it.

The EmailRecognizer's domain check is forced to the bundled public-suffix snapshot, so nothing touches the network.

`src/sanitise/schemas.py` is a verbatim copy of spec §8.1, used to build against. Replace it with Person C's published file at integration.
