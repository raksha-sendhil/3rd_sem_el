# sanitise: Agent 1, Person A (Deterministic Logic)

This is everything in Agent 1 that runs without a model: Presidio, spaCy, the custom regex recognizers, span normalisation and the merge function. It covers stages 2, 3 and 6 of spec §3.1.

## Interfaces (spec §8.2)

```python
from sanitise.rules import rule_detect, merge_spans

rule_spans = rule_detect(text)  # list[SensitiveSpan], method="rule"
final = merge_spans(rule_spans, llm_spans)  # what Person C puts into DetectorOutput.spans
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
rule_detect(text)  # all layers (default)
rule_detect(text, layers=("core", "spec", "general"))  # e.g. no healthcare/email extras
```

| Layer | What it adds | Why |
|---|---|---|
| **core** | Presidio + spaCy `en_core_web_lg` built-ins. PERSON→NAME, PHONE_NUMBER→PHONE (regions US/GB/IN/CA/AU/DE/FR/IL/BR), EMAIL_ADDRESS→EMAIL, LOCATION, DATE_TIME→DATE, SSN/passport/credit card/IBAN/IP…→ID_NUMBER (a UK NHS number only with "NHS" written just before it, since many phone numbers pass its checksum), NRP/URL→OTHER | These are the explicit identifiers in spec §3.1, and they're general to any domain. |
| **spec** | AGE (`32 years old`, `aged 54`, `my father is 67`, in words: `he is eleven,`, `aged twelve`), PHYSICAL (`6 ft 3 inches`, `approximately 220 pounds`, `BMI of 31`), RELATION (`my son`, `my mother-in-law`, `Hi Mom`) | These are spec categories Presidio has no recognizer for, and the §6 demo requires them. The patterns match phrasing; they don't encode medical knowledge. |
| **general** | NAME phrasings spaCy misses for uncommon names (`my name is Nila`, `Hello, this is Kavya`, `Dr. Prakruti`, `call me Ishani`), ADDRESS (street addresses, `City, ST 12345`, P.O. boxes, UK postcodes), labelled ID_NUMBER (`MRN:`, `policy no.`, `CNIC`), OCCUPATION (`works as a …`, `is a retired teacher`) | Presidio misses these explicit identifiers in any domain. |
| **healthcare** | Named facilities → LOCATION (`Apollo Hospital`, `Cleveland Clinic`) | Specific to ChatDoctor. spaCy tags these ORGANIZATION. |
| **email** | Enron phone formats (`713/853-1234`, `x3-5761`, `ext. 5512`), signature title lines (`President, Enron Wholesale Services`) | Specific to Enron. Switch this layer off if the fixtures have no emails. |
| **legal** | ID_NUMBER (PAN `ABCPM1234K`, Aadhaar `2345 6789 0123`, GSTIN, passport, EPIC), FINANCIAL (IFSC, labelled bank accounts, card numbers, UPI ids), CASE_NUMBER (`CS(COMM) 412/2024`, `O.S. No. 412/2024`, `W.P.(C) 1234 of 2023`, `FIR No. 233/2025`, `CC/118/2023`, CNR), COURT (`Bombay High Court`, `High Court of Delhi`, `NCLT Mumbai`), MONEY (`Rs 4,25,00,000`, `₹1.2 crore`, `25 lakh rupees`), Indian phones (`+91 98450 12345`) and PIN-code addresses (`Flat 4B, Koregaon Park, Pune 411001`), ORGANIZATION (company names with a legal-form suffix, `M/s ...`), MATTER_CODENAME (`Project Falcon`), party names (`Client: X`, `Respondent - Smt. X`, both sides of `X v. Y`) | The legal build spec's category table. Indian formats, as the spec's corpus uses. |

After normalisation every NAME span gets its party role where the words just before it give one (`rules/roles.py`): `client` (client, petitioner, complainant, the addressee after "Dear"), `party` (respondent, defendant, accused, opposing party, the right side of `v.`) or `witness`. It is recorded in the rationale as `role=client`, applied to every mention of the same name and to a lone first name or surname of it, and decides the placeholder prefix (`[CLIENT_1]`, `[PARTY_1]`, `[WITNESS_1]`, otherwise `[PERSON_1]`). A name after `Adv.` or `Justice` gets no role: it is a lawyer or judge. spaCy often reads a whole cause title as one PERSON, so a NAME containing ` v. ` is split in two first.

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

`EMAIL > CASE_NUMBER > FINANCIAL > ID_NUMBER > PHONE > ADDRESS > MONEY > AGE > PHYSICAL > DATE > COURT > MATTER_CODENAME > ORGANIZATION > NAME > OCCUPATION > LOCATION > RELATION > MEDICAL_CONDITION > SYMPTOM > LEGAL_ISSUE > QUASI_COMBO > OTHER`

The legal categories (3 Nov) slot in so that a labelled account number is FINANCIAL rather than a generic ID, `Pune District Court` is COURT rather than LOCATION, and `Mehta Builders Pvt Ltd` is ORGANIZATION rather than NAME.

When a weaker span only partly overlaps a stronger one, the part nobody claimed is kept, as long as it still carries content. Recall comes first, so coverage is never silently lost.

## Positions the team should confirm (feed into D's annotation guide)

- **Age is treated as sensitive at any value.** This is stricter than HIPAA Safe Harbor, which only covers ages of 90 and over (spec §9).
- **Durations are not DATE.** `approximately 10 weeks` is left for the LLM pass, which matches the spec demo, where it appears as a QUASI_COMBO.
- **ORGANIZATION is not requested from Presidio.** In Enron it is almost always "Enron" itself, and in legal text spaCy calls `High Court`, `NI Act` and `Section 138` organisations. The legal layer finds only company names with a legal-form suffix (`Pvt Ltd`, `LLP`, `Builders`) or `M/s`; other employers and counterparties go to the LLM pass.
- **LEGAL_ISSUE is not produced by rules.** It is kept anyway (it is what lawyers search for), unless it is rare; the LLM pass finds it.
- **MEDICAL_CONDITION and SYMPTOM are not produced by rules.** They are left to the LLM pass.
- **Score threshold is 0.35.** It drops Presidio's near-zero guesses (US_BANK_NUMBER 0.05 on any digit run) and keeps phone numbers (0.4).

## Running

```bash
pytest tests/rules -q -m "presidio or not presidio"   # 79 tests, ~20 s (spaCy load dominates)
python -m sanitise.rules "My son is 32 years old and lives at 42 Elm Street."
sanitise detect --input data/fixtures/case_07.json --rules-only --verbose
```

The code is in `src/sanitise/rules/`; the pipeline reaches it through the seams
`src/sanitise/agents/detector/rules.py` and `merge.py`. Setup is in the main README.

`/v1/detect/rules` (Person C) is just `DetectorOutput(doc_id=..., spans=rule_detect(text))`.

The rule pass takes about 0.7 s on a 2000-character document once spaCy is loaded. Loading takes about 8 s once per process, and `get_analyzer()` caches it.

The EmailRecognizer's domain check is forced to the bundled public-suffix snapshot, so nothing touches the network.

The rule-pass dev tools are in `eval/` (`score_rules.py`, `build_gold.py`, `RESULTS.md`) and
score against A's own dev sets in `data/gold_draft/` and `data/gold_heldout/`. The official
gold set for the project's reported metrics is D's `data/gold/` (see `docs/evaluation.md`).
