# Agent 1: rule pass evaluation (Person A)

Re-run with `python eval/score_rules.py data/gold_heldout --show` (or `data/gold_draft`). Match rule from spec §3.2: overlap plus same category.

## Headline: 10 held-out documents, 98 gold spans

These documents were written and annotated **before** the round-2 rule fixes and never used to tune them. They're in `data/gold_heldout/`.

| | Precision | Recall | F1 | Leak coverage |
|---|---|---|---|---|
| Before fixes | 0.91 | 0.78 | 0.84 | 0.83 |
| **After fixes** | **0.95** | **0.88** | **0.91** | **0.92** |

| Category | Precision | Recall | Gold spans |
|---|---|---|---|
| NAME | 1.00 | 0.91 | 23 |
| LOCATION | 0.85 | 0.86 | 14 |
| RELATION | 1.00 | 0.91 | 11 |
| DATE | 0.83 | 1.00 | 10 |
| PHYSICAL | 1.00 | 1.00 | 9 |
| OCCUPATION | 1.00 | 0.56 | 9 |
| AGE | 1.00 | 0.88 | 8 |
| PHONE | 1.00 | 1.00 | 6 |
| ID_NUMBER | 1.00 | 0.67 | 3 |
| EMAIL | 1.00 | 1.00 | 3 |
| ADDRESS | 1.00 | 0.50 | 2 |

Still missed:
- occupations with no cue phrase ("as Senior Analyst", "pharmacist in Toronto")
- some single first names, and spaCy calling "Chris Germany" a place
- "Happy 60th!"
- Aadhaar numbers written with spaces
- small towns ("Thane")

## What each layer contributes (held-out set)

`python eval/score_rules.py data/gold_heldout --ablation`

| Layers | Precision | Recall | F1 | Leak coverage |
|---|---|---|---|---|
| core (Presidio + spaCy only) | 0.79 | 0.48 | 0.60 | 0.59 |
| + spec (AGE, PHYSICAL, RELATION) | 0.91 | 0.74 | 0.82 | 0.81 |
| + general (address, labelled IDs, occupation) | 0.94 | 0.82 | 0.87 | 0.87 |
| + healthcare (hospitals, clinics) | 0.94 | 0.86 | 0.90 | 0.91 |
| + email (Enron phones, signature titles) | 0.95 | 0.88 | 0.91 | 0.92 |

Presidio alone catches under half of the identifiers. The spec layer gives the biggest single gain, and precision rises too, because "32 years old" becomes AGE instead of a wrong DATE.

## Development set: 20 documents, 152 gold spans

Its misses guided the round-2 fixes, so this score is optimistic. It's in `data/gold_draft/`.

| | Precision | Recall | F1 |
|---|---|---|---|
| Round 1 | 0.95 | 0.84 | 0.89 |
| Round 2 (tuned on this set) | 0.99 | 0.96 | 0.98 |

## False-alarm check: 1,000 real ChatDoctor (GenMedGPT-5k) patient messages

20 spans were flagged in total:
- 8 correct relations ("my baby", "my father")
- 4 relative dates kept by design ("a few days ago")
- 8 false alarms: capitalised medical terms that spaCy tagged as a person ("Fibroadenoma", "NASH", "Varicocele")

That is about **1 false alarm per 125 messages**. A medical-term stoplist for NAME would remove most of these.

## What changed in round 2

- named hospitals and clinics are now LOCATION
- ages: "my father is 67", "died at 55", "80th birthday"
- occupations after an age or gender ("26 yr old female nurse"), bare appositives ("…, retired army officer,") and email signature titles
- kinship terms used as names ("Hi Mom")
- decimal weights ("2.4 kg")
- Enron extensions ("x3-5761")
- CNIC and NHS labels
- times, seasons and "weekend" are no longer DATE
- eponymous diseases ("Peyronie's disease") are no longer NAME

A regression was caught on real data and fixed: "Hi, doctor," was briefly being flagged as an OCCUPATION.

## Caveats

- **The gold sets are drafts.** Both were written and annotated by Claude, not Person D, and nobody has reviewed them yet. The official figure comes from D's reviewed gold set on real HealthCareMagic and Enron records.
- **The held-out set is small.** With 98 spans, ±0.05 is a fair error bar on recall.
- **This is the rule pass only.** It has no LLM pass, and MEDICAL_CONDITION and QUASI_COMBO are not scored.
