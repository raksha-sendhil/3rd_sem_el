# 300 real Indian judgments for testing the agents

300 judgments and orders from Indian courts, built by `tools/legal/build_indian_corpus.py`
(seed 2026). `manifest.json` gives each one's court, case number, date, the link to the court's
PDF, and its spot checks.

| Court | Documents |
|---|---|
| Supreme Court of India (2023, 2024, 2025: 30 each) | 90 |
| High Courts of Karnataka, Delhi, Bombay (Aurangabad), Kerala, Madras, Calcutta | 24 each |
| High Courts of Telangana, Gujarat, Gauhati | 22 each |

**Source.** The open datasets of the Supreme Court of India and the Indian High Courts on the
AWS Open Data Registry (`indian-supreme-court-judgments`, `indian-high-court-judgments`), under
CC-BY-4.0. They hold the courts' own published PDFs and case metadata. The High Court orders are
from 2024.

**Each file** is the start of the judgment: the cause title (parties, often with S/o, age,
address) and the opening facts, at most 1,800 characters, so it goes through the agents in one
part. Page headers, page numbers, margin letters and digital-signature stamps are removed.

## How each document was checked

The builder tries candidates in a fixed random order, spread across case types, and keeps one
only if:

1. The PDF has an English text layer that reads as words. Scans and Hindi-only orders are out.
2. It does not concern a sexual offence, a child or a minor, where Indian law restricts naming
   people.
3. The order is not a non-prosecution, withdrawal or default dismissal, which has little text.
4. At least one party who is a person, taken from the court's own metadata, appears in the
   excerpt. The State, companies and officials are not counted.
5. At least one legal term appears: a section, an Act, or a word such as bail, eviction or decree.

813 candidates were tried to keep 300.

**`must_hide`** lists the party names from (4), plus any phone numbers, emails, PAN,
Aadhaar-style numbers, vehicle numbers and S/o-W/o-D/o names in the excerpt. None of these come
from our agents. **`must_keep`** lists the legal terms from (5). They are checks, not a full
annotation: a document has other personal details (villages, ages, dates) that the lists do not
name. Those show up as "direct ids left" or in the coloured view.

## Baseline (rules only, no model)

Agent 1's rules with a fixed rewrite, on all 300:

| | Result |
|---|---|
| documents failed | 0 |
| must-hide identifiers hidden | 265/516 (51%) |
| must-keep terms kept | 562/705 (80%) |

| Court | hidden | kept |
|---|---|---|
| Gujarat HC | 28% | 88% |
| Kerala HC | 30% | 87% |
| Karnataka HC | 34% | 88% |
| Delhi HC | 34% | 83% |
| Bombay HC | 46% | 78% |
| Gauhati HC | 51% | 87% |
| Madras HC | 53% | 86% |
| Calcutta HC | 59% | 89% |
| Supreme Court | 69% | 69% |
| Telangana HC | 81% | 80% |

Most misses are names written in capitals in cause titles ("ATULBHAI VITHALBHAI BHANDERI",
"RAMBHA DEVI"), which the rule pass does not recognise. The model pass, run with Gemma 4 on
Kaggle, has to close this gap.

## Rebuilding

```bash
pip install pandas pyarrow pypdf        # pdftotext (poppler) is used when installed
python tools/legal/build_indian_corpus.py
```

Downloads are cached in `data/raw/indian_judgments/`, which is not committed.
