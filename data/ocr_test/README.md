# OCR test set

45 synthetic documents (5 of each of 9 types) whose exact text is known, for measuring the
PDF/image input step (`src/sanitise/ingest/`). Every person, hospital, bank, ID number and phone
number is made up. Rebuild or resize with `python tools/make_ocr_testset.py` (`--per-type`,
`--seed`, `--types`); score with `python tools/ocr_eval.py` (see `docs/document_input.md`).

| Type | Domain | File | What it tests |
|---|---|---|---|
| `discharge_e` | health | PDF, text layer, 2 pages | electronic discharge summary; letterhead, patient banner and footer repeat on page 2 |
| `lab_e` | health | PDF, text layer | electronic lab report with a results table |
| `discharge_scan` | health | PDF, images only, 2 pages | the discharge summary scanned (tilt, blur, noise, grey) |
| `lab_photo` | health | JPEG | phone photo of a lab report (perspective, shadow, paper tint) |
| `prescription_hw` | health | JPEG | OPD prescription: printed pad, handwritten patient line, complaints, Rx and advice |
| `consent_form_hw` | health | JPEG | printed consent form filled in by hand |
| `mixed_file` | health | PDF, 3 pages | electronic lab page + scanned consent form + prescription photo |
| `bank_statement_e` | finance | PDF, text layer | bank statement: account holder, account no, IFSC, transactions table |
| `affidavit_scan` | law | PDF, images only | scanned affidavit with a signature |

Handwriting is drawn with nine handwriting fonts (neat to hard cursive), word by word on a
wobbling baseline. That is still neater than real handwriting, so the evaluation also reads real
handwritten lines from the IAM database (`--iam N`).

## gold.jsonl

One JSON object per document:

| Field | Meaning |
|---|---|
| `id`, `file`, `type`, `domain`, `about` | which document |
| `routes` | per page, how it should be read: `text` (text layer) or `vision` (vision model) |
| `gold` | the full text, with repeated header/footer lines kept only on their first page |
| `gold_pages` | the text of each page as printed (headers/footers on every page) |
| `removals` | the header/footer lines a correct removal takes out (`page`, `line`) |
| `details` | the private details in the document (names, phones, ID numbers, address parts) |

Gold text is written line by line in reading order: table rows as their cells separated by
spaces, `Label: value` pairs as printed. The scorer ignores case, spacing, Markdown marks and
`[signature]` / `[stamp: ...]` / `[illegible]` tags, so only reading errors count.
