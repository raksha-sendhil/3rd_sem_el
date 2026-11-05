# PDF and image input

Hospitals, law firms and banks keep most of their records as PDFs and scans, not as plain text.
This step turns a file into text before Agent 1. Agents 1-3, the store and the RAG answers do not
change.

## What comes in

| Input | Examples | How it is read |
|---|---|---|
| Electronic PDF (has a real text layer) | discharge summaries, lab and radiology reports exported from the hospital system; bank statements; contracts and judgments | text layer, exactly (PyMuPDF4LLM; tables become Markdown tables) |
| Scanned printed page | outside reports a patient brings, old paper files, notarised affidavits | vision model |
| Printed form filled in by hand | admission and consent forms, insurance claim forms | vision model |
| Handwriting | OPD prescriptions, ward notes | vision model |
| Phone photo | a report or prescription sent on WhatsApp | vision model, after fixing rotation and size |
| Mixed PDF | one patient file: typed summary + scanned reports + photo of a prescription | page by page, each page its own way |
| Text file | `.txt`, `.md` | as is |
| Electronic health record | HL7 v2 messages (`.hl7`), FHIR JSON (`.fhir`, `.json`), CDA / CCDA XML (`.xml`) exported from a hospital system | written out as readable text: patient, encounter, diagnoses, observations, notes (`electronic_parser.py`) |

Out of scope: medical images (X-ray, CT, DICOM), ECG waveforms (ECG interval reports as text are
handled), audio and video. Only text leaves this
step, so logos, photos, signatures, stamps and barcodes never reach the store.

## The pipeline (`src/sanitise/ingest/`)

```
file ─► what is it? (first bytes: PDF / PNG / JPEG / TIFF / WebP / HEIC / HL7 / FHIR / CDA / text)
        ├─ health record (HL7 v2, FHIR JSON, CDA XML) ─► written out as text (electronic_parser.py)
        ├─ PDF, per page ─ trustworthy text layer? ─ yes ─► PyMuPDF4LLM: text + Markdown tables
        │                                          └ no ──► render at 200 dpi ─┐
        └─ image (each frame of a TIFF) ─► fix camera rotation, resize ────────┴─► vision model:
                                       Qwen3-VL 8B on Ollama (default) or Gemini (OCR_BACKEND=gemini)
     ─► repeated headers/footers removed after their first page
     ─► cut into parts of at most max_doc_chars (2,000) at paragraph breaks ─► Agents 1-3
```

* **Text layer check** (`pdf.py`): a page's text layer is used only if it has visible text (some
  scanners add an invisible OCR layer of poor quality; those pages are read again), at least 30
  characters, words that look like words (a broken font gives garbage), and it is not a
  full-page picture with a few words on top.
* **Vision model** (`vision.py`): `qwen3-vl:8b-instruct` on Ollama by default, so the page
  never leaves the machine. With `SANITISE_OCR_BACKEND=gemini` the same prompt goes to the Gemini
  API instead (`OCR_GEMINI_MODEL`, default `gemini-2.5-flash`; the agents' Gemini keys, tried in
  turn). That needs no GPU, but the page image goes to Google. Temperature 0. The prompt is
  the same for every domain: copy exactly, no corrections or expansions; each `Label: value` on its
  own line; tables as Markdown; `[illegible]` instead of a guess; `[signature]` and
  `[stamp: ...]` for marks. Fences and runaway repeated lines are removed from the reply.
* **Header/footer removal** (`headers.py`): a line in the first 12 or last 4 lines of a page that
  is on at least half the pages (and 2) is kept on its first page only. Lines match when they are
  the same apart from case, spacing and Markdown marks, or nearly the same (OCR noise) with exactly
  the same digits, so `Date: 12/03/2024` and `Date: 13/03/2024` are never merged. Only page numbers
  ("Page 2 of 4") match with different digits. A page is never emptied. Every removed line is a
  copy of a line that is kept, so removal cannot lose information. On noisy scans some copies
  stay in (see the results), which is the safe way to fail.
* **Parts** (`split.py`): the agents take at most 2,000 characters, so longer text is cut at blank
  lines, then line ends, then spaces. After header removal the patient banner is only in part 1.
* **Agent 1 rules** (`rules/recognizers.py`): new labelled IDs (UHID, IP/OP/MR/lab/bill/
  registration/claim/policy/cheque numbers, ABHA, Aadhaar, PAN, IFSC, GSTIN, CIN; values may
  contain `/`) and three unlabelled Indian formats (PAN `ABCDE1234F`, Aadhaar `1234 5678 9012`,
  IFSC `SBIN0001234`).

* **Simple file-to-text** (`facade.py`): `ingest_file` / `extract_text_multimodal`, used by the
  document loader to read a folder of PDFs, images and records. PDFs and images go through the
  pipeline above when a vision model answers; otherwise through the simpler `pdf_parser.py`
  (pypdf) and `image_parser.py` (Gemini Vision, Tesseract or EasyOCR).

Settings (environment variables, `SANITISE_` prefix): `OCR_BACKEND` (ollama / gemini), `OCR_MODEL`,
`OCR_GEMINI_MODEL`, `OCR_DPI` (200), `OCR_MAX_SIDE`
(2000 px), `OCR_MIN_SIDE` (1200 px), `OCR_TIMEOUT_S`, `INGEST_MAX_MB` (25), `INGEST_MAX_PAGES` (30),
`INGEST_STRIP_HEADERS` (true).

## How to use it

**Dashboard.** Documents -> Your own -> Choose file accepts PDFs and images. The API reads the
file (`POST /v1/ingest`), the text appears in the box with one chip per page (text layer or
vision model, seconds taken) and any warnings. Check or correct the text, pick a part if the file
was long, then run the three agents. The backend needs the vision model: `OCR = True` in cell 1
of `notebooks/backend_kaggle.ipynb`. A scanned page takes tens of seconds. A Cloudflare quick
tunnel cuts requests at 100 s, so use ngrok for multi-page scans.

**One file from the command line** (Kaggle cell 2 runs this):

```bash
python tools/ingest_file.py scan.pdf                # print each page's route and the text
python tools/ingest_file.py photo.jpg --sanitise    # and Agents 1-3's rewrite of each part
python tools/ingest_file.py prescription_hw_03      # a test-set document: also prints its CER
python tools/ingest_file.py https://example.org/report.pdf --no-strip --force-vision
```

**Metrics** (Kaggle cell 3 runs this): `notebooks/ocr_kaggle.ipynb`, or

```bash
python tools/fetch_real_testset.py              # once: download the 150 real documents (data/ocr_real)
python tools/ocr_eval.py --n 18 --iam 20        # 18 documents (every type once) + 20 IAM lines
python tools/ocr_eval.py --domain health        # all health documents, synthetic and real
python tools/ocr_eval.py --set data/ocr_real    # only the real set (150)
python tools/ocr_eval.py --types clinocr_handwriting,rx_handwritten_word
python tools/ocr_eval.py --no-vision --types discharge_e,lab_e,contract_e   # laptop, no GPU
```

**Unattended per-domain run:** `notebooks/ocr_bulk_kaggle.ipynb` with Kaggle's **Save Version ->
Save & Run All**. It runs on Kaggle's servers, so the laptop can be off. It writes one report per
domain plus IAM, `SUMMARY.md` and `ocr_results.zip` into the version's Output.

Readings are cached in `<out>/pred/`, so a run stopped by a disconnect continues where it
stopped. `--fresh` reads everything again, for example after changing the prompt or the model.

## The metric

**CER, the character error rate**: the characters that must be inserted, deleted or changed to
turn what was read into the known text, divided by the length of the known text. 0% is perfect and
2% means 2 wrong characters in 100. It is the standard OCR measure, so it can be compared with
published numbers. Before comparing, both texts are normalised so only reading errors count: case,
spacing, Markdown marks and the `[signature]`/`[stamp]`/`[illegible]` tags are ignored. An
`[illegible]` word therefore counts as missing.

The report also gives, from the same run:

* **WER**: the same measure on words;
* **word F1**: words read correctly whatever their order. Receipts and forms in the real set
  give only words with boxes; their gold text is rebuilt line by line, so its order is
  approximate and CER also counts order differences. Quote word F1 for those sources;
* **private details read exactly**: names, phones, IDs and address parts found exactly in the text.
  This is the privacy link: a detail read as "Ravl Kumar" is not found by Agent 1, yet still points
  at the person;
* **pages routed correctly**: text layer vs vision model;
* **header/footer removal**: share of repeated copies removed, and lines lost (removed lines whose
  content is nowhere else in the text; should always be 0).

## The test sets

Two sets, scored together or apart (`--set`, `--domain`).

**Synthetic, `data/ocr_test/` (45, in the repository).** Five each of 9 types: an electronic and
a scanned 2-page discharge summary, an electronic lab report and a phone photo of one, a
handwritten prescription, a hand-filled consent form, a mixed 3-page file, an electronic bank
statement and a scanned affidavit (35 health, 5 finance, 5 law). `tools/make_ocr_testset.py` builds
them from blocks whose text is known, so no hand labelling is needed. Every person and number is
made up. Scans and photos are degraded renders: tilt, perspective, shadow, blur, noise and JPEG.
Handwriting uses nine handwriting fonts, drawn word by word on a wobbling baseline. Only this set
has the private-details list. `--per-type` and `--seed` make bigger or different sets.

**Real, `data/ocr_real/` (150, in the repository).** 50 per domain, from public datasets that
come with their text; `tools/fetch_real_testset.py` rebuilds them. Sources, licences and the
research-use limits are in `data/ocr_real/README.md`.

| Domain | Items |
|---|---|
| health | 30 ClinOCR-Bench: clinical documents with fictional, AI-written content, physically printed, then crumpled, photocopied, put in sleeves, rotated and scanned, with human-audited text (real paper and scanning, synthetic content; its handwriting subset uses handwriting fonts). Subsets: normal, handwriting, poor quality, rotated, tables, mixed. 20 real doctors' handwritten medicine names from real prescriptions (Bangladesh), 20 different medicines A to Z |
| finance | 25 real scanned receipts and tax invoices (SROIE), 25 real receipt photos (CORD) |
| law | 35 real SEC contracts, 2 pages each, half as PDF and half scanned (CUAD), 15 real noisy scanned forms and letters (FUNSD) |

Synthetic handwriting is neater than real handwriting, so `--iam N` adds N real handwritten
English lines from the IAM Handwriting Database (Hugging Face `Teklia/IAM-line`). Quote the
ClinOCR handwriting subset, the prescription words and IAM as the handwriting numbers. Real
prescriptions are harder still, and no model reads all of them, which is why the dashboard shows
the text for checking before the agents run.

## Results so far

Without a GPU (laptop, `--no-vision`), text-layer route:

| Documents | CER | Word F1 | Notes |
|---|---|---|---|
| 15 synthetic electronic (discharge, lab, bank statement) | 0.0% | 100% | 115 of 115 private details read exactly |
| 18 real SEC contracts (CUAD, as PDF) | 0.3% | 99.5% | the rest is spacing in multi-column layouts |

Every page was routed correctly. Header removal on these 19 multi-page documents removed 60 of 61
repeated lines (letterhead, banner and the "Source:" line CUAD prints on every page) and lost
none.

Header removal on simulated vision output (the gold pages with each label on its own line and
random character errors): 126 of 126 repeated lines removed with no errors, 102 with 1% character
errors, 73 with 3%, and 0 lines lost in every case.

The vision-model numbers (scans, photos, handwriting, IAM) need the Kaggle GPU: run
`notebooks/ocr_kaggle.ipynb` and add its report here.

## Law and finance

The input step holds for both. Routing, the vision prompt and header removal have nothing medical
in them, and the test sets cover both: synthetic bank statements and affidavits, plus 50 real
finance documents (receipts, invoices) and 50 real legal ones (SEC contracts, scanned forms). What
differs is the documents, and so the parts around this step:

| | Law | Finance |
|---|---|---|
| Typical input | contracts, judgments, petitions, affidavits, notices; mostly electronic, older ones scanned; stamp paper, seals, notary stamps, signatures on every page | bank statements, invoices, salary slips, tax forms, KYC copies, cheques |
| Layout | long (20-200 pages), numbered clauses, footnotes, line numbers in the margin, case number in every header | long tables (statements run to many pages), borderless tables, totals |
| Works as is | text layers, scans, repeated headers (case number, firm letterhead), parts | text layers, ruled tables, scans |
| Changes needed | Agent 1 rules for case numbers (`W.P. No. 1234/2024`), bar enrolment numbers, survey/property numbers; parts cut at clause boundaries instead of 2,000 characters; the rewrite will need a section-by-section approach for long documents | Agent 1 rules for account numbers without a label, card numbers, UPI IDs (`name@bank`), CIN/GSTIN (labels already added); borderless tables can lose their columns in the text layer (the vision model with `--force-vision` is the fallback); totals and balances must be kept as they are |
| Not in scope | handwritten court records in regional languages | cheque images (MICR line), ID-card photos |

In short: the input step is shared. Each domain needs its own Agent 1 patterns and test documents.
Legal documents also need longer parts and a rewrite that works section by section.

## Known limits

* The vision path has been tested only with a mocked model and simulated output; it has not yet
  run on a GPU. Kaggle is the first real run.
* A scanned page takes tens of seconds on a T4. A 30-page scan is minutes.
* English only (the IAM set and the prompt); Indian-language documents need a test set first.
* Vision models can "correct" a misspelt drug or invent a plausible value. The prompt forbids it
  and the CER shows how often it happens, but the dashboard's check-the-text step is the real
  safeguard.
* Tables without ruling lines in electronic PDFs may come out as plain lines.
