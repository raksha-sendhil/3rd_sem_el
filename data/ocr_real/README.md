# Real OCR test set

150 real documents with known text, 50 per domain. The files are not in the repository (licences,
and the rule that no real document is committed): fetch them with

```bash
python tools/fetch_real_testset.py                  # 50 per domain, about 30 MB, a few minutes
python tools/fetch_real_testset.py --per-domain 20  # smaller, same mix
```

The script writes `docs/` and `gold.jsonl` here (same format as `data/ocr_test`, plus `source`,
`licence`, `url` and `order`). It always picks the same items, so everyone gets the same set.

| Domain | Source | Items | What it is | Licence |
|---|---|---|---|---|
| health | [ClinOCR-Bench](https://huggingface.co/datasets/Daniele0025/ClinOCR-Bench) | 30 | clinical documents with fictional, AI-written content (radiology, lab, intake, discharge, medication forms), physically printed, then crumpled, photocopied, put in plastic sleeves, rotated and scanned; human-audited transcriptions. 6 subsets: normal 5, handwriting 6 (handwriting fonts, printed and scanned: not real handwriting), poor quality 5, rotated 5, tables 5, mixed artifacts 4. Real paper and scanning, synthetic content, so no patient data | MIT |
| health | [Doctor's Handwritten Prescription BD](https://huggingface.co/datasets/lance-format/handwriting-ocr) | 20 | real doctors' handwriting: medicine names cut from real prescriptions (Bangladesh), with labels; 20 different medicines spread A to Z | research use |
| finance | [SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) | 25 | real scanned receipts and tax invoices (ICDAR 2019) | research use |
| finance | [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 25 | real receipt photos | CC BY 4.0 |
| law | [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad) | 35 | real contracts filed with the SEC, 2 pages each: 18 as the PDF itself (text layer), 17 scanned (rendered and degraded like the synthetic scans) | CC BY 4.0 |
| law | [FUNSD](https://guillaumejaume.github.io/FUNSD/) | 15 | real noisy scanned forms, faxes and letters from the tobacco litigation document archive | non-commercial research |

**Gold order.** ClinOCR-Bench, the prescriptions and CUAD give the text in reading order (`order:
exact`). SROIE, CORD and FUNSD give words with boxes. Their gold text is rebuilt line by line (same
row = same line, left to right), so the order is approximate (`order: approximate`). For those,
CER also counts order differences where no word was misread. Quote the order-free word F1 for
them.

**Checks.** A CUAD page is used only if its text layer reads as printed: no unknown glyphs, no
proof watermark, no quotes turned into `?`, no lost "fi" ligatures. US Supreme Court opinions were
tried and dropped: the preliminary prints carry a "Page Proof Pending Publication" watermark inside
their text layer and lose their "fi" ligatures ("fve", "affrmed").

**What it does not cover.** Real hospital records with patient data (none can be published), real
handwritten full prescriptions with line transcriptions (no public set found; the BD set has
single medicine names), Indian-language documents, and real bank statements (private by nature;
the synthetic set has them).
