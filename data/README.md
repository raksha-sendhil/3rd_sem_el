# Data (owner: Person D)

| directory         | contents                                                         |
|-------------------|------------------------------------------------------------------|
| `fixtures/`       | the 20 fixture documents, one file each (`case_01.json` ...)     |
| `gold/`           | one gold annotation file per fixture, spec 8.3 format            |
| `raw/`            | full ChatDoctor / Enron dumps — git-ignored, never commit       |

**Fixture file** — any format `sanitise.loader` reads; the simplest is `{"text": "..."}`.
ChatDoctor records (`{"instruction", "input", "output"}`) work as-is: the patient
`input` is the document unless `SANITISE_CHATDOCTOR_INCLUDE_ANSWER=true`.

**Gold file** (spec 8.3):

```json
{
  "doc_id": "a3f2c91b4e7d8a65",
  "text": "My son has been experiencing left-sided abdominal pain ...",
  "spans": [
    { "text": "my son",       "category": "RELATION", "start": 0,   "end": 6 },
    { "text": "32 years old", "category": "AGE",      "start": 182, "end": 194 },
    { "text": "rare occupation combined with a specific condition",
      "category": "QUASI_COMBO", "start": null, "end": null }
  ],
  "annotator": "D",
  "reviewed_by": "A"
}
```

`doc_id` must equal `sanitise.loader.compute_doc_id(text)`; `text` must be the
exact document text so offsets line up. `sanitise score` checks every offset
against `text` and names the file if one does not fit.
