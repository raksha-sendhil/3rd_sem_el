# Example case files

Made-up clinical records (every name, number and place is fictional) for trying the case-file
support: paste one into the dashboard's Documents page. A document is treated as a case file when
it has three or more labelled clinical sections (C/O, H/O, Diagnosis, Plan, MRN ...); anything
else goes through the pipeline exactly as before.

| File | Style |
|---|---|
| `opd_child_malaria.txt` | Indian outpatient case sheet: a child on antimalarials, guardian's phone, vitals, labs, plan |
| `us_progress_note_copd.txt` | US clinic progress note: tab-separated header fields, medications, vitals, assessment |
| `discharge_summary_mi.txt` | Indian discharge summary: heart attack, procedure, discharge medications, follow-up |
