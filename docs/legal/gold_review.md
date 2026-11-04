# Gold-set review checklist (legal)

The spec asks for `gold_facts/` and `gold_routing/` (20 documents) to be hand-labelled by D
and reviewed by B. The generator drafts both from what it wrote, so the files exist from the
start. Each one carries `"annotator": "generator"` and `"reviewed_by": null` until a person has
checked it. Set `reviewed_by` to your initial when a file is done.

## `gold/` (300 span files, Agent 1)

Written while the documents are generated, so the offsets are exact. Spot-check 20 files with
`python -m sanitise.evaluation.validate_gold --gold data/legal/gold` (0 errors expected) and
look for:

- [ ] NAME spans without titles ("Bose", not "Mr. Bose"); lawyers and judges are NAME too
- [ ] ORGANIZATION without "M/s"; courts are COURT, not ORGANIZATION
- [ ] ADDRESS as one span from the flat or house number to the PIN code
- [ ] a QUASI_COMBO wherever the client's occupation, city and legal issue appear together

## `gold_facts/` (Agent 2)

- [ ] every fact's `value` appears word for word in the document
- [ ] the advice, documents requested, outcome and next step are `high`; the court and the
      legal issue are `high` (the spec's coverage target names them)
- [ ] subjects are `client`, `opposing_party`, `witness`, or `person_<n>` for lawyers
- [ ] nothing that matters is missing (add facts by hand; the generator only knows its own
      template fields)

## `gold_routing/` (Agent 3)

The actions come from the spec's policy table, written out separately in the generator rather
than imported from the router, so they are an independent check.

- [ ] placeholder: NAME, PHONE, EMAIL, ID_NUMBER, FINANCIAL, CASE_NUMBER, ADDRESS,
      MATTER_CODENAME
- [ ] generalise: ORGANIZATION, COURT, LOCATION, DATE, MONEY, AGE, OCCUPATION, and the four
      rare legal issues
- [ ] keep: LEGAL_ISSUE (common), RELATION

## `qa.json` (B)

40 questions, one per fact pattern. Each question's `expected` phrases come from the advice
or document text, and `doc_ids` lists every document that contains them.

- [ ] the question names no client, place or date
- [ ] the expected phrase is something a lawyer would accept as the answer
- [ ] the expected phrase survives sanitisation (no names, numbers or dates in it)

## `data/ontology/` (B)

Generated from `src/sanitise/legal/vocab.py` by `python -m sanitise.legal ontology`, so every
city, job and issue in the corpus has a ladder. To add vocabulary, edit `vocab.py` and
regenerate rather than editing the JSON by hand.
