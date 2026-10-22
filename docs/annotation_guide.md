# Annotation Guide - Agent 1 (Sensitive Information Detector)

**Owner:** Person D  |  **Version:** 1.2  |  **Status:** Signed off 18 Oct 2026 (all ⚑ items accepted as proposed; rulings D11-D20 added after review). 22 Oct: `SYMPTOM` category added (D21-D23), team sign-off pending

This guide is the specification for what the detector is expected to catch. If you annotate
a case the guide does not cover, decide it, then add the rule to the **Decision log** at the
bottom. Never silently improvise.

---

## 0. Day 1 decisions (signed off)

| # | Decision | Proposed position | Why |
|---|---|---|---|
| ✔1 | Offset convention | Python string indices, **end-exclusive**: `text[start:end] == span text` | Matches the spec example (`"my son"` is 0-6). Person A's output must use the same. |
| ✔2 | Is age sensitive? | **Yes. Every age is annotated `AGE`.** | HIPAA Safe Harbor allows exact ages under 90, so this is stricter than the regulation. We choose strict because age is a core quasi-identifier and the session monitor needs it as a trackable attribute. Cost: lower precision if the detector is over-eager; recall is the primary figure anyway. |
| ✔3 | One-to-one matching | **Yes.** One gold span is credited by at most one prediction. | Stops one large predicted span or duplicate predictions from inflating the score. Not in the original spec, so the team must approve it. |
| ✔4 | Category list | The 15 values in `schemas.py` (`SYMPTOM` added 22 Oct, D21) | Changing it after annotation starts invalidates annotations, so adding `SYMPTOM` meant relabelling all 20 gold documents. |
| ✔5 | Plain symptoms | **Annotated as `SYMPTOM`** (3.14). Changed 22 Oct (D21); until then they were not annotated | A separate category keeps symptoms apart from diagnoses: later stages can keep symptoms (useful for answering questions) while protecting rare diagnoses, and the session monitor can track them separately. |

---

## 1. General rules

1. **Annotate from the text and this guide only.** Never annotate by looking at what the detector outputs.
2. **Span text is copied exactly** from the document, including internal punctuation. No leading or trailing whitespace.
3. **Spans never overlap.** If two categories compete for the same characters, use the precedence in section 4.
4. **Annotate the shortest span that fully carries the identifier.** `32 years old`, not `He is 32 years old,`.
5. **Annotate every occurrence**, not just the first. If "Marta" appears twice, that is two `NAME` spans.
6. **Annotate only what applies to a specific individual** in the document. General statements about people or diseases are not annotated (see 3.12).
7. **When unsure, annotate** (the system is deliberately conservative: a miss is a leak), then flag the span in the review so the second person rules on it.
8. Offsets are checked by `python -m sanitise.evaluation.validate_gold`. Run it before every handoff.

---

## 2. Categories at a glance

| Category | Annotate | Do not annotate |
|---|---|---|
| NAME | Personal names, full or partial, including first-name-only greetings and sign-offs | Company names, drug names, a name inside an email address (the whole address is `EMAIL`) |
| AGE | Any statement of a person's age | Durations ("two weeks"), "elderly", "child" |
| DATE | Calendar dates and years that refer to an event: `3 March 2022`, `2015`, `Tuesday, October 9, 2001` | Durations (`10 weeks`), vague time (`last month`, `next week`, `yesterday`) |
| PHONE | Phone numbers in any format | Policy or ID numbers |
| EMAIL | Email addresses | - |
| ID_NUMBER | Patient, employee, policy, account, national ID numbers | Phone numbers, dates, measurements |
| ADDRESS | Street-level addresses, including flat/apartment and postcode | A city on its own (that is `LOCATION`) |
| LOCATION | Named city, town, island, hospital, school, university, named workplace site | Generic places (`the hospital`, `home`) |
| OCCUPATION | A person's job or role | Generic activity not a job (`I walk daily`) |
| RELATION | Kinship term that points at a specific person: `my son`, `His wife`, `My father` | Generic mentions (`parents should...`) |
| PHYSICAL | Height, weight, build | Blood pressure, lab values, test results |
| MEDICAL_CONDITION | A named diagnosis, disease or specific clinical finding that applies to a specific person | Plain symptoms (those are `SYMPTOM`), procedures, drugs, general medical statements |
| SYMPTOM | A symptom or complaint that a specific person has: `fever`, `dry cough`, `left-sided abdominal pain` (see 3.14) | Negated or only-asked-about symptoms, named diagnoses (`MEDICAL_CONDITION`), general statements |
| QUASI_COMBO | A combination of attributes that together narrow the candidate pool (see 3.13) | A combination made only of attributes already common in the population |
| OTHER | Sensitive identifying info that fits nowhere else. Use rarely and write the reason in the decision log | - |

---

## 3. Category rules with examples

### 3.1 NAME
- ✅ `Priya Raman` · `Marta` in "Hi Marta," · `Daniel` in a sign-off · `Anita D'Souza`
- ❌ `Enron`, `metformin`, `rajesh.kumar84@examplemail.com` (this is `EMAIL`)
- Titles are excluded from the span: annotate `Ortega` in "Dr. Ortega".

### 3.2 AGE (✔2)
- ✅ `34 year old` · `6 years old` · `92 years old` · `58 years old`
- ✅ Annotate the number plus its unit words as written: `32 years old`, not just `32`.
- ❌ `an elderly man`, `for 10 weeks`, `in her thirties` (vague; see decision log D4)

### 3.3 DATE
- ✅ `21 January 2024` · `2015` (a year naming an event) · `Tuesday, October 9, 2001`
- ❌ `last month`, `next week`, `for approximately 10 weeks`, `two days`
- A time of day (`3:42 PM`) is not annotated.

### 3.4 PHONE / EMAIL
- ✅ `+91 98450 12345` · `(415) 555-0142` · `713-555-0199` · `anita.dsouza@examplemail.com`

### 3.5 ID_NUMBER
- ✅ `HC-2291746` (patient ID) · `E-40871` (employee) · `POL-77-410398` (policy)
- Annotate the number only, not the label (`Patient ID:` is excluded).

### 3.6 ADDRESS vs LOCATION
- ADDRESS: `14/3 Lakeview Apartments, Bannerghatta Road, Bengaluru 560076` (one span, whole address)
- LOCATION: `Coonoor`, `Pune`, `Greenfield Primary School`, `AIIMS Delhi`
- If a LOCATION is part of an ADDRESS it is inside that one ADDRESS span, not a second span.
- `the Tulsa desk` → annotate `Tulsa` only.

### 3.7 OCCUPATION
- ✅ `pediatric neurosurgeon` · `lighthouse keeper` · `tea taster` · `radiographer` · `undergraduate student`
- Annotate without the article: `lighthouse keeper`, not `a lighthouse keeper`.
- Qualifiers that make the job rarer stay in: `national-level fencer`, `natural gas scheduler`.

### 3.8 RELATION (spec example: `my son`)
- ✅ `My son` · `my wife` · `His wife` · `My father` · `My sister`
- Annotate the possessive plus the kin word, as written. Case follows the text.
- A possessive form is split: in `my husband's reports`, annotate `my husband` only.
- ❌ `A classmate`, `a neurologist` (not kinship)

### 3.9 PHYSICAL
- ✅ `6 ft 3 inches` · `approximately 220 pounds` · `78 kg` · `5 ft 9 inches`
- Qualifiers like `approximately` stay inside the span if they precede the number (matches the spec example).
- ❌ `150/95` (blood pressure is a clinical value, not body build)

### 3.10 LOCATION for institutions
Hospitals, schools and universities are `LOCATION` because they narrow where a person is.
- ✅ `Fortis Hospital, Mulund` (one span) · `St. Thomas Hospital` · `IIT Madras`

### 3.11 DATE ranges
`from 1 August 2001 to 30 September 2001` is **two** DATE spans, one per date.

### 3.12 MEDICAL_CONDITION (✔5)
Annotate when the text names a diagnosis, disease or specific clinical finding **about a specific person**.
- ✅ `vertigo` · `type 2 diabetes` · `asthma` (a classmate's) · `multiple myeloma` · `leukocytosis` · `enlarged lymph nodes` · `neuropathy` (even when suspected: "Can this be neuropathy?")
- ❌ `abdominal pain`, `cough`, `headache`, `fever`, `rash`, `weakness`, `palpitations`: these are `SYMPTOM` (3.14)
- ❌ `coronary artery bypass`, `C-section` (procedures)
- ❌ "vitamin D deficiency is common in people who work indoors" (general statement, no specific person)
- A symptom is a `SYMPTOM` span (3.14) and can also be part of a `QUASI_COMBO` (3.13).
- Qualifiers like `mild` are left out: annotate `leukocytosis`, not `mild leukocytosis`.

### 3.13 QUASI_COMBO
Positions are not annotated; write a short description of the combination in `text`.
**Test:** does the combination narrow the candidate pool to a small group, when each member on its own would not? Annotate if **two or more** of these apply:
- a rare or niche occupation
- a rare condition, or an unusual combination of symptoms
- a small or specific location (small town, island, named institution)
- a precise timeline (exact date plus a named place)
- an extreme value (age 90+, unusual height)

✅ `rare occupation (lighthouse keeper) + single island location + dermatitis`
✅ `precise delivery date + named hospital + age`
✅ `left-sided abdominal pain + bilateral pelvic pain (symptom combination)`
❌ `age 41 + librarian` (both common in the population)
❌ `type 2 diabetes + a big city` (common)
❌ A combination that only restates a single explicit span

One description per distinct combination. Do not list several overlapping descriptions of the same combination.

### 3.14 SYMPTOM (✔5, D21-D23)
Annotate a symptom or complaint that a **specific person in the document has or had**.
- ✅ `fever` · `headache` · `dry cough` · `wheezing` · `breathless` · `palpitations` · `forgetful`
- ✅ Keep the body part or type it affects: `left-sided abdominal pain`, `rash on both hands`, `ringing in my left ear`, `swelling of the tongue`, `weakness on his right side`.
- Leave out severity and frequency words before it (`mild`, `severe`, `persistent`) and durations (`for two days`): `persistent rash on both hands` → `rash on both hands`.
- A later mention of the same symptom is annotated again (rule 5): "The `pain` is worse after meals".
- ❌ Negated or only asked about: `no fever`, "when should I worry about a headache?" (case_10 annotates the first `headache`, which the writer has, not the one in the question).
- ❌ Named diagnoses and specific clinical findings stay `MEDICAL_CONDITION`: `vertigo`, `leukocytosis`, `enlarged lymph nodes`.
- ❌ Lab values and measurements (`150/95`), drugs, procedures, general statements ("vitamin D deficiency is common…").

---

## 4. Precedence when categories compete for the same characters

Use the **most specific** category. This is the order the detector applies
(`CATEGORY_PRECEDENCE` in `src/sanitise/rules/categories.py`), so annotators and code agree:

`EMAIL > ID_NUMBER > PHONE > ADDRESS > AGE > PHYSICAL > DATE > NAME > OCCUPATION > LOCATION > RELATION > MEDICAL_CONDITION > SYMPTOM > QUASI_COMBO > OTHER`

AGE and PHYSICAL rank above DATE because Presidio tags "32 years old" as a date; ID_NUMBER
ranks above PHONE because long ID numbers look like phone numbers. MEDICAL_CONDITION ranks above
SYMPTOM so a named finding wins over a symptom on the same words.

Examples: `daniel.whitcomb@example-energy.com` → `EMAIL` only (no `NAME`). A hospital inside an address → part of `ADDRESS`.

---

## 5. Review protocol

1. Annotator sets `"annotator"`. Reviewer is a different person and sets `"reviewed_by"` only after reading every span.
2. Reviewer checks: missed spans, wrong category, boundary errors, QUASI_COMBO judgement.
3. Disagreements go to the whole team. The ruling is added to the Decision log in the same hour.
4. A change to a ruling means re-checking every earlier document for the same pattern.
5. `python -m sanitise.evaluation.validate_gold --gold data/gold --fixtures data/fixtures --strict` must pass before final scoring.

---

## 6. Decision log

| ID | Date | Question | Ruling | Raised by | Settled by |
|---|---|---|---|---|---|
| D1 | Day 1 | Offsets | Python indices, end-exclusive | D | team |
| D2 | Day 1 | Age sensitive? | Yes, every age is `AGE` | D | team |
| D3 | Day 2 | Plain symptoms | Not annotated alone; only inside QUASI_COMBO. **Superseded by D21.** | D | team |
| D4 | Day 2 | Vague ages (`in her thirties`, `elderly`) | Not annotated. Revisit if real data has many. | D | team |
| D5 | Day 2 | Procedures | Not `MEDICAL_CONDITION` | D | team |
| D6 | Day 2 | `RELATION` in third person (`His wife`) | Annotated, same as `my wife` | D | team |
| D7 | Day 2 | Suspected conditions (`Can this be neuropathy?`) | Annotated | D | team |
| D8 | Day 2 | General medical statements | Not annotated | D | team |
| D9 | Day 3 | Hospital / school / university category | `LOCATION` | D | team |
| D10 | Day 3 | Durations (`10 weeks`) | Not `DATE`; may support a QUASI_COMBO | D | team |
| D11 | 18 Oct | Who detects `MEDICAL_CONDITION`? | The LLM pass. Rules cannot tell a diagnosis from a symptom (3.12). The LLM quotes the condition and the pipeline grounds the quote to exact offsets, so it is scored like any positional span | C | team |
| D12 | 18 Oct | LLM quote not found in the text | Dropped as a hallucination. Applies to MEDICAL_CONDITION, OCCUPATION and LOCATION from the LLM pass | C | team |
| D13 | 18 Oct | `my baby` (a newborn) | `RELATION`: a kinship term pointing at a specific person (case_08) | C (review) | team |
| D14 | 18 Oct | A place named on its own and again inside an address | Two spans: `LOCATION` for the standalone mention, inside `ADDRESS` for the other (rule 5, every occurrence; case_13) | C (review) | team |
| D15 | 18 Oct | Exact date of a medical event + named hospital | `QUASI_COMBO` (precise timeline + named institution), as in case_08; added to case_20 | C (review) | team |
| D16 | 18 Oct | Relative time tagged as a date (`the early morning`, `quarterly`, `each day`) | Not `DATE`. A date needs a digit, a month, a weekday or a named holiday; the rule pass enforces this | C | team |
| D17 | 18 Oct | Synthetic fixtures | Accepted for Phase 2. Real ChatDoctor / Enron records may replace them later, annotated by hand | team | team |
| D18 | 22 Oct | Names and ages the rules miss ("She is Sunaina", "He is eleven") | Still `NAME` / `AGE`. The rule pass now catches "my name is X", "this is X" after a greeting, "Dr. X", "call me X" and ages written in words; the LLM pass may return `NAME` and `AGE` as a safety net | C | C (team sign-off pending) |
| D19 | 22 Oct | A 10-digit phone number that passes the UK NHS checksum | `PHONE`. A number counts as an NHS `ID_NUMBER` only when "NHS" is written just before it | C | C (team sign-off pending) |
| D20 | 22 Oct | Hospital department or ward ("pediatrics department") | Part of the place, never `OCCUPATION` | C | C (team sign-off pending) |
| D21 | 22 Oct | Symptoms as their own category | New category `SYMPTOM` (3.14); supersedes D3. All 20 gold documents relabelled: 17 `SYMPTOM` spans added (124 → 141); case_10 now has one span, so 2 documents have nothing sensitive | C | C (team sign-off pending) |
| D22 | 22 Oct | `SYMPTOM` span boundaries | Keep the body part or type; leave out severity / frequency words and durations | C | C (team sign-off pending) |
| D23 | 22 Oct | Negated or hypothetical symptoms (`no fever`, "should I worry about a headache?") | Not annotated | C | C (team sign-off pending) |
| | | *(add new rows here)* | | | |
