# Private Matter Assistant: demo script

**Length:** about 6 minutes. Both pipelines appear side by side in the Next.js dashboard. The
dashboard plays back a Kaggle run, because Kaggle does not accept incoming connections.

## Before you start (30 minutes before)

1. Download `bundle.zip` from the Kaggle notebook's last cell and unzip it, for example to
   `bundle/`.
2. Start the API on the laptop, replaying the run:

   ```powershell
   $env:SANITISE_DOMAIN = "legal"; $env:SANITISE_REPLAY_DIR = "bundle"
   sanitise serve
   ```

   No bundle yet? `$env:SANITISE_LLM_BACKEND = "mock"` serves the 20-client demo slice
   (`data/legal/demo`) with canned answers instead.
3. Start the dashboard (`cd web; npm run dev`) and open http://localhost:3000.
4. Open `bundle/REPORT.md` and `bundle/charts/privacy_utility.png` in a second window.

Check that the Documents tab lists 300 documents and the Attacks tab lists nine attacks
(direct, linkage and stepwise against `c001`, `c002` and `c003`).

## The script

### 1. The problem (45 s)

> "A litigation firm wants an assistant over its own files. Pasting those files into a model
> leaks client names, PANs and case numbers. Worse, harmless answers add up: 'any
> cheque-bounce matters in Maharashtra?', 'which one involves a doctor?', 'how old?', and
> only one client fits. Section 132 of the Bharatiya Sakshya Adhiniyam and the DPDP Act make
> either leak serious."

### 2. One document, before and after (75 s)

Documents tab, open `c001_intake_note`. Point out, left to right:

- the client's name becomes `[CLIENT_1]`, and "Dear Mr. Bose" in the advice email gets the
  same token;
- PAN, phone, email, address and case number `S.C.C. No. 995/2024` become placeholders;
- "Judicial Magistrate First Class Court at Pune" becomes "a magistrate's court", and the
  amount becomes a band;
- the age, job and city name the client together, so they go up two steps
  ("middle-aged", "a professional", "a city in western India") instead of one;
- the legal issue and the advice are untouched.

> "Raw text never leaves the Kaggle notebook. Gemma 4 does the cleaning there, and only this
> right-hand version is ever sent to Gemini."

### 3. A lawyer's question (60 s)

Compare tab: *"What did we advise in past cheque-bounce matters where the cheque was returned
for insufficient funds?"*

Both answers give the same advice (a statutory demand notice to the drawer). Open the payload
inspector: the baseline payload has names and phone numbers, the protected one has
placeholders. Both pipelines retrieved the same four documents.

### 4. The attacks (2 min)

Attacks tab:

1. **Direct** (`direct_c001`): the baseline names the client in round 1; the protected
   pipeline never does.
2. **Linkage** (`linkage_c001`): one question with age, city, job and issue. Same result.
3. **Stepwise** (`stepwise_c001`): play the four rounds. Watch the candidate-pool chart:
   the baseline drops to one client; on the protected side the monitor refuses from round
   2 ("which one involves a doctor?"), so the pool never drops below 5.

> "Without the monitor the protected answers still never name the client, but the pool
> shrinks. The monitor is what stops narrowing over a whole conversation."

### 5. The numbers (60 s)

Show `REPORT.md`: the seven core metrics with their targets, the agent-level table, and the
privacy-utility scatter (the protected pipeline should sit top left). Read out:

- PII leakage rate (target ≤ 1%) and extraction ROUGE-L (target ≤ 0.20);
- utility retention (target ≥ 0.90): the cost of protection in answer quality;
- membership inference AUC (target ≤ 0.60). Some signal remains, because the legal issue
  and the advice are kept, and the report says so;
- Agent 1 before and after fine-tuning (`agent1_before.json`, `agent1_after.json`).

### 6. Close (15 s)

> "Fictional data only, no formal guarantee like differential privacy. A real firm would add
> access control, audit logs and a legal review. But the design keeps raw client text inside
> the firm, and the numbers show what it costs."

## If something fails

| Problem | Fallback |
|---|---|
| The API will not start | `SANITISE_LLM_BACKEND=mock` serves the demo slice |
| A question is not in the recording | the replay answers with a quote from the first record; say so, and use the prepared questions |
| The dashboard will not build | show `REPORT.md` and the charts, then the answers in `results/answers.jsonl` |
