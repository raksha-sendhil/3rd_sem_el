# Real legal documents for testing the agents

Fourteen public legal documents for testing Agents 1-3 on real text, next to the fictional
training corpus in `data/legal/corpus/`. `manifest.json` gives each one's court, type, date,
source link, licence and spot checks; `docs/` holds the text.

**Each file is an excerpt, not the whole document.** It keeps the caption and the paragraphs
with the facts, in the source's own words. Page headers, footnote markers, digital-signature
stamps and the paragraphs of legal argument are left out, and `...` marks a cut inside a
paragraph. A short line under the title gives the citation.

| # | Document | Court or body | Type | Source |
|---|---|---|---|---|
| in01 | Saroj v. IFFCO-TOKIO General Insurance (2024) | Supreme Court of India | motor accident compensation; age from Aadhaar vs school certificate | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_10_939_951_EN.pdf) |
| in02 | Rajesh Kumar v. National Insurance Co. (2024) | Supreme Court of India | consumer complaint, insurance claim | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_12_1108_1116_EN.pdf) |
| in03 | Prakash Bhalotia v. Indra Chandra Goyal (2024) | Supreme Court of India | shop tenancy, eviction under the UP Rent Act | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_9_988_998_EN.pdf) |
| in04 | Lenin Kumar Ray v. Express Publications (2024) | Supreme Court of India | termination of an employee, Industrial Disputes Act | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_10_2303_2313_EN.pdf) |
| in05 | Tirith Kumar v. Daduram (2024) | Supreme Court of India | family land dispute, Hindu Succession Act and tribal custom | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_12_665_679_EN.pdf) |
| in06 | R. Shama Naik v. G. Srinivasiah (2024) | Supreme Court of India | specific performance of a sale agreement | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_11_1325_1328_EN.pdf) |
| in07 | Bijoy Kumar Moni v. Paresh Manna (2024) | Supreme Court of India | dishonoured cheque, Section 138 NI Act, statutory notice | [SC open data](https://indian-supreme-court-judgments.s3.amazonaws.com/data/pdf/year=2024/english/2024_12_1946_1996_EN.pdf) |
| in08 | Shivappa v. Manjunath JD (MFA 1790/2019) | High Court of Karnataka | motor accident appeal; cause title with S/o, age, occupation, addresses, vehicle and policy numbers, injuries | [HC open data](https://indian-high-court-judgments.s3.amazonaws.com/data/pdf/year=2024/court=29_3/bench=karnataka_bng_old/KAHC010115312019_1_2024-07-19.pdf) |
| in09 | Ashish Kapur v. Deepak Seth (ARB.P. 1524/2024) | High Court of Delhi | arbitration; partnership deeds, spouses, LLPs | [HC open data](https://indian-high-court-judgments.s3.amazonaws.com/data/pdf/year=2024/court=7_26/bench=dhcdb/DLHC010659682024_1_2025-02-03.pdf) |
| in10 | Yogesh Vig v. Sunila Sharma (RC.REV. 258/2019) | High Court of Delhi | two-line adjournment order (a short control document) | [HC open data](https://indian-high-court-judgments.s3.amazonaws.com/data/pdf/year=2024/court=7_26/bench=dhcdb/DLHC010201312019_1_2024-11-27.pdf) |
| in11 | Brijesh v. CPIO, I&FC Department, GNCTD (2026) | Central Information Commission | RTI second appeal; officials, application numbers, a phone number | [Indian Kanoon](https://indiankanoon.org/doc/131026555/) (official copies: [cic.gov.in](https://cic.gov.in/decision)) |
| in12 | iGATE Global Solutions: amendment to the CFO's employment agreement (2011) | none (a contract) | Indian executive contract: home address, age, pay | [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/1024732/000119312511037679/dex102.htm) |
| uk01 | Department for Work and Pensions v. Kiani [2026] EAT 168 | Employment Appeal Tribunal (UK) | dismissal for opening a son's benefit record; health details | [Find Case Law](https://caselaw.nationalarchives.gov.uk/eat/2026/168) |
| us01 | Lucy v. Zehmer, 196 Va. 493 (1954) | Supreme Court of Appeals of Virginia | sale of a farm written on a restaurant check | [Caselaw Access Project](https://static.case.law/va/196/cases/0493-01.json) |

## Sources and licences

- **Indian Supreme Court and High Court judgments**: the open datasets on the AWS Open Data
  Registry (`indian-supreme-court-judgments`, `indian-high-court-judgments`), released under
  CC-BY-4.0, which collect the courts' own published PDFs. Judgments are public records, and
  Section 52(1)(q) of the Copyright Act, 1957 allows judicial proceedings to be reproduced.
- **Central Information Commission**: the decision as published on Indian Kanoon; the
  Commission publishes the same orders at cic.gov.in.
- **SEC EDGAR**: an exhibit to iGATE Corp's 2010 annual report, a public filing.
- **UK**: The National Archives' Find Case Law, under the Open Justice Licence.
- **US**: the Caselaw Access Project (Harvard Law School Library), public domain.

## Choices

- Mostly Indian, as the project is: Supreme Court, two High Courts, a statutory commission and
  an Indian contract, with Indian identifiers (S/o, "aged about", PIN codes, vehicle registration
  numbers, Rs amounts in lakhs, policy and RTI numbers). The UK and US documents check that the
  agents are not tuned only to Indian formats.
- A mix of shapes: a cause title in capitals, narrative facts, a quoted notice, a contract
  with a signature block, a bilingual commission order (English kept), and a two-line order.
- Left out: cases about sexual offences or children, where Indian law restricts publishing
  identities, and a maintenance case that named a minor.
- These are real people's names from public records. Use them only to test the agents, and do not
  republish the outputs as statements about the people.

## Spot checks

`must_hide` lists identifiers that should not survive Agent 3, and `must_keep` lists legal
terms that should. They are spot checks a reader can verify by eye, not a full gold
annotation. `tests/legal/test_real_docs.py` checks that every one appears in its document.
