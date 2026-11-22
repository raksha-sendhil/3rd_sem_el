# Agent 1 against Gretel's labels

Rule pass only (no model), on the English documents of up to 2,000 characters in Gretel's **test** split: 2,322 documents. Found = every character of the labelled value is removed or generalised. Reproduce with

    python tools/finance/agent1_check.py --split test

Before = the pipeline copied from healthcare, with only the fix for a date-filter regex that hung on long digit runs (without it the count does not finish). After = the finance pipeline.

| label | labelled | found before | found after | partly after | missed after |
|---|--:|--:|--:|--:|--:|
| account_pin | 77 | 11 (14.3%) | 52 (67.5%) | 0 | 25 |
| api_key | 39 | 7 (17.9%) | 34 (87.2%) | 3 | 2 |
| bank_routing_number | 83 | 62 (74.7%) | 67 (80.7%) | 0 | 16 |
| bban | 78 | 5 (6.4%) | 74 (94.9%) | 0 | 4 |
| company | 2272 | 117 (5.1%) | 1293 (56.9%) | 148 | 831 |
| credit_card_number | 71 | 45 (63.4%) | 64 (90.1%) | 6 | 1 |
| credit_card_security_code | 78 | 3 (3.8%) | 51 (65.4%) | 0 | 27 |
| customer_id | 87 | 36 (41.4%) | 56 (64.4%) | 2 | 29 |
| date_of_birth | 135 | 123 (91.1%) | 123 (91.1%) | 4 | 8 |
| driver_license_number | 62 | 43 (69.4%) | 45 (72.6%) | 9 | 8 |
| email | 761 | 733 (96.3%) | 733 (96.3%) | 12 | 16 |
| employee_id | 76 | 38 (50.0%) | 52 (68.4%) | 0 | 24 |
| first_name | 125 | 83 (66.4%) | 102 (81.6%) | 0 | 23 |
| iban | 90 | 85 (94.4%) | 88 (97.8%) | 2 | 0 |
| ipv4 | 66 | 65 (98.5%) | 65 (98.5%) | 0 | 1 |
| ipv6 | 73 | 52 (71.2%) | 52 (71.2%) | 0 | 21 |
| last_name | 92 | 34 (37.0%) | 48 (52.2%) | 2 | 42 |
| local_latlng | 60 | 0 (0.0%) | 56 (93.3%) | 4 | 0 |
| name | 3554 | 2527 (71.1%) | 2748 (77.3%) | 252 | 554 |
| passport_number | 63 | 51 (81.0%) | 53 (84.1%) | 0 | 10 |
| password | 50 | 1 (2.0%) | 26 (52.0%) | 0 | 24 |
| phone_number | 494 | 365 (73.9%) | 475 (96.2%) | 5 | 14 |
| ssn | 73 | 66 (90.4%) | 68 (93.2%) | 0 | 5 |
| street_address | 1968 | 488 (24.8%) | 1709 (86.8%) | 183 | 76 |
| swift_bic_code | 122 | 2 (1.6%) | 120 (98.4%) | 0 | 2 |
| user_name | 24 | 2 (8.3%) | 7 (29.2%) | 1 | 16 |
| **all** | 10673 | 5044 (47.3%) | **8261 (77.4%)** | 633 | 1779 |

Money amounts changed: 8 before, 2 after, of 5,150.

## How it was tuned

- The rules were tuned on the first 1,500 rows of the **train** split (49.4% -> 78.1%, at the last tuning count). Test misses were never printed or read.
- The KYC evaluation documents come from the test split. While the data builder was checked, some of its questions, answers and one NIST answer file were seen. The spaced SSN pattern follows the NIST form's box layout, so SSN recall on those scans is not an independent result.
- The quick corporate and cheque results prompted the amount fixes. Those fixes were tried on train documents and an invented report page, not on the evaluated documents.

## Reading it

- About one labelled value in five is not fully found by rules alone. Companies, names, user names and passwords are the weakest. The LLM pass adds to this, and its share is not measured here.
- Some labels are noise ("Date of Birth" as a date of birth, "[Phone Number]" as a phone), so 100% is not reachable.
- `date` and `time` labels are not counted ("monthly" and "30 years" are labelled dates).
