# Sample data for mock mode

Small, fictional data that lets the API (and the dashboard) run end to end without a model
and before the real corpus exists. With `SANITISE_LLM_BACKEND=mock`, `sanitise serve` uses:

* the 20 documents in `data/fixtures/` as the corpus, sanitised deterministically at startup;
* `people.json`: eight people, four of them the people in fixtures case_01, case_04, case_09
  and case_20, plus four who share some of their attributes so candidate pools are not trivial;
* `attacks/`: one attack of each kind (direct, linkage, stepwise).

The real data (Person D) goes in `data/corpus/`, `data/attacks/` and the other folders listed
in docs/phase3_plan.md section 3; nothing here is used outside mock mode.
