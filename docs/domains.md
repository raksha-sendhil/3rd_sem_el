# Healthcare and legal: two pipelines, one app

The dashboard's **Healthcare / Legal** toggle switches between two complete pipelines:

| | Healthcare | Legal |
|---|---|---|
| Code | `src/sanitise` (this repo's own pipeline) | `src/sanitise_legal` (the legal branch's pipeline, imported as it is) |
| Agents 1-3 | rules + Gemini (`SANITISE_AGENTS_*`) | rules + the legal profile in `config/domains.json` (`gemma4:e4b` on Ollama by default) |
| Data | `data/corpus`, `data/attacks`, `data/fixtures` | `data/legal` (120 fictional clients, attacks, QA, real judgments), `data/legal/ontology` |
| Session monitor | ledger of kinds of detail + candidate pool over the store (`docs/session_monitor.md`) | candidate pool over the client table (`sanitise_legal.monitor.pool`) |
| API paths | `/v1/...` | `/legal/v1/...`, the same paths |
| Reading files (step 0: PDF, scans, photos) | `/v1/ingest` | `/legal/v1/ingest`: the same reader, then the legal agents |

Choosing Healthcare runs exactly the code that was on `main` before the legal pipeline came in;
nothing in `src/sanitise`'s agents, prompts, monitor or scoring changed.

## How the legal code got here

The two pipelines grew apart from the first commit, so a line-by-line merge would have mixed
them inside the same files (62 of them conflicted). Instead `tools/import_legal.py` copies the
legal branch into its own package:

* `src/sanitise` on the legal branch → `src/sanitise_legal` here. Only the package name changes
  (`sanitise` → `sanitise_legal` in imports, `-m` commands and module strings).
* `data/ontology` → `data/legal/ontology`. Main keeps its own tables.
* `data/legal`, `config/domains.json`, `tests/legal`, `tools/legal`, `docs/legal`,
  `examples/legal_inputs` and the two legal notebooks come along with the same renames. The
  branch's shared test fixtures become `tests/legal/conftest.py`.
* One small patch, listed in the script, so the legal loader skips main's Target Info query
  list in `data/attacks`.

`src/sanitise_legal/IMPORTED_FROM` records the branch and commit. To take newer legal work:

```bash
git fetch origin
python tools/import_legal.py --ref origin/law-pipeline-final
python -m pytest tests/legal tests/integration/test_legal_mount.py -q
```

`src/sanitise_legal` is replaced as a whole on every import. Make legal changes on the legal
branch, or here and copy them back, never in both places.

## How the app serves both

* `sanitise.api.app.create_app` builds the legal pipeline's own FastAPI app
  (`sanitise_legal.api.app`, always on its legal profile) and mounts it at `/legal`. It also
  runs that app's start-up and shut-down.
* Two paths are added in front of the mount, because the legal API lacks them:
  * `POST /legal/v1/ingest` (file reading, the same for every domain);
  * `GET /legal/v1/sessions/{id}/monitor`, which answers `monitor: "pool"`: the legal monitor's
    verdicts come with every answer, and it keeps no ledger of kinds of detail.
* `SANITISE_LEGAL_ENABLED=false` serves healthcare only. If `sanitise_legal` cannot be imported,
  the API logs it and serves healthcare only.
* **Dashboard:** `web/src/api/client.ts` sends legal requests to `/legal/...` (and every request
  still carries `?domain=`). Switching the toggle remounts the pages, so each reloads its
  documents and attacks from the chosen pipeline. The legal attack kinds (`direct`, `linkage`)
  have their own cards. The leakage chart leaves out "rewriting alone" when the API does not
  report it (legal reports two set-ups).

## Running legal for real

Mock mode (`SANITISE_LLM_BACKEND=mock`) serves the legal demo documents with no model. For
real answers the legal store has to be built once, with the legal profile's models:

```bash
python -m sanitise_legal.legal generate --data data/legal   # only if data/legal is missing
python -m sanitise_legal.legal build --data data/legal --store data/legal/store.sqlite
```

On Kaggle, the legal notebooks (`notebooks/legal_kaggle.ipynb`,
`notebooks/legal_agents_kaggle.ipynb`) install the extras (`pip install -e ".[legal]"`, plus
`".[finetune]"` for Part 3), pull `gemma4:e4b` and run these commands.

## Evaluation

`tools/eval_suite` runs legal on whichever code it is given. On the legal branch (`--repo` = a
clone of it) it uses that branch's `sanitise`; on a repo with `src/sanitise_legal` it uses that
package (`tools/eval_suite/domains.py`, `pkg`). The metrics and results are the same either way.

## Not unified (yet)

* Two monitors: the ledger one (healthcare) and the candidate-pool one (legal).
* Two scorers with different field lists (`sanitise.evaluation.attacks`,
  `sanitise_legal.evaluation.attacks`).
* `src/sanitise_legal` still contains the legal branch's own (older) copy of the medical
  profile. Only its legal profile is used here.

Bringing these together is follow-up work; the toggle and the two pipelines do not depend on it.
