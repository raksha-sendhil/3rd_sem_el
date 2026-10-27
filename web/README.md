# sanitise dashboard

A Next.js page that shows Standard RAG and Protected RAG side by side. It talks only to the
HTTP API (`docs/phase3_plan.md`, section 6.4).

| View | What it shows |
|---|---|
| Compare | One question sent to both pipelines: the answers, the records used, the time taken, and the exact text the model was given. A table below keeps every question asked in the session. |
| Attacks | The scripted attacks in `data/attacks` (or `data/sample/attacks`). Running one plays the rounds back one per second, with charts of details revealed and people still matching, and a table view. |
| Documents | Each stored record, original next to sanitised, with every change marked and the reason for it. |

## Run it

Needs Node.js 20 or newer.

```bash
# 1. the API, in mock mode (no model needed) from the repo root
SANITISE_LLM_BACKEND=mock sanitise serve

# 2. the dashboard
cd web
npm install
npm run dev        # http://localhost:3000
```

To use the real model on Colab, run `notebooks/backend.ipynb` up to cell 4, copy the address
it prints, and put it in `web/.env.local`:

```
NEXT_PUBLIC_API_URL=https://<printed-address>.trycloudflare.com
```

then restart `npm run dev`. The API allows requests from `localhost:3000` by default.

## When the API changes

`src/api/schema.d.ts` holds the API types. After changing the API models, export the
OpenAPI document and regenerate them:

```bash
# from the repo root
python -c "import json; from sanitise.api.app import create_app; from sanitise.config import Settings; print(json.dumps(create_app(Settings(llm_backend='mock')).openapi(), indent=1))" > web/src/api/openapi.json
cd web && npm run types
```

## Design

Built on the IBM Carbon Design System's visual language: IBM Plex Sans and Mono, the
gray-10 layer on white, square corners, 1px rules, blue-60 for anything you can click.
Standard RAG is always purple (#6929c4) and Protected RAG always cyan (#1192e8), in every
chart, swatch and label; the pair was checked for colour-blind separation.
