# sanitise dashboard

A Next.js page that shows Standard RAG and Protected RAG side by side. It talks only to the
HTTP API (`docs/phase3_plan.md`, section 6.4).

| View | What it shows |
|---|---|
| Compare | One question sent to both pipelines: the answers (personal details leaked by Standard RAG marked in red, placeholder tokens in yellow), the records used, the time taken, the session monitor's verdict, and the exact text sent to the model. Later questions in the session collect in a table. |
| Attacks | The three attacks, one per attacker capability: context dump (one prompt injection), stepwise inference (several innocent prompts) and payload interception (a passive observer of the traffic to the model). Rounds stream in as they finish, with the share of the target's details recovered, the people still matching, and a per-round table. |
| Documents | Each stored record, original next to sanitised, with every change marked, the reason for it, and the extracted facts. |

## Run it

Needs Node.js 20 or newer.

```powershell
# Windows PowerShell, from the repo root, with .venv activated
$env:SANITISE_LLM_BACKEND = "mock"; sanitise serve
```

```bash
# 1. the API, in mock mode (no model needed) from the repo root (macOS/Linux)
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

Pixel windows on warm paper: Pixelify Sans for titles, Silkscreen for labels and buttons,
VT323 for the raw text sent to the model, and Bricolage Grotesque for reading. Each panel is a
window with a coloured title bar. Standard RAG is always red (#f2412f) and Protected RAG always
blue (#2f6bff), in every window, chart and label; the pair was checked for colour-blind
separation.
