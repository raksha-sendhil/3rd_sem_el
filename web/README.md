# sanitise dashboard

A Next.js page that shows Standard RAG and Protected RAG side by side. It talks only to the
HTTP API (`docs/phase3_plan.md`, section 6.4).

| Page | What it shows |
|---|---|
| Compare (`/`) | One question sent to both pipelines: each answer with leaked details marked red, tokens purple and generalised phrases amber; how many details leaked; the session monitor's verdict on the protected answer; the time taken and records used; and "What the model saw", the exact prompt. Runs collect in a session table with their verdicts. |
| Attacks (`/attacks`) | The three attacks: context dump, stepwise inference and payload interception. Rounds stream from the API and play back at 1x, 2x or 4x, or one step at a time. Charts of the Secret Recovery Rate and the people still matching, a live transcript (what was disclosed, the pool left and the monitor's verdict per round), and the target's details as each pipeline gives them away. |
| Documents (`/documents`) | Each stored record before and after the three agents, with counts per agent. Hovering a highlight rings its twin on the other side and in the table of changes. |

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

White cards on a dotted paper background (#f6f4ef), styled with Tailwind. Bricolage Grotesque
for headings and numbers, Instrument Sans for reading, JetBrains Mono for labels and ids.
Standard RAG is coral (#ff4d3d) and Protected RAG teal (#0fb5a0) in every panel and chart. The
agents are blue (Detect), purple (Extract) and amber (Rewrite). Icons come from lucide-react,
charts from recharts and toasts from sonner.

Illustrations: put the image files in `public/images/` and set their paths in
`src/lib/images.ts`. Until then each one shows a placeholder.
