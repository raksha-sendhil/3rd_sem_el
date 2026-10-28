# sanitise dashboard

A Next.js page that shows Standard RAG and Protected RAG side by side. It talks only to the
HTTP API (`docs/phase3_plan.md`, section 6.4).

| Page | What it shows |
|---|---|
| Overview (`/`) | The two pipelines as a flowchart, with a sample record's details on each side. |
| Attacks (`/attacks`) | Two chats side by side, one per pipeline (`POST /v1/query`). Picking context dump or stepwise inference fills the attack's prompt into both chats; after each reply the next prompt is filled in. Replies are typed out as they arrive, with leaked details marked red, tokens purple and generalised phrases amber, a running count of leaked details per chat, and the session monitor's verdict on protected replies. Payload interception opens the exact request each pipeline sends to the model (Ollama `POST /api/chat`), side by side, for each of the attack's questions. |
| Documents (`/documents`) | A stored record, or one you upload or paste, through each agent: Agent 1's detected details in the original, Agent 2's facts as points per person, and Agent 3's rewrite. Hovering a highlight rings its twin. An uploaded document runs the agents one at a time (`/v1/detect`, `/v1/extract`, `/v1/rewrite`), each panel filling in as its agent finishes. |

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
agents are blue (Detect), purple (Extract) and amber (Rewrite). Icons come from lucide-react.

Illustrations: put the image files in `public/images/` and set their paths in
`src/lib/images.ts`. Until then each one shows a placeholder.
