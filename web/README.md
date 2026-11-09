# sanitise dashboard

A Next.js page that shows Standard RAG and Protected RAG side by side. It talks only to the
HTTP API (`docs/phase3_plan.md`, section 6.4).

| Page | What it shows |
|---|---|
| Overview (`/`) | The two pipelines as a flowchart, with a sample record's details on each side. |
| Attacks (`/attacks`) | Two chats side by side, one per pipeline (`POST /v1/query`). Picking context dump or stepwise inference fills the attack's prompt into both chats; after each reply the next prompt is filled in. Replies are typed out as they arrive, with leaked details marked red, tokens purple and generalised phrases amber, a running count of leaked details per chat, and the session monitor's verdict on protected replies. Payload interception opens the exact request each pipeline sends to the model (Ollama `POST /api/chat`), side by side, for each of the attack's questions. **Session monitor:** each protected reply's badge shows the monitor's verdict (allowed, generalised, blocked) and how many records still match; a panel under the chats lists, per question, the kinds of detail the session now knows and the candidate pool against k (`GET /v1/sessions/{id}/monitor`); *Score this attack* plays the picked script and charts the Secret Recovery Rate per round for standard RAG, rewriting alone and rewriting + monitor (`POST /v1/attacks/{id}/run`). |
| Documents (`/documents`) | A stored record, or one you upload or paste, through each agent: Agent 1's detected details in the original, Agent 2's facts as points per person, and Agent 3's rewrite. Hovering a highlight rings its twin. A chosen file (PDF typed or scanned, photo or scan of a report or prescription, health record `.hl7` / `.fhir` / `.xml`) is read first (`POST /v1/ingest`) and shown as **step 0, read the file**: the file itself (image, PDF viewer or the raw record) next to the text read from each page, how each page was read (text layer, vision model, health record), the seconds taken, the repeated header/footer lines removed and any warnings. Below it is the text the agents will get, editable, part by part for a long file, and the button that runs them. The agents then run one at a time (`/v1/detect`, `/v1/extract`, `/v1/rewrite`), each panel filling in as its agent finishes. |

**Use case toggle** (top right): Healthcare, Legal, Finance. Every API request carries the chosen
one as `?domain=medical|legal|finance` (the same way as the legal branch), and the choice is
remembered in the browser. A backend with one domain ignores it. Only Healthcare is enabled for
now: Legal is on its own branch and Finance is not built yet, so both are greyed out with the
reason on hover. `src/lib/domains.ts` lists the domains (label, colour, upload hint, file types).

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

In mock mode a typed PDF and a health record are read for real (no model is needed for
them); a scanned page or photo comes back as a placeholder, and the agents' results are samples.
The API address comes from `web/.env.local` (`NEXT_PUBLIC_API_URL`). An environment variable
wins over that file, which helps to try the local API without editing it:

```powershell
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"; npm run dev
```

To use the real models (vision model for scans and photos, Gemini or Ollama for the agents), run
`notebooks/backend_kaggle.ipynb` on Kaggle (`docs/kaggle_setup.md`; `OCR = True` in cell 1) or
`notebooks/backend.ipynb` on Colab up to the serve cell, copy the address it prints, and put it in
`web/.env.local`:

```
NEXT_PUBLIC_API_URL=https://<your-domain>.ngrok-free.app
```

then restart `npm run dev`. The API allows requests from `localhost:3000` by default. With
ngrok set up (see the notebook's first cell) the address is the same in every session; with
the Cloudflare fallback (`https://<printed-address>.trycloudflare.com`) it changes each time.
On Windows PowerShell, write the file with `-Encoding ascii` (other encodings add a byte-order
mark that hides the variable).

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
