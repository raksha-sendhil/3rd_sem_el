# Running the backend on Kaggle

The notebook `notebooks/backend_kaggle.ipynb` does what `notebooks/backend.ipynb` (Colab) does:
Ollama with llama3:8b on a GPU, the Gemini API for Agents 1-3, the API, and a public tunnel. The
dashboard (`web/`) still runs on your own laptop and talks to the tunnel address.

## Colab and Kaggle compared

| Need | Colab | Kaggle | Works? |
|---|---|---|---|
| GPU for llama3:8b | T4 | T4 x2 or P100 (one is used) | Yes |
| Secrets (API keys) | key icon, `google.colab.userdata` | Add-ons -> Secrets, `kaggle_secrets` | Yes, different call (already changed) |
| Internet, `apt`, `curl`, `pip` | yes | yes, **after you switch Internet On** (needs a phone-verified account) | Yes |
| Ollama install and `ollama pull` | yes | yes (root shell, same script) | Yes |
| ngrok and Cloudflare tunnels | yes | yes (outbound connections are allowed) | Yes |
| Google Drive checkpoints | yes | **no Drive** | Replaced by `/kaggle/working` + Save Version |
| Form fields (`#@param`) | yes | no, plain variables at the top of each cell | Changed |
| Working folder | `/content` | `/kaggle/working` | Changed |
| Session length | about 12 h, disconnects when idle | up to 12 h, stops after a period of inactivity, about 30 GPU hours a week | Check the quota page |

The notebook was produced from the Colab one by a script and the cells were not run on Kaggle
before this guide was written. The first run is the test: if a cell fails, send the error to the
person who wrote this.

## One-time setup (each person)

1. **Kaggle account** at kaggle.com. Open Settings -> Phone verification and verify. Without it you
   get no internet and no GPU in notebooks.
2. **Gemini key(s)**: aistudio.google.com -> Get API key. Several keys, comma-separated, are used in
   turn when one reaches its daily limit (one per Google Cloud project).
3. **ngrok** (so the dashboard address never changes): ngrok.com -> free account ->
   *Your Authtoken* (copy it) and *Domains* -> claim the free static domain, e.g.
   `name.ngrok-free.dev`. A static domain can only be used by one running notebook at a time, so
   if two people run at once, each needs their own ngrok account and domain. Without ngrok the
   notebook falls back to a Cloudflare quick tunnel (no account, new address each time).

## Run the notebook

1. kaggle.com -> **Create** -> **New Notebook** -> **File -> Import Notebook** and upload
   `notebooks/backend_kaggle.ipynb` from the repo.
2. Right sidebar -> **Session options**:
   - **Accelerator**: GPU T4 x2 (or P100).
   - **Internet**: On.
3. **Add-ons -> Secrets -> Add secret**, three of them, each ticked "attached" for this notebook:

   | Label | Value |
   |---|---|
   | `GEMINI_API_KEY` | your key, or several separated by commas |
   | `NGROK_AUTHTOKEN` | the ngrok authtoken |
   | `NGROK_DOMAIN` | e.g. `name.ngrok-free.dev` (no `https://`) |

4. In cell **1 Setup**, leave `BRANCH = "main"` (it has everything, case files included) and
   check `MODEL = "llama3:8b"` and `AGENTS_BACKEND = "gemini"`. Edit the plain variables at the
   top of the cell instead of using form fields.
5. Run the cells in order (Shift+Enter on each; do not use "Run All" the first time):

   | Cell | Notes |
   |---|---|
   | 1 Setup | clones the repo, installs packages, installs Ollama and downloads llama3:8b (about 5 min) |
   | 2 Evaluate | optional, 20 fixtures |
   | 3 Restore | only when you saved a store earlier (see below) |
   | 4 Build | set `DOCS` (20 for a quick demo, 150 for the metric). Resumes if re-run in the same session |
   | 5 Save | writes `/kaggle/working/sanitise/store.tar.gz` |
   | 6 Serve | starts the API on port 8000 and the tunnel; prints `API address for the dashboard: https://...` |
   | 7 Smoke test | one question through the tunnel; prints BASELINE and PROTECTED answers |
   | 8 Target Info | Metric 1, 30-50 min for 150 documents |

   Leave the notebook tab open while the API is needed. Cell 6 keeps running in the background;
   when the session stops, the address stops answering.

## Keeping the store between sessions

Everything under `/kaggle/working` is lost when the session ends, unless you save a version:

- After cell 4 and 5, click **Save Version** (top right). Quick Save is enough, with the
  Output option on. The files `sanitise/store.tar.gz` and the checkpoints are then stored as the
  notebook's output.
- In a later session, right sidebar -> **Input -> Add Input -> Your Work / Notebook Output Files**
  and add this notebook's output (or a Dataset you made by uploading `store.tar.gz`). Then run
  cell 1, **cell 3 (Restore)**, and cell 6. This is the demo-day route, about 5-8 minutes.
- Other option: Output tab -> download `store.tar.gz`, and later upload it as a private Dataset
  (Datasets -> New Dataset) and attach it as an input.

Cell 3 searches `/kaggle/input` for `store.tar.gz` or `store_<N>.sqlite`, so the exact input path
does not matter.

## Connect the dashboard (on your laptop)

Needs Node.js 20 or newer. From the repo root:

```powershell
cd web
npm install
```

Create `web/.env.local` with the address cell 6 printed. In PowerShell, write it as ASCII
(other encodings add a hidden byte-order mark that hides the variable):

```powershell
"NEXT_PUBLIC_API_URL=https://name.ngrok-free.dev" | Out-File -Encoding ascii .env.local
npm run dev
```

macOS / Linux:

```bash
echo "NEXT_PUBLIC_API_URL=https://name.ngrok-free.dev" > .env.local
npm run dev
```

Open http://localhost:3000. The header shows the API address; if it shows "Cannot reach the API",
check that the notebook session is running and cell 6 printed a tunnel address.

- With ngrok and a static domain the address is the same every session, so `.env.local` is written
  once. With the Cloudflare fallback (`https://<words>.trycloudflare.com`) it changes every time:
  edit `.env.local` and restart `npm run dev` each session.
- The API only accepts browser requests from `localhost:3000` / `127.0.0.1:3000`. To serve the
  dashboard from another address, add it to `api_cors_origins` in `src/sanitise/config.py`.
- Test the tunnel without the dashboard:

  ```bash
  curl -H "ngrok-skip-browser-warning: 1" https://name.ngrok-free.dev/v1/health
  ```

## Problems

| Symptom | Fix |
|---|---|
| `kaggle_secrets` error or "Add the Kaggle secret" | Add-ons -> Secrets, make sure the secret is ticked for this notebook, rerun cell 1 |
| `git clone` or `pip` fails, no internet | Session options -> Internet On (and phone verification) |
| Answers take minutes, `nvidia-smi` empty | Accelerator is None; set GPU and rerun from cell 1 (changing it restarts the session) |
| ngrok "domain already in use" / `ERR_NGROK_334` | Another session still holds the domain: stop it, or use your own ngrok account |
| ngrok fails, Cloudflare used | Normal fallback; new address, update `.env.local` |
| Cloudflare 429 | Retry in a few minutes, or use ngrok |
| Dashboard shows "Cannot reach the API" | Session stopped, wrong address in `.env.local`, or `npm run dev` not restarted after editing it |
| Version conflicts at `pip install -r requirements.txt` | Kaggle ships its own packages; send the error text, it is usually one pin to loosen |
| Session stopped after idle time | Rerun from cell 1, restore the store (cell 3), then cell 6 |
| Gemini daily limit reached | Add more keys (comma-separated) to `GEMINI_API_KEY` |
