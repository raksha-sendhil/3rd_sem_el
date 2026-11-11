# Running the agents notebook

`notebooks/legal_agents_kaggle.ipynb` runs only Agents 1-3: detect, extract facts, rewrite and
leak-check. It runs them on real legal documents and on any documents you give it, and shows
each agent's output. There are two document sets:

- **`indian300`** (the default): 300 Indian judgments, 90 from the Supreme Court and 210 from
  nine High Courts, from the courts' open datasets. Their checks come from the courts' own case
  metadata. See `data/legal/real/indian300/README.md`.
- **`curated`**: 14 hand-picked documents (12 Indian, 1 UK, 1 US) with checks written by hand.
  See `data/legal/real/README.md`. There is no store, search, answering model, attacks, session
monitor, fine-tuning or dashboard. Those are in `notebooks/legal_kaggle.ipynb`, and the session
monitor is a teammate's part.

## 1. Open it on Kaggle

1. Go to <https://www.kaggle.com/code>, then **New Notebook**, then **File**, **Import Notebook**.
   Choose the **GitHub** tab and paste
   `https://github.com/raksha-sendhil/3rd_sem_el/blob/law-pipeline-final/notebooks/legal_agents_kaggle.ipynb`.
   You can also download the `.ipynb` from GitHub and upload it under the **File** tab.
2. In the right-hand panel, **Session options**:
   - **Accelerator**: GPU T4 x2. Only one GPU is used.
   - **Internet**: on. It needs to clone the code and download Gemma 4. Kaggle asks for a
     verified phone number the first time.
   - **Persistence**: Files only. Finished documents are then kept between sessions.
3. Gemini only (see step 2): **Add-ons**, **Secrets**, add `GEMINI_API_KEY` and tick it for
   this notebook. Keys come from <https://aistudio.google.com/apikey>. Never paste a key into a
   cell.

## 2. Choose the model (cell 1)

| `BACKEND` | Runs on | Your text goes to | Speed | Use it for |
|---|---|---|---|---|
| `"ollama"` (default) | Gemma 4 E4B on the Kaggle GPU | nowhere; it stays in the notebook | ~1-2 min per part | the real result; private documents |
| `"gemini"` | Google's Gemini API | Google | fast, but limited to about 10 calls a minute on the free tier | quick checks on public documents |
| `"mock"` | rules only, no model | nowhere | seconds | the baseline; checking the setup |

The other settings in cell 1:

- `DOC_SET`: `"indian300"` (default), `"curated"`, or `"both"` (all 314).
- `LIMIT`: 150 by default, which takes 15 documents from each court in about 2.5-3.5 hours
  with Gemma on a T4. `None` runs all 300, which takes about 5-7 hours with Gemma on a T4.
  That fits in one Kaggle session (12 hours at most), or in two, because finished documents are
  kept. Set it to a number such as 30 for a quick trial, which takes documents in turn from each
  court.
- `MODEL`: another model. Any Ollama tag works with `"ollama"` (cell 3 pulls it), and for
  example `"gemini-3.8-flash"` works with `"gemini"`.
- `MAX_CHARS`: the size of a part. 1,800 is the default and the maximum, because Agent 1
  refuses text longer than 2,000 characters. Lower it if the model's output often breaks (the
  **fallbacks** column).
- `ONLY`: run a few documents. For example `["in08_kar_hc_motor_accident"]`.
- `RUN_REAL_DOCS = False` skips the set and runs only your own documents.

## 3. Run it

Click **Run All**, or run the cells one by one from the top.

| Cell | What happens | What you should see |
|---|---|---|
| 2 | clones `law-pipeline-final`, installs the packages and the spaCy model | `code version: <hash> ...` |
| 3 | installs Ollama, starts it and pulls `gemma4:e4b` (about 10 min the first time) | `gemma4:e4b` in the `ollama list` table |
| 4 | loads the set | "300 documents in the set; running 150", documents per court, the first rows |
| Baseline | the rules alone, with no model | totals: on all 300, about 51% of the must-hide identifiers hidden and 80% of the legal terms kept |
| Agents 1-3 | the documents with the model | one line per document, then the totals side by side and a table per document |
| 5 | one document in full | the coloured spans, the safe text, the spot checks, the facts and the routing |
| Worst 10 | the documents with the most leaks, in full | where the agents still fail |
| Your documents | see section 4 | the same table and views for your files |
| Save | `REPORT.md` and `agent_runs.zip` | the report's start |

If the session stops or times out, run the agents cell again. Each finished document is saved
as `/kaggle/working/agent_runs/<backend>/<id>.json` and is loaded instead of run again.

## 4. Your own documents

Supported formats are `.txt`, `.md`, `.pdf` (with a text layer), `.docx`, and `.json` with a
`text` field. There are three ways to load them.

1. **Many files: a Kaggle dataset.** In the right-hand panel, click **Add Input**, then
   **Upload**. Drop the files or a zip, name the dataset (for example `my-legal-docs`) and click
   **Create**. Then in the "Your own documents" cell set:
   ```python
   MY_INPUTS = ["/kaggle/input/my-legal-docs"]
   ```
   Folders are searched recursively.
2. **A few files: upload into the notebook.** Run `!mkdir -p /kaggle/working/my_inputs`, then
   upload the files into that folder from the **Output** panel. For a public link, run
   `!wget -P /kaggle/working/my_inputs <url>`. Then set
   `MY_INPUTS = ["/kaggle/working/my_inputs"]`.
3. **Paste text** between the triple quotes of `PASTED` in the same cell.

To get pass/fail spot checks for your own document, save it as `.json` with the lists:

```json
{
  "title": "Notice to tenant",
  "text": "....",
  "must_hide": ["Ravi Kumar", "98450 12345", "Flat 3B"],
  "must_keep": ["eviction", "notice period"]
}
```

`examples/legal_inputs/` has a fictional `.txt` and `.json` to copy. They are the default
`MY_INPUTS`; remove that line once you have your own.

Running locally instead of on Kaggle, from the repository root (the `mock` backend needs no
model, and `ollama` needs Ollama running with `gemma4:e4b` pulled):

```python
import asyncio
from sanitise.legal.real_docs import load_manifest, load_inputs, run_agents, write_report
docs = load_manifest() + load_inputs("path/to/my/folder")
results = asyncio.run(run_agents(docs, backend="ollama", out_dir="agent_runs"))
write_report(results, "agent_runs/REPORT.md")
```

## 5. Reading the results

The totals compare the rules alone with the full agents:

- **must-hide identifiers hidden**: how many of the listed identifiers are gone from the safe
  text. The target is 100%. Every one still there is named in the **leaked** column.
- **must-keep terms kept**: how many of the listed legal terms survived. A **dropped** term
  means the rewrite lost substance.
- **direct identifiers left**: names, phones, emails, IDs, case numbers, accounts and addresses
  that Agent 1 found but Agent 3 left in. This works for documents without spot checks too.
- **parts with a fallback**: parts where the model's output was broken and a rule-based step
  took over:
  - `agent1_rules_only`: Agent 1 used only its rules.
  - `agent2_rules_only`: Agent 2 made one fact per span.
  - `agent3_in_place`: Agent 3 replaced spans in the original wording.

In the full view of a document:

- **Original, with Agent 1's spans**: each detected span is coloured by category. Hover over a
  span to see whether a rule or the model found it. Anything personal that is not coloured was
  missed by Agent 1.
- **Agent 2: facts**: one row per fact, as subject, relation, importance and value.
- **Agent 3: what happened to each span**: one row per span, showing whether it was kept,
  generalised (for example "a district in Karnataka"), replaced by a placeholder (`[PERSON_1]`)
  or removed.
- **Agent 3's safe text**: the only text an answering model would ever see.

What to look for on real text:

- **Indian cause titles in capitals** ("SHIVAPPA @ SHIVAKUMAR S/O HUCHAPPA, AGED ABOUT 40
  YEARS"). The rules alone miss most of these, so the model pass has to catch them.
- **Over-redaction**. Judges' and advocates' names, statutes and courts are public. Hiding them
  costs readability but leaks nothing.
- **Splitting**. A long document is split into parts, and each part gets its own
  placeholders, so `[PERSON_1]` in part 1 and part 2 may be different people.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ollama pull` is slow or fails | Check that Internet is on, then run cell 3 again. The download resumes. |
| `LLMUnreachable` on every document | Ollama is not running. Run cell 3 again; `/kaggle/working/ollama.log` shows why. |
| Many `agent3_in_place` fallbacks | Lower `MAX_CHARS` (for example to 1200), or try `MODEL = "gemma4:e4b"` if you changed it. |
| `No key` with `gemini` | Add the `GEMINI_API_KEY` secret and tick it for this notebook, then run cell 3 again. |
| Gemini `429` or quota errors | Wait a minute and run the cell again. Replies already received come from the cache. |
| A PDF gives "no text layer" | It is a scan. Run OCR first (for example `ocrmypdf in.pdf out.pdf`) or paste the text. |
| Changed the code on GitHub | Run cell 2 again (it pulls), then **Run**, **Restart & clear cell outputs**, and run from the top. |
