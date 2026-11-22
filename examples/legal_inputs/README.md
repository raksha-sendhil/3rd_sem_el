# Your own inputs for the agents notebook

`notebooks/legal_agents_kaggle.ipynb` reads every supported file in the folders listed in
`MY_INPUTS` (this folder is the default). The two files here are **fictional** and only show the
formats; replace them with your own, or point `MY_INPUTS` at a Kaggle dataset.

| Format | What is read |
|---|---|
| `.txt`, `.md` | the whole file |
| `.pdf` | the text layer (a scanned PDF needs OCR first) |
| `.docx` | the paragraphs |
| `.json` | `text` (required); optional `title`, `must_hide`, `must_keep` |

`must_hide` lists strings that must not appear in Agent 3's safe text (names, phone numbers,
addresses, IDs); `must_keep` lists legal terms that must survive the rewrite. Both are checked
case-insensitively and reported per document. Without them you still get "direct ids left":
identifiers Agent 1 found that Agent 3 did not remove.

Documents over 1,800 characters are split into parts at paragraph breaks and each part goes
through the agents on its own. This README is skipped when the folder is loaded.
