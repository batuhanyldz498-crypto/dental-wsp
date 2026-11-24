# GitHub Setup, Private Development & Public Release Workflow

This guide walks through publishing `dental-wsp` cleanly from local setup to a **private** GitHub repository during validation, and switching it to **public** once benchmark results are complete.

---

## 1. Local Verification Before Pushing

From `C:\Users\Lenovo\Desktop\dental-wsp`:

```powershell
python -m pip install -e ".[dev]"
pytest -v
```

Verify repository hygiene (must pass with zero raw medical files, private splits, or credentials):

```powershell
pytest tests/test_repo_hygiene.py -v
git status
```

---

## 2. Re-Authenticate GitHub CLI & Create Private Repo

Because local `gh` authentication for `batuhanyldz498-crypto` needs a fresh login:

```powershell
gh auth login -h github.com
```

Follow the interactive browser prompt, then create the **private** repository and push `main`:

```powershell
cd C:\Users\Lenovo\Desktop\dental-wsp
gh repo create batuhanyldz498-crypto/dental-wsp --private --source=. --remote=origin --push
```

Confirm GitHub Actions CI (`.github/workflows/ci.yml`) turns green under the **Actions** tab.

---

## 3. Running Validation & Collecting Manual WSP Replies

1. Keep raw datasets strictly outside Git (e.g., Google Drive `WSP_Work/` or local `data/`, which is git-ignored).
2. For each new task (`akudental_cls`, `denpar_seg`, `denpar_cls`, `dentvoxel_seg`, `dentvoxel_cls`):
   - Paste `benchmarks/<dataset>/<seg|cls>_prompt.md` into a fresh GPT / Claude chat (no web search, no tools, no image uploads).
   - Save each raw reply text into `inbox/<task>/<family>_<rep>.txt` (git-ignored).
   - Lock the reply into a portable, non-private weight lock:
     ```powershell
     dental-wsp lock-reply <task> --slot gpt_1 --raw-file inbox/<task>/gpt_1.txt --out benchmarks/<dataset>/locks/<task>.responses.lock.json
     ```
   - Commit only the non-private `design.lock.json` and `responses.lock.json` files (never commit patient splits, overlays, or raw datasets).
3. Run validation notebooks in Colab (`notebooks/01_*.ipynb` through `03_*.ipynb`) keeping all radiograph/CBCT overlays in private runtime memory.

---

## 4. Flipping Private → Public After Final Publication

Before switching visibility to **Public**, complete this checklist:
- [ ] All 6 tasks have locked `design.lock.json`, `responses.lock.json`, and finalized non-private summary tables.
- [ ] All notebooks under `notebooks/` have outputs cleared (`nbstripout` or Colab "Clear all outputs").
- [ ] `pytest tests/test_repo_hygiene.py -v` passes.
- [ ] Dataset citations and license notices in `README.md` and `benchmarks/*/DATASET.md` are verified.
- [ ] Switch visibility:
  ```powershell
  gh repo edit batuhanyldz498-crypto/dental-wsp --visibility public --accept-visibility-change-consequences
  ```
