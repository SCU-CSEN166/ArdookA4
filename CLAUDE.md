# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SCU CSEN 166 term team project (Team A4) for Ardook: a system that takes grant listings and tells Ardook which ones it is eligible for and how good a fit each is. Four team members work in this repo.

**Scope is finding grant leads, not applying.** Don't model application logistics (fees, bank accounts, registrations, travel). Only exclude a grant when a confirmed fact breaks a hard requirement. When the deciding fact is unknown, keep the grant as a conditional lead that names the fact to check (e.g. "Strong if nonprofit").

**Ardook facts live in `docs/ardook-profile.md`.** It's the source of truth for eligibility rules and the org profile; don't hard-code facts about Ardook elsewhere, and treat anything it marks ❓ as unknown. In short: a revenue-generating laundry social enterprise in Yerevan, Armenia (established 2020, launched 2022), staffed by home-working mothers of children with disabilities, and expanding to Tbilisi, Georgia. **Scope is Armenia and Georgia only:** grants limited to other countries (e.g. US-only) are not leads. Ardook doesn't build AI. The spreadsheet's "Ardook Data" (an AI venture) is unresolved and listed in the profile's Open items, along with Ardook's legal form.

**Current state:** planning stage. There is no source code yet, only `README.md`, the grant spreadsheet, and CI. The README's planned layout is `requirements.txt`, `src/`, `tests/`, `evaluation/`, `examples/`. The next deliverable (Week 6) is a runnable baseline.

## README.md is a graded deliverable

`README.md` is the single, term-long project report, not ordinary docs. It is organized by week (Week 5 plan → Week 9 final), and each week has a fixed deliverable and per-member contribution tables that are graded.
- Fill in the existing template sections and `[placeholder]` fields; don't restructure or remove headings.
- When code is added, keep the "Setup and Reproducibility" section (requirements, install, run, evaluation commands) in sync with it.
- Per the README: no personal info, private WhatsApp exports, credentials, or restricted grant/application records in the repo; README examples must be synthetic or anonymized.
- All four members edit this one file, so it is the main merge-conflict hotspot.

## Planned pipeline (README §2)

1. **Ingestion / metadata extraction:** parse grant listings (JSON/CSV/scrapes) into geographic constraints, legal-entity requirements (501(c)(3), foreign nonprofit, for-profit), focus areas, deadline, award range.
2. **Rule-based disqualification filter:** hard constraints against Ardook's profile (e.g. drop US-only 501(c)(3) or lab-based STEM grants; the README notes NSF/NIH grants often fail on US-only or STEM-only requirements).
3. **Dense retrieval:** `sentence-transformers/all-MiniLM-L6-v2` (alt: `intfloat/multilingual-e5-small`) embeds Ardook's profile and grant summaries; top-K retrieval.
4. **Rerank + fit tiering:** `BAAI/bge-reranker-base` cross-encoder scores (profile, grant) pairs into High / Moderate / Weak fit. `facebook/bart-large-mnli` is a candidate for focus-area tagging.
5. **Grounded rationale:** `Qwen/Qwen2.5-0.5B-Instruct` writes an eligibility summary that must not invent unstated grant requirements.

Baselines to compare against: BM25 (`rank_bm25`) keyword retrieval and a keyword-overlap threshold classifier. Planned fallbacks: hybrid BM25 + dense with reciprocal rank fusion, or sequential boolean eligibility gates if the cross-encoder can't handle multi-factor rules. Evaluation: precision, recall and F1 against human judgments (two members cross-check labels).

## Data: the grant spreadsheet

`Copy of Ardook_Data_All_Grant_Recommendations_UPDATED_v3.xlsx.xlsm` (the filename has spaces, so quote it) is a snapshot. The README links the live Google Sheet as the source of record. It's a macro-enabled workbook, so read it with `openpyxl` / `pandas.read_excel(..., engine="openpyxl")`.
- **`Grant Tracker` sheet** (~37 grants, labeled by Ardook staff; the planned dev set): `Priority / Deadline Order`, `Applied?`, `Program`, `Type`, `Amount / Benefit`, `Deadline`, `Current Status (as last reported)`, `Fit`, `Core Requirements`, `Why It Fits Ardook Data`, `Key Concern / Disqualifier`, `Recommended Action`, `Official Source`.
- **`Summary` sheet:** derived counts; not training data.
- **`Fit` is not a clean label.** Base tiers are `Excellent` / `Strong` / `Moderate`, but there are also ranges that use an en dash (`Moderate–Strong`, `Weak–Moderate`) and conditional free text (`Excellent only if nonprofit`, `Strong if deep-tech criteria met`, …). The README calls both out as expected failure cases, so mapping these to tiers is an explicit design decision.
- `Applied?` takes the values `Yes` / `No` / `Not interested`. `Type` is free text (grant, prize, accelerator, credits, …).
- The dataset is tiny and grows by hand. The README plans an 80/10/10 train/test/validation split and dropping closed grants.

## CI (`.github/workflows/`)

`ci.yml` runs on PRs to `main` (against the PR merged into current `main`) and on pushes to `main`. To reproduce it locally:

```bash
pip install -r requirements.txt pytest ruff==0.17.1   # CI uses Python 3.12
ruff check --select E9,F63,F7,F82 .                   # hard errors only (syntax, undefined names), not style
pytest -q                                             # all tests
pytest tests/test_foo.py::test_bar                    # a single test
```

- CI installs PyTorch from the CPU-only index (`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`); runners have no GPU.
- pytest's "no tests collected" (exit 5) is treated as a pass.
- The hygiene job fails on: secrets anywhere in git history (gitleaks, full history), leftover `<<<<<<<` / `>>>>>>>` conflict markers (`=======` is deliberately not checked because Markdown uses it for headings), and any file over 10 MB. Datasets and model weights must stay out of git.

`merge-conflicts.yml` runs whenever `main` changes. It labels open PRs that now conflict with `main` as `merge-conflict` and comments once. `pull_request` workflows (including CI) don't run while a PR has conflicts.

The repo is private on GitHub's free org plan, so branch protection / required checks are unavailable. CI results are advisory, and the team relies on convention not to merge failing PRs.
