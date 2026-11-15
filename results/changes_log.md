# Changes Attempted After Stress-Testing (Week 8)

Changes made after the hidden test, with before/after numbers from the same code and the same saved model answers. Hidden-test numbers after the first change were measured **after** seeing the TA's labels, so they are not blind results; the official hidden-test result is the submitted one (accuracy 0.40, `results/hidden_test_report.md`).

| # | Change | Worked? | Evidence | Keep / revise / revert |
|---|---|---|---|---|
| 1 | **Ardook Data profile** (`docs/ardook-data-profile.md`, from the sponsor's spreadsheet) and a `--profile` switch, because the TA's labels judge Ardook Data, not Ardook | Yes | Hidden test, same recipe: accuracy 0.40 → 0.80, F1 0.40 → 0.83. Dev for Ardook Data (sponsor's 28 rows): Fit accuracy 0.43–0.50 | Keep. Ask the TA/sponsor which organization the project targets |
| 2 | **Country abbreviations** ("U.S.", "USA", "US" match "United States") | Yes (a correctness fix) | Hidden, Ardook Data: 0.90 → 0.80. The 0.90 relied on the bug: Tory Burch was ruled out as "U.S. only" for a US-based venture | Keep |
| 3 | **New "limits" questions:** regions within a country, minimum and maximum years in operation, minimum and maximum revenue, months converted to years | Partly | With Qwen3-4B, for Ardook: Women's Fund Armenia now ruled out ("limited to Aragatsotn"), D-Prize ruled out for the right reason ("under 1.5 years", was "18 years"), IOM ELEVATE conditional on Gyumri/Shirak, Chicago Foundation limited to Chicago, Entrepenista no longer wrongly excluded as "under 1 year". Ardook dev F1 0.387 → 0.377 (one grant, Dream Makers, moved from Low to Moderate). TA hidden score unchanged (0.70): Armenia-only grants were already excluded for Ardook Data by country | Keep with Qwen3-4B or larger. **Revise for NuExtract-2B:** it reads "founded within 10 years" as a 10-year *minimum* and "6 months" as 6 years (its scores didn't change) |

## Still open

- **Qwen3-32B hasn't answered the new "limits" questions.** That needs the Akash GPU again, about $1–2. Until then, the hybrid recipe (32B eligibility + 4B fit) runs without them.
- **Revenue minimums only add a condition, they don't exclude.** Ardook and Ardook Data revenues are unknown (profile open items), so Tory Burch's $75K minimum becomes "Still to confirm", not "Can't apply".
- **A province can match where a grant actually lists specific towns.** IOM ELEVATE lists towns in Shirak, but not Gyumri, yet it's treated as possible for Gyumri.

## How to reproduce

```bash
python src/run_pipeline.py --input hiddenTest/Ardook_hidden_test_grants.xlsx --backend llm --rules-only              # Ardook
python src/run_pipeline.py --input hiddenTest/Ardook_hidden_test_grants.xlsx --backend llm --profile ardook-data --rules-only
python src/run_pipeline.py --input data/real_grants.csv --labeled-only --backend llm --rules-only
python evaluation/evaluate.py
```
