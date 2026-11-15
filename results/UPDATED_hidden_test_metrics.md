# Hidden Test Results

10 grants from the TA. Positive class = **Relevant** (TA label "Good"). A grant is Relevant when it isn't ruled out by a confirmed requirement and its Fit is High or Moderate.

| Version | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Submitted during lab (profile: Ardook, the laundry business) | 0.40 | 0.40 | 0.40 | 0.40 |
| After the fix (profile: Ardook Data), measured after seeing the TA's labels | 0.80 | 0.71 | 1.00 | 0.83 |

## Submitted during lab (profile: Ardook, the laundry business)

| | Predicted Relevant | Predicted Not relevant |
| --- | ---: | ---: |
| **Actually Relevant** | 2 (TP) | 3 (FN) |
| **Actually Not relevant** | 3 (FP) | 2 (TN) |

Wrong: #1 Cisco Global Problem Solver, #2 Women's Fund Armenia, #5 Women Founders Network Fast Pitch, #6 IOM ELEVATE, #9 People in Need (Armenia), #10 Women Who Tech

## After the fix (profile: Ardook Data), measured after seeing the TA's labels

| | Predicted Relevant | Predicted Not relevant |
| --- | ---: | ---: |
| **Actually Relevant** | 5 (TP) | 0 (FN) |
| **Actually Not relevant** | 2 (FP) | 3 (TN) |

Wrong: #3 Tory Burch Fellows, #8 D-Prize

## Notes

- The TA's labels judge **Ardook Data** (a woman-led, early-stage AI/data startup), while the submitted run used the profile of Ardook, the laundry social enterprise. That mismatch caused most of the submitted run's errors.
- The Ardook Data profile (`docs/ardook-data-profile.md`) was built from the sponsor's spreadsheet only, but after the TA's answers were known, so its score is not a blind result. The official result is the submitted one.
- Remaining errors: #3 Tory Burch needs at least $75K in revenue (the checklist doesn't ask about revenue minimums yet); #8 D-Prize was rated a good fit through Ardook's poverty-reduction mission, while the TA judged it not to be AI/data work.
- With 10 grants, each one changes accuracy by 0.10.

Reproduce: `python evaluation/hidden_test.py --eligibility-run outputs/Ardook_hidden_test_grants_api_qwen3-32b_ardook-data --fit-run outputs/Ardook_hidden_test_grants_llm_qwen3-4b_direct_ardook-data --labels hiddenTest/ardook_hidden_test_labels.xlsx`
