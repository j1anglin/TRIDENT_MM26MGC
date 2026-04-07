# Sample Submission

This directory contains starter submission templates generated from the current `private_test` identifiers in the blind data package.

Files:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

Schema:

- `typeb_oeq.jsonl`: `{"sample_id":"...","response":""}`
- `typea_oeq.jsonl`: `{"sample_id":"...","response":""}`
- `mcq.jsonl`: `{"question_id":"...","response":""}`
- `tfq.jsonl`: `{"question_id":"...","response":""}`

Canonical `response` formats for the current starter-kit tools:

- `typeb_oeq.jsonl`: first non-empty line should contain `Likely Authentic` or `Likely Manipulated`, followed by a short reasoning paragraph. Example: `{"sample_id":"...","response":"Likely Manipulated\nThe face region shows temporal inconsistency and boundary artifacts."}`
- `typea_oeq.jsonl`: free-form artifact description text. Example: `{"sample_id":"...","response":"Observable artifacts include mouth-boundary mismatch, inconsistent teeth rendering, and temporal flicker around the cheeks."}`
- `mcq.jsonl`: selected option letters as a string. Example: `{"question_id":"...","response":"A, C"}`
- `tfq.jsonl`: `True` or `False` as a string. Example: `{"question_id":"...","response":"True"}`

Notes:

- These templates are intended for phase 2 `private_test` packaging and format validation.
- These flat JSONL files are not the same as the richer `starter_kit_outputs/<task>/<model>/<split>/*.jsonl` records used by `evaluate_predictions.py`.
- During earlier phases, use `evaluate_predictions.py` on `starter_kit_outputs/` for local testing and scoring on the public splits.

Validate from the starter-kit root with:

```bash
python3 validate_submission.py --submission sample_submission
```
