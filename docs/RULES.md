# TRIDENT Challenge Rules

This document summarizes the participant-facing rules that are directly relevant to the released blind package and starter-kit tooling.

Items such as deadlines, submission caps, prizes, or team-eligibility policies may be announced separately by the organizer.

## Tasks

The challenge includes four tasks:

- `tfq`
- `mcq`
- `typea_oeq`
- `typeb_oeq`

## Package Splits

The participant package may include:

- `train`
- `public_val`
- `private_test`
- `starter_kit`
- `sample_submission`

Public answers are included only where the package explicitly provides them. `private_test` answers are intentionally withheld.

## Input Prompts And Submission Protocols

TRIDENT uses a flexible prompting workflow together with a strict output packaging format.

Participants may use the provided blind data, prompt templates, and starter-kit scripts to build their systems. Final predictions must be packaged into standardized JSONL files so they are compatible with the automated evaluation pipeline.

### Structured VQA: `tfq` and `mcq`

These tasks ask automated questions about the existence or location of specific artifacts.

- `tfq`: answer with a binary response string, using `True` or `False` in the starter-kit tooling
- `mcq`: answer with the selected option letters, for example `A, C`

### Type-A Open-Ended Questions: `typea_oeq`

The sample is known to be manipulated. The model should produce a structured text paragraph describing the observable artifacts.

### Type-B Open-Ended Questions: `typeb_oeq`

The sample authenticity is unknown. The model should produce:

- a binary authenticity judgment: `Likely Authentic` or `Likely Manipulated`
- a concise reasoning paragraph supporting that judgment

## Final Submission Format

Final submissions must be UTF-8 JSONL files with one JSON object per line.

Required files:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

Required identifier fields:

- `typeb_oeq.jsonl`: `sample_id`
- `typea_oeq.jsonl`: `sample_id`
- `mcq.jsonl`: `question_id`
- `tfq.jsonl`: `question_id`

Every record must contain a string-valued `response` field.

Use `sample_submission/` as the template for the final `private_test` submission package, and validate it with:

```bash
python3 validate_submission.py --submission sample_submission
```

## Local Evaluation And Official Evaluation

- Use `public_val` for local debugging and development.
- Use `private_test` only for final submission packaging.
- Official leaderboard ranking uses the organizer-frozen scorer, metrics, and evaluator definitions.
- For OEQ evaluation, the organizer-frozen official evaluator uses OpenAI `gpt-5.4-mini`.
- Local starter-kit evaluation is intended for iteration and sanity checking.

## Human Annotation And Private-Test Handling

The following are not allowed for `private_test`:

- manual inspection to derive labels or artifact answers
- manual annotation of private-test samples
- human-in-the-loop answer editing on a sample-by-sample basis

## Disallowed Behavior

Unless the organizer explicitly states otherwise, the following should be treated as disallowed:

- reverse-engineering hidden labels from package artifacts
- exploiting leaked private answers
- using organizer-only files or metadata not distributed in the participant package
- submitting malformed or intentionally corrupted outputs

## Disclosure And Reproducibility

Participants should be prepared to disclose, if requested by the organizer:

- model identifiers and checkpoints
- external datasets or retrieval corpora
- synthetic data sources
- prompting or adaptation procedures
- API-based model usage
- package version used for the submission
- inference code or scripts needed to reproduce the submission
