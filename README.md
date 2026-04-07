# TRIDENT Starter Kit

This folder is for participants who want to:

- run a baseline model on the released splits
- score predictions locally on `public_val`
- validate a final `private_test` submission package

Run all commands below from this `starter_kit/` directory.

## What To Use When

During development:

- Use `run_baseline.py` to generate predictions.
- Use `evaluate_predictions.py` to score those predictions on `train` or `public_val`.
- Look under `starter_kit_outputs/` for the generated records and summary JSON files.

When preparing the final submission:

- Start from `sample_submission/`.
- Fill in the required `response` fields for each task file.
- Use `validate_submission.py` to check that the package is complete and correctly formatted.

Important:

- `evaluate_predictions.py` does not read `sample_submission/*.jsonl`.
- `validate_submission.py` does not score predictions.

## Quick Start

1. Install dependencies:

```bash
bash install.sh
```

2. Check that the package wiring works:

```bash
python3 run_baseline.py --task typeb_oeq --split public_val --dry-run
```

3. Run a public validation baseline:

```bash
python3 run_baseline.py \
  --task all \
  --split public_val \
  --model Qwen/Qwen3-VL-8B-Instruct
```

4. Score the generated outputs:

```bash
python3 evaluate_predictions.py \
  --task all \
  --split public_val \
  --predictions-root starter_kit_outputs \
  --model Qwen/Qwen3-VL-8B-Instruct
```

5. Validate a final submission package:

```bash
python3 validate_submission.py --submission sample_submission
```

By default, the starter kit looks for the blind package under `tridf-blind/`. If your copy lives elsewhere, pass `--data-root /path/to/tridf-blind`.

## Common Workflows

### Local testing on `public_val`

Use this path while developing your system:

```bash
python3 run_baseline.py \
  --task all \
  --split public_val \
  --model microsoft/Phi-4-multimodal-instruct

python3 evaluate_predictions.py \
  --task all \
  --split public_val \
  --predictions-root starter_kit_outputs \
  --model Phi-4-multimodal-instruct
```

Outputs are written under:

```text
starter_kit_outputs/<task>/<model>/<split>/*.jsonl
starter_kit_outputs/evaluation_results/
```

### Final phase-2 submission packaging

Use this path when you are preparing a `private_test` submission:

```bash
python3 validate_submission.py --submission sample_submission
```

Expected final files:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

## Tasks

- `typeb_oeq`: decide whether a sample is authentic or manipulated, then explain the decision
- `typea_oeq`: describe the visible manipulation artifacts in a sample known to be fake
- `mcq`: choose which listed artifacts are present
- `tfq`: answer True or False to an artifact verification question

`typeb_oeq` and `typea_oeq` share the same OEQ artifact-evaluation pipeline.

## Supported Baseline Models

- `Qwen/Qwen3-VL-8B-Instruct`
- `microsoft/Phi-4-multimodal-instruct`

Model notes:

- `Qwen/Qwen3-VL-8B-Instruct` supports image and video only.
- `Phi-4-multimodal-instruct` supports image, video, and audio.

## Submission Format

The final submission package uses flat JSONL files. This is different from the richer baseline-output records under `starter_kit_outputs/`.

Required identifier fields:

- `typeb_oeq.jsonl`: `sample_id`
- `typea_oeq.jsonl`: `sample_id`
- `mcq.jsonl`: `question_id`
- `tfq.jsonl`: `question_id`

Canonical `response` conventions for the current starter-kit tools:

- `typeb_oeq`: first non-empty line contains `Likely Authentic` or `Likely Manipulated`, followed by a short justification paragraph
- `typea_oeq`: structured free-form artifact description text
- `mcq`: selected option letters such as `A, C`
- `tfq`: `True` or `False`

Validator behavior:

- `validate_submission.py` checks required ids and that `response` is a string.
- It warns on empty responses.
- It does not enforce the semantic response format above.

## Local Scoring Notes

- `evaluate_predictions.py` scores the richer records under `starter_kit_outputs/`.
- Per-model summaries are written under `starter_kit_outputs/evaluation_results/`.
- Missing predictions lower the reported metrics.
- OEQ artifact mappings are written under `starter_kit_outputs/evaluation_results/oeq_mappings/` by default.
- Official OEQ evaluation uses the organizer-frozen OpenAI model `gpt-5.4-mini`.
- Local runs default to the bundled local evaluator unless you override `--oeq-evaluator-backend` and `--oeq-evaluator-model`.
- For local OEQ evaluation, `--oeq-evaluator-batch-size` controls the micro-batch size used by the local text evaluator. The default is `10`.

For detailed metric definitions, read [docs/METRICS.md](/project/aimm/trident/starter_kit/docs/METRICS.md).

### Optional OEQ evaluator override

If you want to override the OEQ artifact evaluator backend:

```bash
python3 evaluate_predictions.py \
  --task typea_oeq \
  --split public_val \
  --predictions-root starter_kit_outputs \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --oeq-evaluator-backend openai \
  --oeq-evaluator-model gpt-5.4-mini
```

Useful aliases:

- `--oeq-evaluator-backend openai --oeq-evaluator-model gpt-mini`
- `--oeq-evaluator-backend gemini --oeq-evaluator-model gemini-flash-lite`
- For API-backed evaluators, `--oeq-evaluator-concurrency 10` increases parallel request throughput.

## Files In This Folder

- `run_baseline.py`: baseline inference runner
- `evaluate_predictions.py`: public-split scorer
- `validate_submission.py`: submission-format validator
- `compute_tcs.py`: standalone TCS recomputation helper for evaluation summaries
- `sample_submission/`: submission templates for all four tasks
- `examples/`: ready-to-run shell scripts
- `docs/`: participant-facing rules and metrics
- `starter_kit/`: internal Python package, runtime helpers, and model wrappers

## Read Next

- `docs/README.md`
- `docs/RULES.md`
- `docs/METRICS.md`

## Example Scripts

```bash
bash examples/run_public_val_qwen_vl.sh
bash examples/score_public_val.sh
```

## Contact

For questions regarding this starter kit, please contact:

- Email: `trident.at.mm26.mgc@gmail.com`
