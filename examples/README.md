# Examples

These scripts are minimal entry points for local testing.

- `run_public_val_qwen_vl.sh`: run all tasks on `public_val` with `Qwen/Qwen3-VL-8B-Instruct`
- `run_public_val_phi4.sh`: run all tasks on `public_val` with `microsoft/Phi-4-multimodal-instruct`
- `score_public_val.sh`: score the current `starter_kit_outputs/` directory on `public_val`

The scorer accepts `--oeq-evaluator-*` flags when you want to override the OEQ artifact evaluator backend. For API backends, you can also set `OEQ_EVALUATOR_CONCURRENCY=10` to send multiple mapping requests in parallel.

Run from the starter-kit root:

```bash
bash examples/run_public_val_qwen_vl.sh
bash examples/score_public_val.sh
```
