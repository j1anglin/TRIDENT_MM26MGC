# Audio Track Sample Submission

This template targets `private_test` and contains IDs for the TRIDENT 2026 Audio
Track only.

Required files:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

Each line is a JSON object with the required id field and a string `response`.
Zip the four JSONL files at the root level when uploading to Codabench.

```bash
zip -r ../TeamAlpha_audio.zip typeb_oeq.jsonl typea_oeq.jsonl mcq.jsonl tfq.jsonl
```
