# TRIDENT Phase 2 Submission

Phase 2 uses separate Codabench competitions for the Image, Video, and Audio tracks.
The JSONL schema is unchanged from the starter kit.

## What To Upload

For each Codabench modality track, upload one ZIP containing these four files:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

Each file should contain only IDs for the track modality you are submitting to.
For example, the Image Track submission should contain image IDs only.

Required fields:

- `typeb_oeq.jsonl`: `sample_id`, `response`
- `typea_oeq.jsonl`: `sample_id`, `response`
- `mcq.jsonl`: `question_id`, `response`
- `tfq.jsonl`: `question_id`, `response`

The `response` field must be a string.

## Track-Specific Validation

Validate a track-specific package from the starter-kit root:

```bash
python3 validate_submission.py --submission sample_submission_image --modality image
python3 validate_submission.py --submission sample_submission_video --modality video
python3 validate_submission.py --submission sample_submission_audio --modality audio
```

If you generated an all-modality submission with an earlier starter kit, it can
still be accepted by a modality track validator. IDs from other modalities are
ignored for that track and reported as warnings.

## Upload Notes

Zip the four JSONL files at the root of the archive. Do not zip a parent folder
around them.

```bash
cd sample_submission_image
zip -r ../trident_image_submission.zip \
  typeb_oeq.jsonl \
  typea_oeq.jsonl \
  mcq.jsonl \
  tfq.jsonl
```

Codabench performs format validation at upload time. Official private-test
scores are computed by the organizers after submission.
