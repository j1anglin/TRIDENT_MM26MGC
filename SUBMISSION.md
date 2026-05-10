# TRIDENT 2026 Final Test Phase Submissions

The Final Test Phase is for official test-set submissions to the TRIDENT 2026
Image, Video, and Audio tracks.

Participants must run inference on the released test set and submit their
predictions through the corresponding Codabench track page.

## Submission Limit

Each team is allowed a maximum of 3 submissions per day per track.

## Tracks

TRIDENT uses separate Codabench competitions and leaderboards for each modality:

- Image Track: use `sample_submission_image/`
- Video Track: use `sample_submission_video/`
- Audio Track: use `sample_submission_audio/`

Each file should contain only IDs for the track modality you are submitting to.

## Submission Format

Submissions must be uploaded as a ZIP archive containing exactly the following
four JSONL files at the root level:

- `typeb_oeq.jsonl`
- `typea_oeq.jsonl`
- `mcq.jsonl`
- `tfq.jsonl`

Recommended ZIP structure for the Audio Track:

```text
TeamAlpha_audio.zip
|-- typeb_oeq.jsonl
|-- typea_oeq.jsonl
|-- mcq.jsonl
`-- tfq.jsonl
```

Do not submit a single JSON file, and do not place the required JSONL files
inside an extra folder.

Each JSONL file should contain one valid JSON object per line. Empty lines are
ignored.

For `typeb_oeq.jsonl` and `typea_oeq.jsonl`, each record must contain:

- `sample_id`
- `response`

For `mcq.jsonl` and `tfq.jsonl`, each record must contain:

- `question_id`
- `response`

The `response` field must be a string, and IDs must not be duplicated within
the same file.

## Local Validation

Validate a track-specific package from the starter-kit root:

```bash
python3 validate_submission.py --submission sample_submission_image --modality image
python3 validate_submission.py --submission sample_submission_video --modality video
python3 validate_submission.py --submission sample_submission_audio --modality audio
```

If you generated an all-modality submission with an earlier starter kit, it can
still be accepted by a modality track validator. IDs from other modalities are
ignored for that track and reported as warnings.

## Creating A ZIP

Zip the four JSONL files at the root of the archive. Do not zip a parent folder
around them.

```bash
cd sample_submission_audio
zip -r ../TeamAlpha_audio.zip \
  typeb_oeq.jsonl \
  typea_oeq.jsonl \
  mcq.jsonl \
  tfq.jsonl
```

## Evaluation

Codabench is used for submission collection and automatic format validation.

Official final scores may be computed offline by the organizers using the
frozen official evaluation pipeline. The organizers may manually update the
Codabench leaderboard based on the official evaluation results.

The official ranking metric is the Tri-Metric Composite Score:

```text
TCS = 0.4 * Detection + 0.3 * Hallucination Robustness + 0.3 * Perception
```

Invalid submissions will not appear as valid on the leaderboard and may still
count toward the daily submission limit.

## Deadline

The result submission deadline is June 10, 2026, 11:59 p.m. Anywhere on Earth
(AoE).

Late submissions will not be considered for the official final ranking.

## Contact

For questions, please contact:

`trident.at.mm26.mgc@gmail.com`
