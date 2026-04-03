# TRIDENT Evaluation Metrics

To evaluate interpretable DeepFake detection systems, TRIDENT uses a tri-perspective protocol covering Perception, Detection, and Hallucination Robustness.

Unless otherwise stated by the organizer, all metrics are computed per sample and macro-averaged over the test set within each modality track.

## Metric Definitions

### I. Perception

Perception measures the model's ability to localize and identify artifacts, evaluated exclusively on manipulated samples.

This dimension is assessed through:

- structured questions: `tfq` and `mcq`
- open-ended generation: `typea_oeq`

For open-ended responses, an LLM-based evaluator parses the generated text and maps it to a discrete reported-artifact set `R_art`, which is then compared against the ground-truth artifact set `Y_art`.

For official evaluation, the organizer-frozen OEQ evaluator uses OpenAI `gpt-5.4-mini`.

#### TFQ Accuracy: `Acc_TFQ`

`Acc_TFQ` is the accuracy on binary verification questions about the presence of artifacts or location cues.

#### MCQ Score: `Score_MCQ`

For a question with:

- correct option set `C`
- `K = |C|`
- total option count `M`
- predicted option set `S`

the score is:

```text
Score_MCQ(S, C) = max(0, sum_{i in S ∩ C}(1 / K) - sum_{i in S \ C}(1 / (M - K)))
```

This balanced scoring rule is designed to reduce the benefit of random guessing.

#### Explanatory Coverage: `Cover`

Using the artifacts extracted by the LLM evaluator, `Cover` measures how much of the ground-truth artifact set is recovered:

```text
Cover(R) = |R_art ∩ Y_art| / max(1, |Y_art|)
```

### II. Detection

Detection measures the holistic ability to distinguish authentic from manipulated media.

This dimension is evaluated on both authentic and manipulated samples using `typeb_oeq`.

#### Detection Accuracy: `Acc_Det`

`Acc_Det` is the standard accuracy of the binary authenticity decision.

### III. Hallucination Robustness

Hallucination Robustness measures whether the explanation remains faithful to the actual artifacts in the sample.

This dimension emphasizes precision of the generated explanation and its trade-off with perception recall.

#### Hallucinated Artifact Rate: `CHAIR`

`CHAIR` measures the proportion of reported artifacts that are incorrect:

```text
CHAIR(R) = 1 - |R_art ∩ Y_art| / |R_art|
```

In practice, this acts as a false-discovery-rate style penalty.

#### Balanced Interpretability Score: `F^0.5`

TRIDENT uses `F^0.5` to prioritize precise, reliable explanations over broad but hallucinated ones.

By treating:

- `1 - CHAIR` as precision
- `Cover` as recall

the metric is:

```text
F^0.5(R) = ((1 + 0.5^2) * (1 - CHAIR(R)) * Cover(R)) / (0.5^2 * (1 - CHAIR(R)) + Cover(R))
```

Empty responses are treated as fully unreliable, implying `CHAIR = 1` and `F^0.5 = 0`.

## Official Ranking: Tri-Metric Composite Score

Final leaderboard ranking uses the Tri-Metric Composite Score (`TCS`).

Three normalized component scores are defined:

```text
S_Det  = 100 * Acc_Det
S_Perc = 100 * (0.5 * Acc_TFQ + 0.5 * Score_MCQ)
S_Hal  = 100 * F^0.5
```

The final composite score is:

```text
TCS = w_Det * S_Det + w_Hal * S_Hal + w_Perc * S_Perc
```

with official weights:

- `w_Det = 0.4`
- `w_Hal = 0.3`
- `w_Perc = 0.3`

This weighting prioritizes:

- accurate authenticity detection first
- reliable, non-hallucinated interpretation second
- fine-grained structured perception third

## Tie-Breaking

If two submissions have the same `TCS`, the tie-breaking order is:

1. higher `S_Det`
2. higher `S_Hal`
3. higher `S_Perc`

## Starter-Kit Summary Field Names

The local starter-kit summaries expose these metric names:

- `typeb_oeq`: `acc_det`
- `typea_oeq`: `cover`, `chair`, `f_0_5`
- `mcq`: `score_mcq`
- `tfq`: `acc_tfq`

The summary JSON written by `evaluate_predictions.py` also reports `tcs_by_modality` for convenience on local runs.
