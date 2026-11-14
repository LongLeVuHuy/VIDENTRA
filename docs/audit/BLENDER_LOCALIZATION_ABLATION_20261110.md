# Blender localization dev-only ablation — 10/11/2026

## Scope

Exploratory visual-only diagnostic on 12 public English dev queries for Big Buck Bunny. Labels are model-generated/model-adjudicated and not human reviewed. Two-film corpus and model weights are fixed; evaluation only reads qrels after predictions are frozen. No test metric was evaluated for this ablation and no production setting is promoted. The initial pilot test has already been observed and is not a fresh heldout set.

## Same 10-second samples: window width

Using identical 126 embeddings, fixed windows of 4, 6, 8, 10, 12, 16 and 20 seconds give moment Hit@1 tIoU>=0.5 of 1/12, 1/12, 2/12, 4/12, 4/12, 2/12 and 2/12. No width improves hit count over the 10-second baseline. Oracle diagnostic over all 10-second anchors/windows yields 4/12; this is not a production candidate-selection mechanism.

## Sampling density: actual Kaggle GPU run

A separate private Kaggle kernel ran SigLIP 2 on Tesla T4 with interval 2 seconds versus the prior 10 seconds. The same two movies, 130 query texts, fixed 10-second windows, and pinned model revision/weights were used. Sparse/dense runs produced 126/626 visual frame vectors; all dense GPU frame PTS/ordinals were checked against local sequential ledgers. Query embedding maximum absolute difference was 0 across runs. No OCR/ASR/relation or latency was measured.

| DEV-only metric | 10-second samples | 2-second samples |
| --- | ---: | ---: |
| Query count | 12 | 12 |
| Moment Hit@1 at tIoU >= 0.5 | 4/12 | 4/12 |
| Mean top-1 temporal IoU | 0.1950 | 0.4028 |
| Oracle diagnostic at tIoU >= 0.5 | 4/12 | 6/12 |
| Top-1 anchor inside labeled interval | 4/12 | 10/12 |

Paired query-level IoU: 9 improved, 1 worsened, 2 tied. These figures describe one source and 12 provisional model-label queries, not a statistically secure result. Oracle windows are not clipped to video end, so are a diagnostic bound only. Sampling density increased but Hit@1 did not; do not infer a system-level win or change the default.

## Artifacts

- Sparse diagnostic: `work/experiments/blender_dev_localization_diagnostic_20261110_v2.json`
- Width ablation: `work/experiments/blender_dev_fixed_window_ablation_20261110.json`
- Dense GPU output: `work/kaggle/blender_dense_gpu_20261110/videntra_visual_pilot/manifest.json`
- Dense prediction SHA-256: `18edb999066b1ab82e279d5afe9f721768e6b074f3bdd52c079c7cd064997224`
- Dense dev-only diagnostic: `work/experiments/blender_dense_dev_localization_20261110.json`

Remaining: labels reviewed by people, larger independent multi-source corpus, meaningful boundary proposals, OCR/ASR and action/role evidence, full A5 verifier and latency/VRAM on target hardware.