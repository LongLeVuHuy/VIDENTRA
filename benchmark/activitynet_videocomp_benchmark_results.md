# Kết quả Benchmark ActivityNet Captions & VideoComp (Sealed Test)

- Config SHA-256: `11b95014f57e58efe39d7242f456007857cdf1e65d7a0b1e63b0959afab38d21`
- VLM Peak VRAM (Modal Qwen2.5-VL-3B NF4 batch=1): `3985.81 MiB` (`<= 6,144 MiB`: `True`)

## 1. ActivityNet Captions (`>= 2` Events, Full-Chain Retrieval)

| Phương pháp | Full-Chain Hit@1 (tIoU>=0.5) | Full-Chain Hit@5 | Given-Video Hit@1 | Video Recall@20 | Per-Event tIoU>=0.5 |
|---|---:|---:|---:|---:|---:|
| Whole-Paragraph Global (PE-bigG + PE-L + SigLIP2-400M) | 1.00% | 1.00% | 1.00% | 100.00% | 14.11% |
| Independent Clause Contrast (No Order DP) | 2.50% | 3.00% | 3.00% | 100.00% | 27.58% |
| Viterbi Forward DP (Unified Span Baseline) | 0.50% | 0.50% | 0.50% | 100.00% | 13.48% |
| Exact 2D Semi-Markov Segment DP + Order Margin | 19.50% | 22.00% | 22.50% | 100.00% | 42.19% |
| Exact 2D Semi-Markov DP + Tuned Clause Alignment & Length Penalty (PE-bigG + PE-L + SigLIP2-400M) | 24.50% | 26.00% | 26.50% | 100.00% | 39.67% |

## 2. VideoComp (`ActivityNet-Comp` + `YouCook2-Comp`, Binary Compositional Accuracy)

| Phương pháp | Miền dữ liệu | Temporal Reordering | Action Replacement | Segment Mismatch | `All` (Tích 3 loại) |
|---|---|---:|---:|---:|---:|
| Global Mean Cosine (PE-bigG + PE-L + SigLIP2-400M) | activitynet_comp | 48.31% | 66.00% | 65.91% | 21.01% |
| Global Mean Cosine (PE-bigG + PE-L + SigLIP2-400M) | youcook2_comp | 42.42% | 54.32% | 76.53% | 17.64% |
| Viterbi Forward DP | activitynet_comp | 66.95% | 76.00% | 74.24% | 37.78% |
| Viterbi Forward DP | youcook2_comp | 77.78% | 81.48% | 85.71% | 54.32% |
| 3-Axis Compositional Scorer (Semi-Markov Order + CoM + Min-Clause + Coverage) | activitynet_comp | 61.86% | 73.00% | 81.06% | 36.61% |
| 3-Axis Compositional Scorer (Semi-Markov Order + CoM + Min-Clause + Coverage) | youcook2_comp | 83.84% | 83.95% | 91.84% | 64.64% |
