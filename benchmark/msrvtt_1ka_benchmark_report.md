# MSR-VTT 1K-A Single-Shot Benchmark Results (VIDENTRA v2.1 — RTX 3060 6GB Target)

- **Evaluation date (UTC)**: 2026-11-23T12:45:20Z
- **Dataset**: MSR-VTT 1K-A (JSFusion, 1,000 videos / 1,000 sealed queries)
- **Active Champion**: `R9d_videntra_v2_1_3b_6gb` | **VLM**: `Qwen/Qwen2.5-VL-3B-Instruct (nf4_4bit)` | **Peak VLM VRAM (Kaggle T4)**: `2792.7 MB`
- **Locked Confirm (`valid_confirm`, 2,000 queries) Hit@1**: `51.75%`
- **Sealed Test (`test_1ka`, 1,000 queries) Hit@1**: `57.20%` (`572/1000`), MRR@10 `0.6693`, R@5 `79.90%`, R@10 `87.40%`

| Config ID | Hit@1 (R@1) | R@5 | R@10 | MRR@10 | Δ Hit@1 vs B1 (pp) | 95% Bootstrap CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| `B0_siglip2_standalone` | 42.30% | 65.10% | 74.10% | 0.5207 | -6.40 | [-9.42, -3.44] |
| `B1_videoprism_standalone` | 48.70% | 75.30% | 82.50% | 0.5972 | +0.00 | [+0.00, +0.00] |
| `R8_videntra_v1_champion` | 55.60% | 79.00% | 86.80% | 0.6549 | +6.90 | [+4.68, +9.13] |
| `R9_pe_bigg_standalone` | 53.60% | 77.40% | 84.60% | 0.6386 | +4.90 | [+2.04, +7.73] |
| `R9_dfn5b_h_standalone` | 41.60% | 66.70% | 75.50% | 0.5244 | -7.10 | [-10.00, -4.25] |
| `R9a_6backbone_raw` | 54.30% | 79.10% | 86.50% | 0.6467 | +5.60 | [+3.19, +7.97] |
| `R9b_6backbone_qb_dis_znorm_mp` | 55.90% | 79.70% | 87.40% | 0.6593 | +7.20 | [+4.82, +9.58] |
| `R9c_6backbone_adapter_multimodal` | 56.00% | 79.90% | 87.40% | 0.6603 | +7.30 | [+4.92, +9.71] |
| `R9d_videntra_v2_1_3b_6gb` | 57.20% | 79.90% | 87.40% | 0.6693 | +8.50 | [+6.21, +10.87] |
| `R9d_videntra_v2_7b_cloud_ref` | 57.10% | 79.90% | 87.40% | 0.6670 | +8.40 | [+6.02, +10.80] |
