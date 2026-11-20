# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-20 13:44:41 | **Run ID:** `1d90b8f9f09f41629f9a500229a47626` | **Git SHA:** `7ead8e0` (Dirty: `False`)  
**LTR Model SHA256:** `7034092de6f91833bfb8e62c67b16eddf2404b9857c2a1c23430e72c3934f1db` | **Eval Mode:** `oof` | **Leakage Free:** `True`  
**Overlap With Eval:** `{"dev_150": 141, "dev_full_859": 789, "heldout_test_201": 0}` | **OOF Stats:** `{"dev_150": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "ablation_A0": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "ablation_A1": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "ablation_A2": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "ablation_A3": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "ablation_A4": {"applied_count": 141, "fallback_count": 9, "total_queries": 150}, "dev_full_859": {"applied_count": 789, "fallback_count": 70, "total_queries": 859}, "heldout_test_201": {"applied_count": 0, "fallback_count": 0, "total_queries": 201}}`  
**Degraded Channels:** `["spatial_patches"]` | **Disabled Channels:** `["e5_caption_index", "e5_text_index", "lexical_db", "spatial_patches", "temporal_rerank", "vlm_caption_lexical"]`  
**Channel Status:** `{"e5_text_index": {"status": "ready", "affected_queries": 1960, "sample_reason": "loaded"}, "lexical_db": {"status": "ready", "affected_queries": 1960, "sample_reason": "connected"}, "e5_caption_index": {"status": "disabled_by_config", "affected_queries": 1960, "sample_reason": "disabled_by_flag"}, "vlm_caption_lexical": {"status": "disabled_by_config", "affected_queries": 1960, "sample_reason": "disabled_by_flag"}, "spatial_patches": {"status": "missing_artifact", "affected_queries": 1960, "sample_reason": "spatial_patches_not_loaded"}, "temporal_rerank": {"status": "disabled_by_config", "affected_queries": 1960, "sample_reason": "disabled_by_flag"}}`  
**Config:** `{"max_msrvtt": 100, "run_all_splits": true, "splits": ["dev_150", "dev_full_859", "heldout_test_201"], "allow_insample_eval": false, "publish_reference": true, "db_path": "data\\processed\\metadata.db", "faiss_path": "data\\processed\\faiss_index.bin", "model_dir": "models\\siglip2", "ltr_model_path": "models\\ranking\\videntra_ltr_v1.json", "ltr_model_resolved_path": "C:\\Users\\DELL\\Desktop\\VIDENTRA\\models\\ranking\\videntra_ltr_v1.json", "ablation_config": "A5", "feature_flags": {"stage2_selection_mode": "exact", "video_proposal_mode": "frame_topk", "video_proposal_top_n": 60, "video_proposal_anchors_per_video": 2, "video_proposal_min_pts_gap": 1.0, "video_proposal_clip_reserve_ratio": 0.0, "disable_pure_visual_lexical_bonus": false, "enable_spatial_patches": true, "enable_vlm_caption_lexical": false, "enable_vlm_caption_dense": false, "enable_temporal_rerank": false, "temporal_rerank_mode": "attention", "temporal_rerank_lambda": 0.0, "temporal_rerank_top_k": 10, "enable_clip_consensus_boost": true, "ltr_blend_weight": 0.015, "a5_preset": null}, "cutoffs": {"top_k": 10, "ranking_mode": "diverse_video", "max_msrvtt": 100}}`  
**Hardware:** `{"platform": "Windows-10-10.0.19045-SP0", "machine": "AMD64", "processor": "Intel64 Family 6 Model 142 Stepping 9, GenuineIntel", "cpu_count": 4, "ram_total_gb": 31.89, "device": "cpu"}`  
**Kiến trúc:** `UnifiedMultimodalSearchPipeline` (Single Source of Truth — SSOT 7-Stage Pipeline)  
**Giới hạn:** các số lịch sử bên dưới chưa được xác minh lại và không dùng để nghiệm thu.  

---

## 1. Bảng Nghiệm Thu Toàn Diện Các Chỉ Tiêu Kế Hoạch Kiểm Toán (P0 $\to$ P4)

| Hạng Mục Nghiệm Thu (Acceptance Criterion) | Ngưỡng Mục Tiêu (Target) | Kết Quả Đạt Được (`dev_150`) | Khoảng Tin Cậy 95% CI | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: |
| **Video Hit@1 (`dev_150`)** | $\ge 59.33\%$ ($\ge 89/150$) | **64.00%** (96/150) | `[56.67%, 71.33%]` | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.7129** (N=150) | `[0.6475, 0.7730]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.7455** (N=150) | `[0.6851, 0.8025]` | **PASS** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **59.09%** (N=22) | `[36.36%, 81.82%]` | **FAIL** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.5176** (N=22) | `[0.3576, 0.6602]` | **FAIL** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0/0 pure-ASR** (N=150 total) | — | **NOT_MEASURED** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **38.23 ms / 93.5 ms** | — | **FAIL / FAIL** |
| **End-to-End Latency ($p_{50} / p_{95}$)** | — | **352.84 ms / 700.69 ms** | — | — |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **291.09 ms / 582.84 ms** | — | **FAIL / FAIL** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **3672.1 MB** (RSS: 3414.1 MB) | — | **PASS** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **$p(\text{Hit@1}) = 0.00050$, $p(\text{MRR}) = 0.00050$** | Paired Sign Test | **PASS** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **64.00%** (96/150) | **+12.67%** | **+6.67%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **81.33%** (122/150) | **+12.66%** | **+14.66%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **84.67%** (127/150) | **+9.34%** | **+12.67%** |
| **MRR** | 0.5936 | 0.6251 | **0.7129** | **+0.1193** | **+0.0878** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6400** | **+0.1267** | **+0.0667** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.7348** | **+0.2069** | **+0.1089** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.7455** | **+0.2135** | **+0.1032** |
| **Moment Hit@1 (tIoU $\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **81.82%** | **+22.73%** | **+9.09%** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **59.09%** | **+18.18%** | **-9.09%** |
| **Moment Hit@1 (tIoU $\ge 0.7$)** | — | — | **40.91%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **0.5176** | **+0.1556** | **+0.0155** |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **0.5208 / 0.3497** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **0.3383** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev_150`** | 150 | **64.00%** (`[56.7%, 71.3%]`) | 81.33% | 84.67% | **0.7129** (`[0.647, 0.773]`) | **0.7455** | 68.67% | 0.7510 |
| **`dev_full_859`** | 859 | **49.01%** (`[45.6%, 52.4%]`) | 73.22% | 80.21% | **0.5915** (`[0.562, 0.620]`) | **0.6423** | 52.50% | 0.6217 |
| **`heldout_test_201`** | 201 | **45.77%** (`[39.3%, 52.8%]`) | 67.66% | 77.61% | **0.5532** (`[0.495, 0.611]`) | **0.6061** | 47.76% | 0.5795 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **86.36%** | **100.00%** | 100.00% | 100.00% | **0.9318** | **1.0000** | 0.9497 | 90.9% | 27.3% |
| **AVC** | 28 | 300 videos | **89.29%** | **92.86%** | 92.86% | 96.43% | **0.9051** | **0.9464** | 0.9186 | 75.0% | 71.4% |
| **MSRVTT** | 100 | 1000 videos | **52.00%** | **55.00%** | 74.00% | 78.00% | **0.6109** | **0.6414** | 0.6522 | 79.0% | 20.0% |

---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Raw 768d SigLIP 2 Visual Baseline | **48.00%** | 68.00% | 72.00% | **0.5567** | **0.5962** | **9.09%** | **0.0347** | 0.0060 |
| **`A1`** | + Decoupled 864d Subspace (B5, B6, B8) | **44.67%** | 67.33% | 72.67% | **0.5391** | **0.5845** | **4.55%** | **0.0228** | 0.0021 |
| **`A2`** | + Multi-Clause & Sequential Alignment (B4, B11) | **43.33%** | 66.67% | 72.67% | **0.5322** | **0.5794** | **4.55%** | **0.0228** | 0.0021 |
| **`A3`** | + Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9) | **54.67%** | 74.67% | 79.33% | **0.6274** | **0.6677** | **4.55%** | **0.0285** | 0.0021 |
| **`A4`** | + Anchor-Aware Contrast Boundary Expansion (B10) | **54.67%** | 74.67% | 79.33% | **0.6274** | **0.6677** | **50.00%** | **0.4164** | 0.2781 |
| **`A5`** | Full VIDENTRA Pipeline (+ Multi-Window Proposal) | **64.00%** | 81.33% | 84.67% | **0.7129** | **0.7455** | **59.09%** | **0.5176** | 0.3383 |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **76.67%**
- **`Visual Recall@120`**: **80.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **30.67%**
- **`Visual Candidate Recall`**: **80.00%** (mean pool: `40.28`)
- **`Union Recall (Visual + Lexical + Caption)`**: **86.00%** (mean pool: `61.1`)
- **`Shortlist Recall (Pre-Rerank)`**: **94.00%** | `@10`: **79.33%** | `@20`: **83.33%** | `@50`: **92.67%** (mean pool: `77.2`)
- **`Reranker Input Recall`**: **94.00%** (mean pool: `77.2`)
- **`Fused Recall@10`**: **79.33%**
- **`Target in Shortlist but Not Top-1`**: **45** (30.00%)

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
| `videoprism+visual` | 99 queries | 66.0% |
| `asr+videoprism+visual` | 28 queries | 18.7% |
| `ocr+videoprism+visual` | 11 queries | 7.3% |
| `asr+ocr+videoprism+visual` | 4 queries | 2.7% |
| `asr+ocr+videoprism` | 3 queries | 2.0% |
| `asr+videoprism` | 3 queries | 2.0% |
| `videoprism` | 1 queries | 0.7% |
| `ocr+videoprism` | 1 queries | 0.7% |

### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `PASS` | 90 | 60.00% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 23 | 15.33% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 6 | 4.00% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 25 | 16.67% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 6 | 4.00% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** Measured (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **344.86 ms** | Trung vị (`p50`): **291.09 ms** | Ngưỡng `p95`: **582.84 ms**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **477.84 ms** | Trung vị (`p50`): **352.84 ms** | Ngưỡng `p95`: **700.69 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **44.17 ms** | Trung vị (`p50`): **38.23 ms** | Ngưỡng `p95`: **93.5 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `7.91 ms`, SQLite FTS5: `6.04 ms`, Multimodal Fusion: `20.55 ms`, Boundary Expansion: `2.76 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **3414.1 MB**, Peak Working Set = **3672.1 MB** (xem verdict ở bảng nghiệm thu).
