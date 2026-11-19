# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-19 16:15:42 | **Run ID:** `2b42562bd2a04d0f9dfe212495cfedbc` | **Git SHA:** `ded80af` (Dirty: `False`)  
**LTR Model SHA256:** `4f6b3174348559d8b65e250377aa5259ce3771761d35b4ce868b8b5ea392c47f` | **Eval Mode:** `oof` | **Leakage Free:** `True`  
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
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.7137** (N=150) | `[0.6510, 0.7752]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.7448** (N=150) | `[0.6866, 0.8040]` | **PASS** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **54.55%** (N=22) | `[31.82%, 77.27%]` | **FAIL** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.4852** (N=22) | `[0.3296, 0.6381]` | **FAIL** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0/0 pure-ASR** (N=150 total) | — | **NOT_MEASURED** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **71.12 ms / 310.13 ms** | — | **FAIL / FAIL** |
| **End-to-End Latency ($p_{50} / p_{95}$)** | — | **835.03 ms / 1898.53 ms** | — | — |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **510.84 ms / 1037.35 ms** | — | **FAIL / FAIL** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **3654.3 MB** (RSS: 3397.3 MB) | — | **PASS** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **$p(\text{Hit@1}) = 0.00050$, $p(\text{MRR}) = 0.00050$** | Paired Sign Test | **PASS** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **64.00%** (96/150) | **+12.67%** | **+6.67%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **81.33%** (122/150) | **+12.66%** | **+14.66%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **84.00%** (126/150) | **+8.67%** | **+12.00%** |
| **MRR** | 0.5936 | 0.6251 | **0.7137** | **+0.1201** | **+0.0886** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6400** | **+0.1267** | **+0.0667** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.7361** | **+0.2082** | **+0.1102** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.7448** | **+0.2128** | **+0.1025** |
| **Moment Hit@1 (tIoU $\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **72.73%** | **+13.64%** | **+0.00%** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **54.55%** | **+13.64%** | **-13.63%** |
| **Moment Hit@1 (tIoU $\ge 0.7$)** | — | — | **36.36%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **0.4852** | **+0.1232** | **-0.0169** |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **0.5273 / 0.3371** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **0.3171** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev_150`** | 150 | **64.00%** (`[56.7%, 71.3%]`) | 81.33% | 84.00% | **0.7137** (`[0.651, 0.775]`) | **0.7448** | 70.67% | 0.7613 |
| **`dev_full_859`** | 859 | **49.01%** (`[45.5%, 52.5%]`) | 73.81% | 80.09% | **0.5898** (`[0.561, 0.618]`) | **0.6407** | 52.62% | 0.6207 |
| **`heldout_test_201`** | 201 | **44.28%** (`[37.8%, 51.2%]`) | 67.66% | 77.11% | **0.5448** (`[0.488, 0.605]`) | **0.5988** | 45.27% | 0.5678 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **77.27%** | **100.00%** | 100.00% | 100.00% | **0.8864** | **1.0000** | 0.9161 | 90.9% | 27.3% |
| **AVC** | 28 | 300 videos | **89.29%** | **92.86%** | 92.86% | 96.43% | **0.9040** | **0.9405** | 0.9174 | 75.0% | 71.4% |
| **MSRVTT** | 100 | 1000 videos | **54.00%** | **58.00%** | 74.00% | 77.00% | **0.6225** | **0.6587** | 0.6588 | 79.0% | 20.0% |

---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Raw 768d SigLIP 2 Visual Baseline | **47.33%** | 66.67% | 71.33% | **0.5537** | **0.5923** | **9.09%** | **0.0347** | 0.0060 |
| **`A1`** | + Decoupled 864d Subspace (B5, B6, B8) | **44.00%** | 66.00% | 72.67% | **0.5320** | **0.5787** | **4.55%** | **0.0228** | 0.0021 |
| **`A2`** | + Multi-Clause & Sequential Alignment (B4, B11) | **44.00%** | 66.00% | 72.67% | **0.5320** | **0.5787** | **4.55%** | **0.0228** | 0.0021 |
| **`A3`** | + Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9) | **55.33%** | 74.00% | 80.00% | **0.6282** | **0.6692** | **4.55%** | **0.0285** | 0.0021 |
| **`A4`** | + Anchor-Aware Contrast Boundary Expansion (B10) | **55.33%** | 74.00% | 80.00% | **0.6282** | **0.6692** | **50.00%** | **0.4164** | 0.2780 |
| **`A5`** | Full VIDENTRA Pipeline (+ Multi-Window Proposal) | **64.00%** | 81.33% | 84.00% | **0.7137** | **0.7448** | **54.55%** | **0.4852** | 0.3171 |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **76.67%**
- **`Visual Recall@120`**: **80.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **30.67%**
- **`Visual Candidate Recall`**: **80.00%** (mean pool: `40.28`)
- **`Union Recall (Visual + Lexical + Caption)`**: **86.00%** (mean pool: `61.1667`)
- **`Shortlist Recall (Pre-Rerank)`**: **94.00%** | `@10`: **78.00%** | `@20`: **83.33%** | `@50`: **92.67%** (mean pool: `77.2733`)
- **`Reranker Input Recall`**: **94.00%** (mean pool: `77.2733`)
- **`Fused Recall@10`**: **78.00%**
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
| `PASS` | 91 | 60.67% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 23 | 15.33% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 6 | 4.00% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 25 | 16.67% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 5 | 3.33% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** Measured (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **600.22 ms** | Trung vị (`p50`): **510.84 ms** | Ngưỡng `p95`: **1037.35 ms**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **1136.35 ms** | Trung vị (`p50`): **835.03 ms** | Ngưỡng `p95`: **1898.53 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **97.95 ms** | Trung vị (`p50`): **71.12 ms** | Ngưỡng `p95`: **310.13 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `14.52 ms`, SQLite FTS5: `9.89 ms`, Multimodal Fusion: `40.04 ms`, Boundary Expansion: `4.91 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **3397.3 MB**, Peak Working Set = **3654.3 MB** (xem verdict ở bảng nghiệm thu).
