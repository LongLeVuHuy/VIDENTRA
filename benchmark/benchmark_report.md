# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-16 20:46:05 | **Run ID:** `0e7ec64853df4e4d8601266062f30813` | **Git SHA:** `02803a2` (Dirty: `True`)  
**LTR Model SHA256:** `432c2c831ec89ef1f9f3d4d796a756c60e4fcccf97b8812171cf8a1df21f8e00` | **Eval Mode:** `oof` | **Leakage Free:** `True`  
**Overlap With Eval:** `{"dev_150": 129, "dev_full_859": 742, "heldout_test_201": 0}` | **OOF Stats:** `{"dev_150": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "dev_full_859": {"applied_count": 742, "fallback_count": 117, "total_queries": 859}, "heldout_test_201": {"applied_count": 0, "fallback_count": 0, "total_queries": 201}}`  
**Degraded Channels:** `["e5_caption_index", "spatial_patches", "vlm_caption_lexical"]` | **Disabled Channels:** `["temporal_rerank"]`  
**Channel Status:** `{"e5_text_index": {"status": "ready", "affected_queries": 1210, "sample_reason": "loaded"}, "lexical_db": {"status": "ready", "affected_queries": 1210, "sample_reason": "connected"}, "e5_caption_index": {"status": "missing_artifact", "affected_queries": 1210, "sample_reason": "e5_caption_index_not_loaded"}, "vlm_caption_lexical": {"status": "missing_artifact", "affected_queries": 1210, "sample_reason": "fts_vlm_caption_not_available"}, "spatial_patches": {"status": "missing_artifact", "affected_queries": 1210, "sample_reason": "spatial_patches_not_loaded"}, "temporal_rerank": {"status": "disabled_by_config", "affected_queries": 1210, "sample_reason": "disabled_by_flag"}}`  
**Config:** `{"max_msrvtt": 100, "run_all_splits": true, "splits": ["dev_150", "dev_full_859", "heldout_test_201"], "allow_insample_eval": false, "ltr_model_path": "models\\ranking\\videntra_ltr_v1.json", "ltr_model_resolved_path": "C:\\Users\\DELL\\Desktop\\VIDENTRA\\models\\ranking\\videntra_ltr_v1.json", "ablation_config": "A5", "feature_flags": {"enable_spatial_patches": true, "enable_vlm_caption_lexical": true, "enable_vlm_caption_dense": true, "enable_temporal_rerank": false, "temporal_rerank_mode": "attention", "temporal_rerank_lambda": 0.0, "temporal_rerank_top_k": 10}, "cutoffs": {"top_k": 10, "ranking_mode": "diverse_video", "max_msrvtt": 100}}`  
**Hardware:** `{"platform": "Windows-10-10.0.19045-SP0", "machine": "AMD64", "processor": "Intel64 Family 6 Model 142 Stepping 9, GenuineIntel", "cpu_count": 4, "ram_total_gb": null, "device": "cpu"}`  
**Kiến trúc:** `UnifiedMultimodalSearchPipeline` (Single Source of Truth — SSOT 7-Stage Pipeline)  
**Giới hạn:** các số lịch sử bên dưới chưa được xác minh lại và không dùng để nghiệm thu.  

---

## 1. Bảng Nghiệm Thu Toàn Diện Các Chỉ Tiêu Kế Hoạch Kiểm Toán (P0 $\to$ P4)

| Hạng Mục Nghiệm Thu (Acceptance Criterion) | Ngưỡng Mục Tiêu (Target) | Kết Quả Đạt Được (`dev_150`) | Khoảng Tin Cậy 95% CI | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: |
| **Video Hit@1 (`dev_150`)** | $\ge 59.33\%$ ($\ge 89/150$) | **62.00%** (93/150) | `[54.00%, 69.33%]` | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.6548** (N=150) | `[0.5818, 0.7223]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.6752** (N=150) | `[0.6053, 0.7401]` | **PASS** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **72.73%** (N=22) | `[54.55%, 90.91%]` | **PASS** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.5121** (N=22) | `[0.3319, 0.6738]` | **FAIL** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0/2 pure-ASR** (N=150 total) | — | **PASS** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **53.71 ms / 124.35 ms** | — | **FAIL / FAIL** |
| **End-to-End Latency ($p_{50} / p_{95}$)** | — | **319.9 ms / 652.41 ms** | — | — |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **NOT_MEASURED / NOT_MEASURED** | — | **NOT_MEASURED / NOT_MEASURED** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **NOT_MEASURED** (RSS: NOT_MEASURED) | — | **NOT_MEASURED** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **NOT_MEASURED** | Paired Sign Test | **NOT_MEASURED** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **62.00%** (93/150) | **+10.67%** | **+4.67%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **70.00%** (105/150) | **+1.33%** | **+3.33%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **74.00%** (111/150) | **-1.33%** | **+2.00%** |
| **MRR** | 0.5936 | 0.6251 | **0.6548** | **+0.0612** | **+0.0297** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6200** | **+0.1067** | **+0.0467** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.6615** | **+0.1336** | **+0.0356** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.6752** | **+0.1432** | **+0.0329** |
| **Moment Hit@1 (tIoU $\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **77.27%** | **+18.18%** | **+4.54%** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **72.73%** | **+31.82%** | **+4.55%** |
| **Moment Hit@1 (tIoU $\ge 0.7$)** | — | — | **45.45%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **0.5121** | **+0.1501** | **+0.0100** |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **0.5941 / 0.3977** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **0.3599** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev_150`** | 150 | **62.00%** (`[54.0%, 69.3%]`) | 70.00% | 74.00% | **0.6548** (`[0.582, 0.722]`) | **0.6752** | 64.67% | 0.6841 |
| **`dev_full_859`** | 859 | **43.77%** (`[40.5%, 47.0%]`) | 62.17% | 68.80% | **0.5142** (`[0.484, 0.544]`) | **0.5558** | 46.45% | 0.5439 |
| **`heldout_test_201`** | 201 | **39.80%** (`[33.3%, 46.3%]`) | 58.21% | 68.66% | **0.4805** (`[0.423, 0.540]`) | **0.5291** | 41.29% | 0.5048 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **86.36%** | **95.45%** | 90.91% | 95.45% | **0.8864** | **0.9773** | 0.9026 | 95.5% | 27.3% |
| **AVC** | 28 | 300 videos | **92.86%** | **92.86%** | 92.86% | 92.86% | **0.9286** | **0.9375** | 0.9286 | 82.1% | 75.0% |
| **MSRVTT** | 100 | 1000 videos | **48.00%** | **50.00%** | 59.00% | 64.00% | **0.5273** | **0.5486** | 0.5542 | 79.0% | 20.0% |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **77.33%**
- **`Visual Recall@120`**: **82.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **31.33%**
- **`Visual Candidate Recall`**: **82.00%** (mean pool: `44.8533`)
- **`Union Recall (Visual + Lexical + Caption)`**: **86.00%** (mean pool: `65.1467`)
- **`Shortlist Recall (Pre-Rerank)`**: **86.00%** | `@10`: **74.00%** | `@20`: **78.00%** | `@50`: **84.67%** (mean pool: `65.1467`)
- **`Reranker Input Recall`**: **86.00%** (mean pool: `65.1467`)
- **`Fused Recall@10`**: **74.00%**
- **`Target in Shortlist but Not Top-1`**: **36** (24.00%)

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
| `visual` | 98 queries | 65.3% |
| `asr+visual` | 28 queries | 18.7% |
| `ocr+visual` | 12 queries | 8.0% |
| `asr+ocr+visual` | 7 queries | 4.7% |
| `asr+ocr` | 2 queries | 1.3% |
| `asr` | 2 queries | 1.3% |
| `ocr` | 1 queries | 0.7% |

### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `PASS` | 90 | 60.00% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 22 | 14.67% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 6 | 4.00% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 29 | 19.33% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 3 | 2.00% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** NOT_MEASURED (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **NOT_MEASURED** | Trung vị (`p50`): **NOT_MEASURED** | Ngưỡng `p95`: **NOT_MEASURED**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **462.53 ms** | Trung vị (`p50`): **319.9 ms** | Ngưỡng `p95`: **652.41 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **62.51 ms** | Trung vị (`p50`): **53.71 ms** | Ngưỡng `p95`: **124.35 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `4.18 ms`, SQLite FTS5: `2.85 ms`, Multimodal Fusion: `44.35 ms`, Boundary Expansion: `1.98 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **NOT_MEASURED**, Peak Working Set = **NOT_MEASURED** (xem verdict ở bảng nghiệm thu).
