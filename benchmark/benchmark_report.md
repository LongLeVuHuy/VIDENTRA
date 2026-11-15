# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-15 15:31:26 | **Run ID:** `08626d79ef744e3080c2fb24c4761cad` | **Git SHA:** `dffdf8a` (Dirty: `True`)\
**LTR Model SHA256:** `432c2c831ec89ef1f9f3d4d796a756c60e4fcccf97b8812171cf8a1df21f8e00` | **Eval Mode:** `oof` | **Leakage Free:** `True`\
**Overlap With Eval:** `{"dev_150": 129, "dev_full_859": 742, "heldout_test_201": 0}` | **OOF Stats:** `{"dev_150": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "ablation_A0": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "ablation_A1": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "ablation_A2": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "ablation_A3": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "ablation_A4": {"applied_count": 129, "fallback_count": 21, "total_queries": 150}, "dev_full_859": {"applied_count": 742, "fallback_count": 117, "total_queries": 859}, "heldout_test_201": {"applied_count": 0, "fallback_count": 0, "total_queries": 201}}`\
**Degraded Channels:** `["e5_caption_index", "spatial_patches"]` | **Disabled Channels:** `["e5_caption_index", "e5_text_index", "lexical_db", "spatial_patches"]`\
**Channel Status:** `{"e5_text_index": {"status": "ready", "affected_queries": 1960, "sample_reason": "loaded"}, "e5_caption_index": {"status": "missing_artifact", "affected_queries": 1960, "sample_reason": "e5_caption_index_not_loaded"}, "lexical_db": {"status": "ready", "affected_queries": 1960, "sample_reason": "connected"}, "spatial_patches": {"status": "missing_artifact", "affected_queries": 1960, "sample_reason": "spatial_patches_not_loaded"}}`\
**Config:** `{"max_msrvtt": 100, "run_all_splits": true, "allow_insample_eval": false, "ltr_model_path": "models\\ranking\\videntra_ltr_v1.json", "ltr_model_resolved_path": "<repo-root>/models/ranking/videntra_ltr_v1.json", "ablation_config": "A5", "cutoffs": {"top_k": 10, "ranking_mode": "diverse_video", "max_msrvtt": 100}}`\
**Hardware:** `{"platform": "Windows-10-10.0.19045-SP0", "machine": "AMD64", "processor": "Intel64 Family 6 Model 142 Stepping 9, GenuineIntel", "cpu_count": 4, "ram_total_gb": null, "device": "cpu"}`\
**Kiến trúc:** `UnifiedMultimodalSearchPipeline` (Single Source of Truth — SSOT 7-Stage Pipeline)\
**Giới hạn:** các số lịch sử bên dưới chưa được xác minh lại và không dùng để nghiệm thu.\

---

## 1. Bảng Nghiệm Thu Toàn Diện Các Chỉ Tiêu Kế Hoạch Kiểm Toán (P0 $\to$ P4)

| Hạng Mục Nghiệm Thu (Acceptance Criterion) | Ngưỡng Mục Tiêu (Target) | Kết Quả Đạt Được (`dev_150`) | Khoảng Tin Cậy 95% CI | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: |
| **Video Hit@1 (`dev_150`)** | $\ge 59.33\%$ ($\ge 89/150$) | **61.33%** (92/150) | `[53.33%, 68.67%]` | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.6515** (N=150) | `[0.5792, 0.7204]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.6727** (N=150) | `[0.6046, 0.7379]` | **FAIL** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **72.73%** (N=22) | `[54.55%, 90.91%]` | **PASS** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.5121** (N=22) | `[0.3319, 0.6738]` | **FAIL** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0/2 pure-ASR** (N=150 total) | — | **PASS** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **40.47 ms / 122.49 ms** | — | **FAIL / FAIL** |
| **End-to-End Latency ($p_{50} / p_{95}$)** | — | **376.21 ms / 775.62 ms** | — | — |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **NOT_MEASURED / NOT_MEASURED** | — | **NOT_MEASURED / NOT_MEASURED** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **NOT_MEASURED** (RSS: NOT_MEASURED) | — | **NOT_MEASURED** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **$p(\text{Hit@1}) = 0.00050$, $p(\text{MRR}) = 0.00050$** | Paired Sign Test | **PASS** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **61.33%** (92/150) | **+10.00%** | **+4.00%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **70.00%** (105/150) | **+1.33%** | **+3.33%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **74.00%** (111/150) | **-1.33%** | **+2.00%** |
| **MRR** | 0.5936 | 0.6251 | **0.6515** | **+0.0579** | **+0.0264** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6133** | **+0.1000** | **+0.0400** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.6591** | **+0.1312** | **+0.0332** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.6727** | **+0.1407** | **+0.0304** |
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
| **`dev_150`** | 150 | **61.33%** (`[53.3%, 68.7%]`) | 70.00% | 74.00% | **0.6515** (`[0.579, 0.720]`) | **0.6727** | 64.00% | 0.6807 |
| **`dev_full_859`** | 859 | **41.56%** (`[38.3%, 44.9%]`) | 61.93% | 68.80% | **0.5010** (`[0.471, 0.531]`) | **0.5459** | 44.70% | 0.5337 |
| **`heldout_test_201`** | 201 | **38.81%** (`[32.3%, 45.3%]`) | 58.21% | 68.66% | **0.4755** (`[0.417, 0.536]`) | **0.5255** | 40.80% | 0.5023 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **86.36%** | **95.45%** | 90.91% | 95.45% | **0.8864** | **0.9773** | 0.9026 | 95.5% | 27.3% |
| **AVC** | 28 | 300 videos | **92.86%** | **92.86%** | 92.86% | 92.86% | **0.9286** | **0.9375** | 0.9286 | 82.1% | 75.0% |
| **MSRVTT** | 100 | 1000 videos | **47.00%** | **49.00%** | 59.00% | 64.00% | **0.5223** | **0.5436** | 0.5505 | 79.0% | 20.0% |

---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Raw 768d SigLIP 2 Visual Baseline | **46.67%** | 64.67% | 70.00% | **0.5394** | **0.5781** | **4.55%** | **0.0143** | 0.0008 |
| **`A1`** | + Decoupled 864d Subspace (B5, B6, B8) | **49.33%** | 63.33% | 68.00% | **0.5523** | **0.5832** | **4.55%** | **0.0175** | 0.0008 |
| **`A2`** | + Multi-Clause & Sequential Alignment (B4, B11) | **49.33%** | 63.33% | 68.00% | **0.5523** | **0.5832** | **4.55%** | **0.0175** | 0.0008 |
| **`A3`** | + Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9) | **61.33%** | 70.67% | 74.00% | **0.6517** | **0.6729** | **4.55%** | **0.0240** | 0.0008 |
| **`A4`** | + Anchor-Aware Contrast Boundary Expansion (B10) | **61.33%** | 70.67% | 74.00% | **0.6517** | **0.6729** | **72.73%** | **0.4986** | 0.3543 |
| **`A5`** | Full VIDENTRA Pipeline (+ Multi-Window Proposal) | **61.33%** | 70.00% | 74.00% | **0.6515** | **0.6727** | **72.73%** | **0.5121** | 0.3599 |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **77.33%**
- **`Visual Recall@120`**: **82.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **31.33%**
- **`Union Recall (Visual + Lexical)`**: **86.00%**
- **`Shortlist Recall (Pre-Rerank)`**: **86.00%**
- **`Fused Recall@10`**: **74.00%**
- **`Target in Shortlist but Not Top-1`**: **37** (24.67%)

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
| `visual` | 97 queries | 64.7% |
| `asr+visual` | 28 queries | 18.7% |
| `ocr+visual` | 13 queries | 8.7% |
| `asr+ocr+visual` | 7 queries | 4.7% |
| `asr+ocr` | 2 queries | 1.3% |
| `asr` | 2 queries | 1.3% |
| `ocr` | 1 queries | 0.7% |

### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `PASS` | 89 | 59.33% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 22 | 14.67% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 8 | 5.33% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 28 | 18.67% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 3 | 2.00% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** NOT_MEASURED (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **NOT_MEASURED** | Trung vị (`p50`): **NOT_MEASURED** | Ngưỡng `p95`: **NOT_MEASURED**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **525.13 ms** | Trung vị (`p50`): **376.21 ms** | Ngưỡng `p95`: **775.62 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **48.48 ms** | Trung vị (`p50`): **40.47 ms** | Ngưỡng `p95`: **122.49 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `4.49 ms`, SQLite FTS5: `3.16 ms`, Multimodal Fusion: `28.93 ms`, Boundary Expansion: `2.38 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **NOT_MEASURED**, Peak Working Set = **NOT_MEASURED** (xem verdict ở bảng nghiệm thu).
