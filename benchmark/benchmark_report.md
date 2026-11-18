# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-18 14:18:27 | **Run ID:** `6999d9928b5247a292555d84152aff1e` | **Git SHA:** `24fbfcc` (Dirty: `False`)  
**LTR Model SHA256:** `1521b6281b354471a85c20a49fb8930dffcfea4da9d38a653c558e13d8cfd002` | **Eval Mode:** `oof` | **Leakage Free:** `True`  
**Overlap With Eval:** `{"dev_150": 138, "dev_full_859": 781, "heldout_test_201": 0}` | **OOF Stats:** `{"dev_150": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "ablation_A0": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "ablation_A1": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "ablation_A2": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "ablation_A3": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "ablation_A4": {"applied_count": 138, "fallback_count": 12, "total_queries": 150}, "dev_full_859": {"applied_count": 781, "fallback_count": 78, "total_queries": 859}, "heldout_test_201": {"applied_count": 0, "fallback_count": 0, "total_queries": 201}}`  
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
| **Video Hit@1 (`dev_150`)** | $\ge 59.33\%$ ($\ge 89/150$) | **65.33%** (98/150) | `[58.00%, 72.67%]` | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.7122** (N=150) | `[0.6460, 0.7732]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.7372** (N=150) | `[0.6725, 0.7965]` | **PASS** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **72.73%** (N=22) | `[54.55%, 90.91%]` | **PASS** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.5443** (N=22) | `[0.3793, 0.6957]` | **PASS** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0/0 pure-ASR** (N=150 total) | — | **NOT_MEASURED** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **31.57 ms / 79.69 ms** | — | **FAIL / FAIL** |
| **End-to-End Latency ($p_{50} / p_{95}$)** | — | **312.66 ms / 713.22 ms** | — | — |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **319.74 ms / 613.13 ms** | — | **FAIL / FAIL** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **3104.7 MB** (RSS: 2844.0 MB) | — | **PASS** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **$p(\text{Hit@1}) = 0.00050$, $p(\text{MRR}) = 0.00050$** | Paired Sign Test | **PASS** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **65.33%** (98/150) | **+14.00%** | **+8.00%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **80.00%** (120/150) | **+11.33%** | **+13.33%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **81.33%** (122/150) | **+6.00%** | **+9.33%** |
| **MRR** | 0.5936 | 0.6251 | **0.7122** | **+0.1186** | **+0.0871** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6533** | **+0.1400** | **+0.0800** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.7326** | **+0.2047** | **+0.1067** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.7372** | **+0.2052** | **+0.0949** |
| **Moment Hit@1 (tIoU $\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **77.27%** | **+18.18%** | **+4.54%** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **72.73%** | **+31.82%** | **+4.55%** |
| **Moment Hit@1 (tIoU $\ge 0.7$)** | — | — | **45.45%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **0.5443** | **+0.1823** | **+0.0422** |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **0.5988 / 0.3928** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **0.3577** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev_150`** | 150 | **65.33%** (`[58.0%, 72.7%]`) | 80.00% | 81.33% | **0.7122** (`[0.646, 0.773]`) | **0.7372** | 68.67% | 0.7396 |
| **`dev_full_859`** | 859 | **49.13%** (`[45.9%, 52.5%]`) | 71.48% | 77.42% | **0.5863** (`[0.558, 0.614]`) | **0.6318** | 52.15% | 0.6126 |
| **`heldout_test_201`** | 201 | **45.27%** (`[38.8%, 51.7%]`) | 66.67% | 75.12% | **0.5452** (`[0.488, 0.603]`) | **0.5944** | 46.77% | 0.5684 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **90.91%** | **100.00%** | 100.00% | 100.00% | **0.9545** | **1.0000** | 0.9664 | 95.5% | 27.3% |
| **AVC** | 28 | 300 videos | **89.29%** | **92.86%** | 92.86% | 96.43% | **0.9069** | **0.9375** | 0.9201 | 82.1% | 71.4% |
| **MSRVTT** | 100 | 1000 videos | **53.00%** | **55.00%** | 72.00% | 73.00% | **0.6043** | **0.6269** | 0.6356 | 76.0% | 20.0% |

---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Raw 768d SigLIP 2 Visual Baseline | **46.67%** | 64.67% | 70.00% | **0.5398** | **0.5784** | **4.55%** | **0.0143** | 0.0008 |
| **`A1`** | + Decoupled 864d Subspace (B5, B6, B8) | **47.33%** | 64.00% | 68.00% | **0.5432** | **0.5765** | **4.55%** | **0.0167** | 0.0008 |
| **`A2`** | + Multi-Clause & Sequential Alignment (B4, B11) | **47.33%** | 64.00% | 68.00% | **0.5432** | **0.5765** | **4.55%** | **0.0167** | 0.0008 |
| **`A3`** | + Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9) | **57.33%** | 70.67% | 74.00% | **0.6322** | **0.6585** | **4.55%** | **0.0232** | 0.0008 |
| **`A4`** | + Anchor-Aware Contrast Boundary Expansion (B10) | **57.33%** | 69.33% | 74.00% | **0.6315** | **0.6578** | **63.64%** | **0.4364** | 0.2999 |
| **`A5`** | Full VIDENTRA Pipeline (+ Multi-Window Proposal) | **65.33%** | 80.00% | 81.33% | **0.7122** | **0.7372** | **72.73%** | **0.5443** | 0.3577 |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch & Stage Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **76.00%**
- **`Visual Recall@120`**: **80.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **30.67%**
- **`Visual Candidate Recall`**: **80.00%** (mean pool: `44.8`)
- **`Union Recall (Visual + Lexical + Caption)`**: **85.33%** (mean pool: `65.42`)
- **`Shortlist Recall (Pre-Rerank)`**: **93.33%** | `@10`: **78.00%** | `@20`: **81.33%** | `@50`: **90.00%** (mean pool: `80.34`)
- **`Reranker Input Recall`**: **93.33%** (mean pool: `80.34`)
- **`Fused Recall@10`**: **78.00%**
- **`Target in Shortlist but Not Top-1`**: **42** (28.00%)

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
| `videoprism+visual` | 103 queries | 68.7% |
| `asr+videoprism+visual` | 26 queries | 17.3% |
| `ocr+videoprism+visual` | 10 queries | 6.7% |
| `asr+ocr+videoprism+visual` | 5 queries | 3.3% |
| `asr+videoprism` | 3 queries | 2.0% |
| `asr+ocr+videoprism` | 2 queries | 1.3% |
| `ocr+videoprism` | 1 queries | 0.7% |

### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `PASS` | 94 | 62.67% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 25 | 16.67% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 4 | 2.67% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 23 | 15.33% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 4 | 2.67% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency:** Measured (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **368.55 ms** | Trung vị (`p50`): **319.74 ms** | Ngưỡng `p95`: **613.13 ms**
- **Độ trễ End-to-End Online (Query Encoding + Retrieval + Rerank + Boundary):**
  - Trung bình (`Mean`): **477.53 ms** | Trung vị (`p50`): **312.66 ms** | Ngưỡng `p95`: **713.22 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **37.67 ms** | Trung vị (`p50`): **31.57 ms** | Ngưỡng `p95`: **79.69 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `4.26 ms`, SQLite FTS5: `5.67 ms`, Multimodal Fusion: `18.85 ms`, Boundary Expansion: `1.75 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **2844.0 MB**, Peak Working Set = **3104.7 MB** (xem verdict ở bảng nghiệm thu).
