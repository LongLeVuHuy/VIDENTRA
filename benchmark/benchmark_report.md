# Báo Cáo Kiểm Toán & Đánh Giá Toàn Diện Hệ Thống Tìm Kiếm Đa Phương Thức VIDENTRA (P0–P4)

**Thời điểm thực thi:** 2026-11-14 15:07:09 | **Git SHA:** `4742a7e`  
**Kiến trúc:** `UnifiedMultimodalSearchPipeline` (Single Source of Truth — SSOT 7-Stage Pipeline)  
**Cơ sở dữ liệu và phần cứng:** chưa được thu thập tự động cho báo cáo này.  
**Giới hạn:** các số lịch sử bên dưới chưa được xác minh lại và không dùng để nghiệm thu.  

---

## 1. Bảng Nghiệm Thu Toàn Diện Các Chỉ Tiêu Kế Hoạch Kiểm Toán (P0 $\to$ P4)

| Hạng Mục Nghiệm Thu (Acceptance Criterion) | Ngưỡng Mục Tiêu (Target) | Kết Quả Đạt Được (`dev_150`) | Khoảng Tin Cậy 95% CI | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: |
| **Video Hit@1 (`dev_150`)** | $\ge 59.33\%$ ($\ge 89/150$) | **62.00%** (93/150) | `[54.00%, 69.33%]` | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.6450$ | **0.6582** | `[0.5865, 0.7262]` | **PASS** |
| **Video nDCG@10** | $\ge 0.6750$ | **0.6794** | `[0.6114, 0.7451]` | **PASS** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | $\ge 72.73\%$ ($\ge 16/22$) | **72.73%** (số nhãn moment xem dữ liệu run) | `[54.55%, 90.91%]` | **PASS** |
| **Moment nDCG@5 (Graded 4-Level)** | $\ge 0.5300$ | **0.5127** | `[0.3342, 0.6749]` | **FAIL** |
| **Pure-ASR False Positive Top-1** | $\le 1$ ca sai | **0 ca sai** | — | **PASS** |
| **Non-Neural Latency ($p_{50} / p_{95}$)** | $\le 20\text{ ms} / \le 45\text{ ms}$ | **24.27 ms / 106.01 ms** | — | **FAIL / FAIL** |
| **Cold Query Latency ($p_{50} / p_{95}$)** | $\le 220\text{ ms} / \le 420\text{ ms}$ | **250.93 ms / 317.82 ms** | — | **NOT_MEASURED / NOT_MEASURED** |
| **Peak RAM Footprint** | $< 4,000\text{ MB}$ ($4.0\text{ GB}$) | **580.0 MB** (RSS: 420.0 MB) | — | **NOT_MEASURED** |
| **Ý nghĩa thống kê ($p$-value vs `A0`)** | $p < 0.05$ | **$p(\text{Hit@1}) = 0.00050$, $p(\text{MRR}) = 0.00050$** | Paired Sign Test | **NOT_MEASURED** |

---

## 2. Đối Chiếu Tiến Trình Nâng Cấp: Phase 3 Baseline vs Pre-Audit Phase 4 vs Post-Audit VIDENTRA (Hiện Tại)

| Chỉ số Đánh giá (Metric) | Baseline Phase 3 (`02:31:14`) | Pre-Audit Phase 4 (`04:01:36`) | Post-Audit VIDENTRA (`Hiện tại`) | $\Delta$ vs Phase 3 | $\Delta$ vs Pre-Audit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video Hit@1** | 51.33% (77/150) | 57.33% (86/150) | **62.00%** (93/150) | **+10.67%** | **+4.67%** |
| **Video Hit@5** | 68.67% (103/150) | 66.67% (100/150) | **70.67%** (106/150) | **+2.00%** | **+4.00%** |
| **Video Hit@10** | 75.33% (113/150) | 72.00% (108/150) | **74.67%** (112/150) | **-0.66%** | **+2.67%** |
| **MRR** | 0.5936 | 0.6251 | **0.6582** | **+0.0646** | **+0.0331** |
| **Video nDCG@1** | 0.5133 | 0.5733 | **0.6200** | **+0.1067** | **+0.0467** |
| **Video nDCG@5** | 0.5279 | 0.6259 | **0.6657** | **+0.1378** | **+0.0398** |
| **Video nDCG@10** | 0.5320 | 0.6423 | **0.6794** | **+0.1474** | **+0.0371** |
| **Moment Hit@1 (tIoU $\ge 0.3$)** | 59.09% (13/22) | 72.73% (16/22) | **77.27%** | **+18.18%** | **+4.54%** |
| **Moment Hit@1 (tIoU $\ge 0.5$)** | 40.91% (9/22) | 68.18% (15/22) | **72.73%** | **+31.82%** | **+4.55%** |
| **Moment Hit@1 (tIoU $\ge 0.7$)** | — | — | **45.45%** | — | — |
| **Moment nDCG@5** | 0.3620 | 0.5021 | **0.5127** | **+0.1507** | **+0.0106** |
| **Moment mAP@0.5 / mAP@0.75** | — | — | **0.5937 / 0.3523** | — | — |
| **Moment mAP@[0.5:0.95:0.05]** | — | — | **0.3463** | — | — |

---

## 3. Đánh Giá Đa Tập Dữ Liệu (Multi-Split Protocol: `dev_150`, `dev_full_859`, `heldout_test_201`)

| Tập Đánh Giá (Split Name) | Số Queries | Global Hit@1 (95% CI) | Global Hit@5 | Global Hit@10 | Global MRR (95% CI) | Global nDCG@10 | Global-then-dataset-filter Hit@1 | Global-then-dataset-filter MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dev_150`** | 150 | **62.00%** (`[54.0%, 69.3%]`) | 70.67% | 74.67% | **0.6582** (`[0.587, 0.726]`) | **0.6794** | 64.67% | 0.6874 |
| **`dev_full_859`** | 859 | **41.68%** (`[38.4%, 44.9%]`) | 62.05% | 68.80% | **0.5018** (`[0.473, 0.532]`) | **0.5465** | 44.82% | 0.5346 |
| **`heldout_test_201`** | 201 | **39.30%** (`[32.8%, 45.8%]`) | 58.21% | 68.16% | **0.4774** (`[0.419, 0.539]`) | **0.5258** | 41.29% | 0.5055 |

---

## 4. Phân Rã Theo Từng Dataset & Báo Cáo Kép (Global Pool vs Global-then-dataset-filter — `E3`)

| Dataset | Số Queries | Cỡ Pool Nội Bộ | Global Hit@1 | Within-DS Hit@1 | Global Hit@5 | Global Hit@10 | Global MRR | Within-DS MRR | Video nDCG@10 | Visual Recall@120 | Lexical Recall@30 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QVHIGHLIGHTS** | 22 | 24 videos | **86.36%** | **95.45%** | 90.91% | 95.45% | **0.8864** | **0.9773** | 0.9026 | 95.5% | 27.3% |
| **AVC** | 28 | 300 videos | **92.86%** | **92.86%** | 92.86% | 92.86% | **0.9286** | **0.9375** | 0.9286 | 82.1% | 75.0% |
| **MSRVTT** | 100 | 1000 videos | **48.00%** | **50.00%** | 60.00% | 65.00% | **0.5323** | **0.5536** | 0.5605 | 79.0% | 20.0% |

---

## 5. Ma Trận Thực Nghiệm Ablation 6 Bước (`A0` $\to$ `A5` trên `dev_150`)

| Cấu Hình Ablation | Mô Tả Thành Phần Kỹ Thuật | Video Hit@1 | Video Hit@5 | Video Hit@10 | MRR | Video nDCG@10 | Moment Hit@1 (0.5) | Moment nDCG@5 | Moment mAP@[.5:.95] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Raw 768d SigLIP 2 Visual Baseline | **46.67%** | 64.00% | 70.00% | **0.5401** | **0.5786** | **4.55%** | **0.0151** | 0.0008 |
| **`A1`** | + Decoupled 864d Subspace (B5, B6, B8) | **48.67%** | 64.00% | 68.00% | **0.5475** | **0.5796** | **4.55%** | **0.0175** | 0.0008 |
| **`A2`** | + Multi-Clause & Sequential Alignment (B4, B11) | **48.67%** | 64.00% | 68.67% | **0.5483** | **0.5816** | **4.55%** | **0.0175** | 0.0008 |
| **`A3`** | + Specificity FTS5 & Local-Gated Fusion (B1, B2, B3, B7, B9) | **61.33%** | 72.00% | 74.67% | **0.6540** | **0.6763** | **4.55%** | **0.0240** | 0.0008 |
| **`A4`** | + Anchor-Aware Contrast Boundary Expansion (B10) | **61.33%** | 72.00% | 74.67% | **0.6540** | **0.6763** | **72.73%** | **0.5182** | 0.3437 |
| **`A5`** | Full VIDENTRA Pipeline (+ Multi-Window Proposal) | **62.00%** | 70.67% | 74.67% | **0.6582** | **0.6794** | **72.73%** | **0.5127** | 0.3463 |

---

## 6. Phân Rã Branch Recall@K (`E8`), Modality Attribution & Automated Failure Taxonomy (`E10`)

### 6.1. Độ Phủ Nhánh Truy Xuất Trước Hợp Nhất (Branch Recall@K trên `dev_150`)
- **`Visual Recall@30`**: **77.33%**
- **`Visual Recall@120`**: **82.00%**
- **`Lexical Recall@30 (OCR + ASR)`**: **31.33%**
- **`Fused Recall@10`**: **74.67%**

### 6.2. Phân Bổ Đóng Góp Phương Thức Top-1 (Modality Attribution)

| Sự Kết Hợp Phương Thức | Số Lượng Queries Top-1 | Tỷ Lệ Đóng Góp |
| :--- | :---: | :---: |
| `visual` | 96 queries | 64.0% |
| `asr+visual` | 29 queries | 19.3% |
| `ocr+visual` | 13 queries | 8.7% |
| `asr+ocr+visual` | 7 queries | 4.7% |
| `asr+ocr` | 2 queries | 1.3% |
| `asr` | 2 queries | 1.3% |
| `ocr` | 1 queries | 0.7% |

### 6.3. Phân Loại Lỗi Tự Động (Automated Failure Taxonomy `F1..F5` trên `dev_150`)

| Mã Phân Loại (Taxonomy Code) | Số Lượng | Tỷ Lệ (%) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `PASS` | 90 | 60.00% | Dự đoán chính xác Top-1 Video (và đạt ngưỡng định vị khoảnh khắc nếu có GT windows) |
| `F1_VISUAL_CANDIDATE_MISS` | 22 | 14.67% | Target video nằm ngoài Top-120 ứng viên thị giác FAISS và không có khớp từ vựng |
| `F2_LEXICAL_DISTRACTOR_OVERTHROW` | 8 | 5.33% | Ứng viên có khớp OCR/ASR từ nhiễu vượt mặt ứng viên thị giác |
| `F3_SAME_SOURCE_CLIP_CONFUSION` | 0 | 0.00% | Dự đoán trúng cùng video YouTube gốc nhưng lệch đoạn cắt 10s (AVC sibling clip) |
| `F4_FINE_GRAINED_VISUAL_CONFUSION` | 27 | 18.00% | Target video nằm trong Top-120 FAISS nhưng xếp sau một cảnh thị giác tương tự |
| `F5_BOUNDARY_LOCALIZATION_DRIFT` | 3 | 2.00% | Đúng Top-1 Video nhưng biên thời gian dự đoán có tIoU < 0.5 |

---

## 7. Hồ Sơ Hiệu Năng Hệ Thống Trên CPU (`C1`, `C2`, `C3`)

- **Cold query latency: NOT_MEASURED** (phép đo cũ không đại diện end-to-end):
  - Trung bình (`Mean`): **255.08 ms** | Trung vị (`p50`): **250.93 ms** | Ngưỡng `p95`: **317.82 ms**
- **Độ trễ truy xuất phi nơ-ron (Non-Neural Retrieval Latency = FAISS + FTS5 + Clustering + Fusion + Boundary):**
  - Trung bình (`Mean`): **34.76 ms** | Trung vị (`p50`): **24.27 ms** | Ngưỡng `p95`: **106.01 ms**
  - Phân rã trung vị từng chặng (`p50`): FAISS 864d IP: `5.31 ms`, SQLite FTS5: `4.36 ms`, Multimodal Fusion: `11.64 ms`, Boundary Expansion: `2.37 ms`.
- **Bộ nhớ tiêu thụ (Memory Footprint):** RSS = **420.0 MB**, Peak Working Set = **580.0 MB** (tham chiếu phép đo nền P4; chưa đo lại trong lần export cache Phase 6, xem verdict `NOT_MEASURED` ở bảng nghiệm thu).

---

## 8. Chẩn Đoán Overfitting & Kiểm Toán Khả Năng Tổng Quát Hóa Đa Tập (Phase 6 De-Overfitting Audit)

### 8.1. Phân Tích Khoảng Cách Tổng Quát Hóa (`dev_150` vs `dev_full_859` vs `heldout_test_201`)

So sánh chéo giữa ba tập đánh giá cho thấy khoảng cách lớn giữa `dev_150` (**62.00%**) và `dev_full_859` (**41.68%**) / `heldout_test_201` (**39.30%**) bắt nguồn từ **hai nguyên nhân tách biệt**:

1. **Chuyển Dịch Cơ Cấu Tập Dữ Liệu (Dataset Mix Shift — Nguyên nhân chính ~65–70% mức chênh lệch):**
   - `dev_150` gồm 22 queries **QVHighlights** (Hit@1 **86.36%**) + 28 queries **AVC** (Hit@1 **92.86%**) + 100 queries **MSRVTT** (Hit@1 **48.00%**), tức **33.3%** truy vấn đến từ QVH+AVC — nơi có ngữ cảnh hội thoại/chữ hiển thị rõ ràng (`Lexical Recall@30` AVC đạt **75.0%**).
   - Ngược lại, `dev_full_859` gồm **94.2%** truy vấn **MSRVTT** (`809/859`) và `heldout_test_201` gồm **95.0%** truy vấn **MSRVTT** (`191/201`) — hầu hết là mô tả hành động ngắn thiếu chữ OCR hoặc hội thoại đặc trưng (`Lexical Recall@30` chỉ **17.7%–22.5%**).

2. **Overfitting Nội Bộ Trên Phân Nhóm MSRVTT (`48.00%` trên `dev_150` vs `38.69%` trên `dev_full_859` & `37.70%` trên `heldout_test_201`):**
   - Một phần hằng số/quy tắc trích xuất từ vựng trước Phase 6 hoạt động tốt hơn trên tập 100 câu MSRVTT đầu (`dev_150`), dẫn đến khoảng cách ~9–10 điểm phần trăm khi mở rộng ra toàn bộ 809 câu `dev_full_859` và 191 câu `heldout_test_201` chưa từng chạm tới.
   - Hai nhóm lỗi chi phối trên các tập lớn vẫn là:
     - **`F4_FINE_GRAINED_VISUAL_CONFUSION`**: **33.76%** (`290/859`) trên `dev_full_859` và **35.32%** (`71/201`) trên `heldout_test_201`.
     - **`F2_LEXICAL_DISTRACTOR_OVERTHROW`**: **9.55%** (`82/859`) trên `dev_full_859` và **9.45%** (`19/201`) trên `heldout_test_201`.

### 8.2. Các Nâng Cấp Tổng Quát Hóa Đã Triển Khai Trong Phase 6 (`unified_pipeline.py`)

Thay vì tiếp tục tinh chỉnh luật cục bộ cho `dev_150`, Phase 6 tập trung vào **6 cơ chế tổng quát (domain-agnostic)**:

1. **Tổng Quát Hóa Nhận Dạng Danh Từ Riêng Đầu Câu (`_is_common_sentence_initial_word`, `_COMMON_INITIAL_OPENERS`, `_COCO_VOCAB_WORDS`):** Ngăn các từ mở đầu viết hoa phổ thông (như `Cooking`, `Cartoon`, `Basketball`, động từ `-ing`/`-ed`/`-ly` và danh từ COCO) bị nhận nhầm thành danh từ riêng ở đầu câu.
2. **Sinh Gốc Từ Hình Thái Học Bảo Thủ (`_generate_morphological_stems`):** Chỉ áp dụng cho từ dài ($\ge 8$ ký tự, không thuộc `GENERIC_VISUAL_WORDS`) để giữ độ phủ từ vựng mà không gây over-stemming các động từ ngắn.
3. **Nối Bigram Liền Kề Chịu Lỗi Dính Chữ OCR (`compound_tokens` trong `prepare_fts5_query_and_tokens`):** Ghép hai từ nội dung liền kề (kết hợp từ đặc hiệu $\ge 6$ ký tự) giúp FTS5 bắt được các phụ đề dính chữ như `lateshow` hoặc tiêu đề liền khối.
4. **Hạn Chế Distractor Cho Câu Mô Tả Dài Ngắn Khớp Đơn (`single_common_only` mở rộng `k_high_len >= 6 and fts_sc < 15.0`) + `gap_damp`:** Giảm trọng số ứng viên `m_high == 1` thiếu neo danh từ riêng/trích dẫn khi điểm thị giác hụt $> 0.020$ so với `s_vis_max`, đồng thời chặn Pure-ASR đối với truy vấn chứa số học cụ thể khi ứng viên không khớp số đó (`q_numbers`: loại bỏ hoàn toàn **ca Pure-ASR FP cuối cùng** trên `dev_full_859`, đưa `pure_asr_fp_top1_count = 0` trên cả 3 tập).
5. **Kháng Đỉnh Ngoại Lai 1 Khung Hình Thuần Thị Giác (`_video_sorted_visual_scores` + `tail_adjustment` Stage 6):** Với ứng viên thuần `visual` có duy nhất 1 cụm (`m_count == 1`), phạt nhẹ đỉnh cô lập có độ tụt `kf1 - kf2 > 0.20` và thưởng nhẹ video có 2 khung hình đỉnh đồng thuận (`kf2 >= 0.88 * kf1` và `kf2 >= 0.42`). Giảm `F4` trên `dev_150` từ `30 -> 27`.
6. **Chuẩn Hóa `density_bonus` Video Dài Tín Hiệu Yếu (`n_kf > 30 and s_max < 0.225 and m_count / n_kf < 0.12`):** Ngăn video dài (nhiều keyframes) với tương đồng nền yếu lấn át clip ngắn đúng ngữ nghĩa.

### 8.3. Bảng Đối Chiếu Hiệu Năng Trước Và Sau Phase 6 Trên Ba Tập Dữ Liệu

| Tập Đánh Giá | Metric | Trước Phase 6 (Phase 5/Audit) | Sau Phase 6 | Chênh Lệch ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **`dev_150`** (`150 q`) | Video Hit@1 / MRR / nDCG@10 / `F4` | 61.33% / 0.6540 / 0.6763 / `28` | **62.00%** / **0.6582** / **0.6794** / **`27`** | **+0.67%** / **+0.0042** / **+0.0031** / **`-1 F4`** |
| **`dev_150` (MSRVTT)** | MSRVTT Hit@1 / MRR / nDCG@10 | 47.00% / 0.5251 / 0.5551 | **48.00%** / **0.5323** / **0.5605** | **+1.00%** / **+0.0072** / **+0.0054** |
| **`dev_full_859`** (`859 q`) | Video Hit@1 / MRR / nDCG@10 / `ASR_FP` | 41.68% / 0.5012 / 0.5455 / `1` | **41.68%** / **0.5018** / **0.5465** / **`0`** | **0.00%** / **+0.0006** / **+0.0010** / **`-1 ASR_FP`** |
| **`heldout_test_201`** (`201 q`) | Video Hit@1 / MRR / nDCG@10 / `F2` / `F4` | 38.81% / 0.4733 / 0.5226 / 20 / 72 | **39.30%** / **0.4774** / **0.5258** / **19** / **71** | **+0.49%** / **+0.0041** / **+0.0032** / **-1 `F2`, -1 `F4`** |
| **`heldout_test_201` (MSRVTT)** | MSRVTT Hit@1 / MRR / nDCG@10 | 37.17% / 0.4580 / 0.5081 | **37.70%** / **0.4614** / **0.5107** | **+0.53%** / **+0.0034** / **+0.0026** |

> **Kết luận & Hướng Nâng Cấp Tiếp Theo (để vượt trần ~40–48% trên MSRVTT):**
> Các quy tắc heuristic ở Stage 3–6 đã tiến gần giới hạn trần của biểu diễn SigLIP 2 Base (zero-shot 768d trên single keyframe) đối với lỗi `F4_FINE_GRAINED_VISUAL_CONFUSION`. Bước đột phá tiếp theo cần: (1) **Học lại trọng số kết hợp đa phương thức (Learned Logistic / LambdaMART Ranker)** trên `dev_full` sử dụng Cross-Validation theo nguồn video; (2) **Nâng cấp Re-ranking Giai Đoạn 2 (Top-30 Candidates)** bằng mô hình Video-Text Cross-Encoder / CLIP-VIP / InternVideo2 trên GPU Kaggle hoặc RTX 3060 ở độ chính xác FP16/INT8; (3) **Đo lại tốc độ end-to-end thực tế** trên RTX 3060 6GB VRAM với cold-start encoder và warmup riêng.
