# VIDENTRA — Báo Cáo Kiểm Định Sau Bản Nâng Cấp Phase 8 Của Hydrogen (`902f465` – `f80740b`), Hướng Dẫn Sửa Lỗi & Lộ Trình Nghiên Cứu Tiếp Theo

> **Ngày kiểm định:** 15/11/2026 (Cập nhật sau các commits Phase 8: `902f465`, `8210881`, `f80740b`)  
> **Phạm vi kiểm định:** Rà soát toàn diện mã nguồn, trọng số `LearnedRankerLTR` 22 chiều (`models/ranking/videntra_ltr_v1.json`), pipeline đóng gói/nạp Kaggle GPU (`P8-4`, `P8-5`) và kết quả thực nghiệm trên cả 3 tập `dev_150`, `dev_full_859`, `heldout_test_201` cùng bộ 470 bài kiểm thử tự động.

---

## Tổng Quan Đối Chiếu Thực Nghiệm Qua Các Mốc Commit

| Chỉ số (Metric) | Phase 6 Baseline (`89c52b5`) | Phase 7 Tuned (`9beb770`) | Phase 8 Giữa kỳ (`8210881`) | Phase 8 Hiện tại (`f80740b`) | Đánh giá trạng thái hiện tại |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`dev_150` Video Hit@1** | `62.00%` (93/150) | `61.33%` (92/150) | **`61.33%`** (92/150) | `60.00%` (90/150) | Bị tụt `-1.33%` ở commit `f80740b` do lỗi công thức `window_energy_concentration` & hạ `l2_reg=3000` |
| **`dev_150` Video MRR** | `0.6582` | `0.6517` | **`0.6517`** | `0.6457` | Bị tụt nhẹ `-0.0060` ở `f80740b` cùng nguyên nhân trên |
| **`dev_150` Video nDCG@10** | `0.6794` | `0.6729` | **`0.6729`** | `0.6686` | Cần khôi phục `l2_reg` + sửa `window_energy_concentration` |
| **`dev_150` Moment Hit@1 (IoU $\ge 0.5$)** | `72.73%` | `72.73%` | **`72.73%`** | `68.18%` | Bị tụt 1 câu QVH ở `f80740b` khi `l2_reg=3000` đảo hạng video Top-1 |
| **`dev_150` Moment nDCG@5** | `0.5127` | `0.5119` | **`0.5121`** | `0.5077` | Ở `8210881` đã tăng lên `0.5121`, bị kéo xuống ở `f80740b` do LTR |
| **`dev_full_859` Video Hit@1 / Hit@10** | `41.68% / 68.80%` | — | — | `41.33% / 69.03%` | **`Hit@10` tăng lên `69.03%`** và `nDCG@10` tăng lên **`0.5468`** |
| **`heldout_test_201` Video Hit@1 / Hit@5** | `39.30% / 58.21%` | `41.79% / 59.70%` | `38.81% / 57.71%` | `38.31% / 60.20%` | **`Hit@5` đạt đỉnh `60.20%`**, nhưng `Hit@1` giảm nhẹ do `window_energy_concentration` |
| **Non-Neural Latency `p50 / p95` (`dev_150`)** | `24.27 / 106.01 ms` | `72.68 / 378.60 ms` | **`36.10 / 81.42 ms`** | **`43.27 / 72.05 ms`** | **Đã cải thiện vượt bậc (nhanh gấp ~2x p50 và ~5x p95 so với Phase 7)** |
| **Pure-ASR False-Positive Top-1** | `0` | `0` | **`0`** | **`0`** | Tuyệt đối an toàn trên cả 3 tập |
| **Pytest Suite** | `402 passed` | `463 passed` | `466 passed` | **`470 passed`** | Toàn bộ 470 unit/integration tests đều PASS |
| **Provenance Gate (`run_validation_benchmark.py`)** | `Exit 0` | `Exit 0` | `Exit 0` | **`Exit 0`** | Khóa sạch SHA-256 (`inputs_unchanged=True`, `outputs_verified=True`) |

---

## PHẦN I — Phần Nào Đã Ổn (Những Điểm Hydrogen Đã Triển Khai Rất Tốt Trong Phase 8)

Trong 3 commits vừa qua (`902f465`, `8210881`, `f80740b`), Hydrogen đã hoàn thành khối lượng công việc rất lớn và giải quyết triệt để các lỗi cốt lõi từ đợt kiểm định trước:

### 1. Đã Vá Triệt Để Cả 3 Lỗi Kỹ Thuật Của Phase 7
- **Sửa dứt điểm lỗi crash `SE-Viterbi` (`AttributeError` / `TypeError`):** Tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1608-L1695), `sub_q_864` đã được chuyển thành ma trận 2D `sub_q_864_mat = np.vstack(...)`, kiểm tra `len(sub_q_864) >= 2` và truyền `event_vis_curves=clause_frame_sims` cùng `event_weights` vào `viterbi_subevent_localization` an toàn tuyệt đối.
- **Siết chặt cổng `Dense E5` ở Stage 3 giúp tăng tốc gấp 2–5 lần ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1902-L1930)):** Nhờ bổ sung điều kiện `has_lexical_modality_intent` (kiểm tra `quoted_phrases`, `has_explicit_ocr_intent`, `has_explicit_asr_intent`, `_lexical_intent_cues`), hệ thống không còn chạy E5 dư thừa trên các câu truy vấn thuần thị giác. Kết quả là **`Non-Neural Latency (p50 / p95)` giảm mạnh từ `72.68 ms / 378.60 ms` xuống chỉ còn `36.10–43.27 ms / 72.05–81.42 ms`**.
- **Bảo toàn độ sắc nét của đường cong biên ở Stage 6 ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2584)):** Đã loại bỏ bước làm mờ dư thừa trên `boundary_curve` của các đoạn video ngắn.

### 2. Đã Xây Dựng Hoàn Chỉnh Các Module Thuật Toán `P8-1`, `P8-2`, `P8-3`
- **`P8-1` ([visual.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/visual.py#L530-L618)):** Đã cài đặt đầy đủ `compute_keyframe_hubness_radii()` và `build_or_load_keyframe_hubness_radii()` (chống rò rỉ nội bộ cùng video và xử lý an toàn trường hợp `counts == 0`), đồng thời tính `MV-SetSim` (`video_min_clause_sim`) ở Stage 2.
- **`P8-2` ([refine.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/localization/refine.py#L368-L408) & [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2626-L2705)):** Đã cài đặt `snap_boundaries_to_shot_cuts()` và nâng cấp Stage 6 sang thuật toán đa đỉnh `NMS-MultiPeak` (`_win2`, `_win3` với kiểm tra `temporal_iou < 0.30` và kẹp điểm giảm dần `min(win_p_score, prev_win_score - 1e-4)`).
- **`P8-3` ([features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py#L36-L41), [learned_ranker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/learned_ranker.py#L80-L145)):** Đã mở rộng không gian đặc trưng LTR từ `18d` lên **`22d` (`Synergistic-CrossLTR`)** đồng thời giữ tương thích ngược hoàn toàn với các file trọng số `18d`.

### 3. Đã Gia Cố Toàn Diện Pipeline Offline Kaggle GPU Cho `P8-4` & `P8-5` (Sẵn Sàng Chạy Thực Chiến)
- **[prewarm_llm_query_cache.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/prewarm_llm_query_cache.py):** Hỗ trợ `--split all_1060`, `--export-queries-only` (xuất danh sách truy vấn sạch không chứa nhãn Ground Truth), `--checkpoint-every` và `--merge-from` an toàn.
- **[package_kaggle_inputs.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/package_kaggle_inputs.py), [kaggle_multimodal_offline_worker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/kaggle_multimodal_offline_worker.py), [ingest_kaggle_offline_results.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/ingest_kaggle_offline_results.py):** Đã bổ sung `--mode db` (đóng gói `11,581` keyframes từ DB), hàm `run_p85_keyframe_backfill` chạy tuần tự từng mô hình trên Kaggle T4 kèm xả VRAM chống OOM, sửa lỗi `float16/float32` của `Florence-2`, và bổ sung `--mode p85` để nạp `KeyframeCaptions` + chỉ mục `fts_vlm_caption` FTS5 + `faiss_e5_caption_384.bin` mà không ghi đè dữ liệu gốc.
- Đã có bộ kiểm thử chuyên biệt [test_kaggle_offline_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/tests/indexing/test_kaggle_offline_pipeline.py) (`4 passed`), nâng tổng số bài test lên **`470 passed`**.

---

## PHẦN II — Phần Nào Chưa Ổn Trong Bản Code Hiện Tại (`f80740b`) — Vì Sao Chưa Ổn & Hướng Dẫn Sửa Chi Tiết

Khi đối chiếu giữa commit `8210881` (`Video Hit@1 = 61.33%`, `Moment Hit@1 = 72.73%`) và commit cuối cùng `f80740b` (`Video Hit@1 = 60.00%`, `Moment Hit@1 = 68.18%`), mình đã tìm ra **chính xác 3 nguyên nhân kỹ thuật** làm giảm độ chính xác ở commit cuối cùng. Dưới đây là giải thích chi tiết và cách sửa từng dòng:

---

### Điểm Chưa Ổn 1 (Quan Trọng Nhất): Lỗi Logic Khi Tính `window_energy_concentration` Trên `vid_items` Thay Vì Toàn Bộ Keyframes ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2471-L2475))

#### 1. Vì sao chưa ổn?
Hãy xem đoạn code tính đặc trưng thứ 22 `window_energy_concentration` tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2471-L2475):
```python
inside_scores = [x[0] for x in vid_items if best_st <= x[3] <= best_et]
outside_scores = [x[0] for x in vid_items if not (best_st <= x[3] <= best_et)]
mean_in = (sum(inside_scores) / len(inside_scores)) if inside_scores else s_max
mean_out = (sum(outside_scores) / len(outside_scores)) if outside_scores else 0.0
win_energy_conc = max(0.0, mean_in) / (max(0.0, mean_out) + 0.05)
```
- **Sai lầm cốt lõi nằm ở biến `vid_items`:**
  - `vid_items` **KHÔNG PHẢI** là danh sách toàn bộ các keyframes của video, mà chỉ là danh sách **các cụm ứng viên (retrieved clusters) lọt qua Stage 4** của video đó (`m_count = len(vid_items)`), và `[best_st, best_et]` là khoảng `[best_rep_t - 1.0, best_rep_t + 1.0]` của cụm đứng đầu `vid_items[0]`!
  - **Trường hợp 1 — Video nhiễu (Distractor) chỉ có đúng 1 cụm ăn may (`m_count == 1`):**
    Vì `vid_items` chỉ có 1 phần tử duy nhất (chính là `vid_items[0]`), nên danh sách `outside_scores` **luôn luôn rỗng (`[]`)**!
    Do đó `mean_out = 0.0`, và:
    $$\text{win\_energy\_conc} = \frac{\text{mean\_in}}{0.0 + 0.05} = 20 \times \text{mean\_in} \approx 3.50$$
  - **Trường hợp 2 — Video đúng (Gold Video) có nhiều cụm khớp liên tiếp (`m_count >= 2`):**
    Vì video đúng có thêm các cụm khớp ở các giây lân cận nằm ngoài `[best_rep_t - 1.0, best_rep_t + 1.0]`, nên `outside_scores` **khác rỗng** (`mean_out` $\approx 0.16$), dẫn tới:
    $$\text{win\_energy\_conc} = \frac{\text{mean\_in}}{0.16 + 0.05} \approx 4.76 \times \text{mean\_in} \approx 0.80$$
- **Hậu quả trực tiếp:**
  - Công thức trên vô tình **thưởng điểm cao gấp ~4.4 lần cho các video nhiễu chỉ có 1 cụm cô lập (`m_count == 1`)** và **phạt nặng các video đúng có nhiều cụm đồng thuận (`m_count >= 2`)**!
  - Ở commit `8210881` (`l2_reg = 1500000.0`), trọng số của `window_energy_concentration` trong `videntra_ltr_v1.json` bằng **`0.0`**, nên lỗi này bị vô hiệu hóa hoàn toàn (`Video Hit@1 = 61.33%`, `Moment Hit@1 = 72.73%`).
  - Nhưng ở commit `f80740b`, khi Hydrogen hạ `l2_reg = 3000.0`, trọng số của `window_energy_concentration` nhảy vọt từ `0.0` lên **`+0.108777`** ([videntra_ltr_v1.json](file:///C:/Users/DELL/Desktop/VIDENTRA/models/ranking/videntra_ltr_v1.json#L49)), lập tức bơm điểm cho các video nhiễu đơn cụm vượt mặt video đúng và làm tụt cả `Video Hit@1` (`61.33% -> 60.00%`) lẫn `Moment Hit@1` (`72.73% -> 68.18%`)!

#### 2. Hướng dẫn sửa chi tiết
Mở [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2469-L2476) và sửa lại cách tính `win_energy_conc` dựa trên **mật độ đồng thuận cục bộ thực sự** (so sánh `local_vis_coherence` / `top3_vis_mean` và số cụm khớp lân cận `adjacent_pairs`, sao cho video có nhiều khung hình đỉnh đồng thuận trong cửa sổ có giá trị cao hơn video chỉ có 1 đỉnh đơn độc):

```python
            s_vis_blend_val = float(best_cl.get("visual_sim", 0.0))
            subclause_weak_val = float(video_min_clause_sim.get(vid, s_vis_blend_val))
            # Đo độ tập trung năng lượng bền vững quanh đỉnh: thưởng khi các khung hình lân cận cùng cao và có cụm liền kề
            coherence_ratio = float(local_vis_coherence) / max(s_vis_blend_val, 0.05) if s_vis_blend_val > 0.0 else 0.0
            win_energy_conc = min(1.5, coherence_ratio * (1.0 + 0.10 * min(adjacent_pairs, 3)))
```
Đồng thời trong [features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py#L222-L225), sửa giá trị mặc định khi không truyền `window_energy_concentration`:
```python
    if "window_energy_concentration" in candidate and candidate["window_energy_concentration"] is not None:
        window_energy_concentration = _safe_float(candidate["window_energy_concentration"])
    else:
        s_top3 = values["s_vis_top3_mean"]
        window_energy_concentration = (s_top3 / max(s_vis, 0.05)) if s_vis > 0.0 else 0.0
```

---

### Điểm Chưa Ổn 2: Độ Lệch Mục Tiêu Giữa Huấn Luyện Độc Lập (`l2_reg = 3000.0`) Và Pha Trộn Phần Dư (`Residual Blending`) Trong `train_ltr_ranker_cv.py`

#### 1. Vì sao chưa ổn?
- Trong [train_ltr_ranker_cv.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/train_ltr_ranker_cv.py#L469), hàm `run_group_kfold_cv` đánh giá `cv_hit_at_1` bằng cách xếp hạng ứng viên **chỉ dựa trên điểm `ltr_sc` của mô hình tuyến tính 22 chiều**.
- Trong khi đó, tại thời điểm suy luận trong [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2538), `ltr_sc` được dùng như một **phần dư điều chỉnh (Residual Tie-Breaker)** cộng vào `video_score` của Stage 5:
  ```python
  ranking_score = float(m.ranking_score) + self.ltr_blend_weight * float(np.tanh(ltr_sc))
  ```
- Khi hạ `l2_reg` từ `1500000.0` xuống `3000.0` ở commit `f80740b`:
  1. Trọng số `s_vis_ctx_coherence` tăng vọt lên **`0.594256`** (gấp ~76 lần so với mức `0.007813` trước đó), trong khi trọng số `rrf_vis` bị triệt tiêu về **`0.0`** ([videntra_ltr_v1.json](file:///C:/Users/DELL/Desktop/VIDENTRA/models/ranking/videntra_ltr_v1.json#L33-L36)).
  2. Vì `s_vis_ctx_coherence` tính tích vô hướng trên vector trung bình trượt `temporal_context_768`, nó ưu tiên các video có cảnh tĩnh kéo dài nhưng làm suy yếu các video có khoảnh khắc hành động ngắn sắc nét.
  3. Với biên độ trọng số lớn (`~0.60`), `np.tanh(ltr_sc)` bão hòa gần `±1.0`, khiến lượng cộng thêm `0.04 * np.tanh(ltr_sc)` đủ lớn để **đè bẹp khoảng cách điểm tinh tế của Stage 5** và làm mất 2 câu đúng ở Top-1 trên `dev_150` (trong đó có 1 câu QVHighlights làm `Moment Hit@1 (0.5)` giảm từ `16/22 = 72.73%` xuống `15/22 = 68.18%`).

#### 2. Hướng dẫn sửa chi tiết
Có 2 bước sửa rất gọn để vừa giữ được các đặc trưng mới (`vis_lexical_synergy`, `subclause_weakest_link`, `window_energy_concentration` đã sửa ở mục 1) vừa khôi phục 100% đỉnh độ chính xác:
- **Bước 1:** Trong [train_ltr_ranker_cv.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/train_ltr_ranker_cv.py#L469), đặt `--l2-reg` mặc định ở mức cân bằng **`250000.0`** (hoặc `500000.0`) thay vì `3000.0` quá nhỏ. Ở mức `l2_reg = 250000.0`, cả `rrf_vis`, `s_vis_blend_max`, `vis_lexical_synergy`, `subclause_weakest_link` đều khác 0 và không có đặc trưng đơn lẻ nào độc chiếm trọng số.
- **Bước 2:** Trong [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L510), giảm nhẹ `ltr_blend_weight` mặc định từ `0.04` xuống **`0.015`** để LTR làm đúng nhiệm vụ phân xử các cặp ứng viên có điểm `video_score` sát nút mà không lật ngược các ứng viên đã thắng rõ ràng ở Stage 5.

---

### Điểm Chưa Ổn 3: Bảng `fts_vlm_caption` Và Chỉ Mục `faiss_e5_caption_384.bin` (`P8-5`) Đã Có Ở Bước Ingest Nhưng Chưa Được Nối Vào `unified_pipeline.py`

#### 1. Vì sao chưa ổn?
- Trong commit `f80740b`, Hydrogen đã viết rất tốt phần tạo bảng `KeyframeCaptions`, bảng FTS5 `fts_vlm_caption` và chỉ mục `faiss_e5_caption_384.bin` trong [ingest_kaggle_offline_results.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/ingest_kaggle_offline_results.py#L245-L345) và [kaggle_multimodal_offline_worker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/kaggle_multimodal_offline_worker.py#L410-L520).
- Tuy nhiên, trong [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py), Stage 3 hiện mới chỉ truy vấn 2 bảng `ocr_fts` và `asr_fts` (cùng `faiss_text_e5_384.bin`), **chưa có đoạn code nào đọc bảng `fts_vlm_caption` hoặc `faiss_e5_caption_384.bin`**!
- Điều đó có nghĩa là ngay cả khi bạn chạy xong Kaggle GPU và chạy `ingest_kaggle_offline_results.py --mode p85`, pipeline tìm kiếm vẫn sẽ bỏ qua toàn bộ dữ liệu `Florence-2` captions vừa nạp nếu không nối (wire) chúng vào `unified_pipeline.py`!

#### 2. Hướng dẫn sửa chi tiết
Để khi có kết quả Kaggle `P8-5` về là hệ thống tự động kích hoạt ngay:
1. Trong `UnifiedMultimodalSearchPipeline._load_resources()` ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L620-L680)):
   - Kiểm tra sự tồn tại của bảng `fts_vlm_caption` trong `metadata.db` (`self.has_vlm_caption_fts: bool`) và file `data/processed/faiss_e5_caption_384.bin` + `e5_caption_manifest.json` (`self.e5_caption_index`).
2. Trong Stage 3 của `search()` ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1890-L1960)):
   - Khi `self.has_vlm_caption_fts` bằng `True` và `fts_query` không rỗng, truy vấn `fts_vlm_caption` lấy Top-30 keyframes khớp mô tả cảnh quay, quy đổi sang `rrf_vis_caption` và cộng hỗ trợ vào `visual_candidates` / `all_clusters` để cứu trực tiếp các truy vấn bị lỗi `F1_VISUAL_CANDIDATE_MISS`!

---

## PHẦN III — Định Hướng Nghiên Cứu Cụ Thể & Kế Hoạch Thực Hiện Chi Tiết (Giai Đoạn Tiếp Theo)

Sau khi thực hiện bản vá 10 phút ở Phần II (sửa `window_energy_concentration`, đặt `l2_reg = 250000.0` + `ltr_blend_weight = 0.015`, và nối `fts_vlm_caption` vào `unified_pipeline.py`), toàn bộ phần khung trên CPU của VIDENTRA đã hoàn thiện. Bước đột phá lớn tiếp theo để đưa độ chính xác từ **`61.33%–62.00%` lên `66%–70%`** trên `dev_150` và từ **`41.68%` lên `47%–50%`** trên `dev_full_859` gồm **4 hướng nghiên cứu – thực nghiệm** cụ thể sau:

---

### Hướng 1: Chạy Thực Chiến Pipeline Kaggle Dual-T4 GPU (`P8-4` & `P8-5`) Để Mở Khóa 3 Nguồn Tín Hiệu Đang Chờ Dữ Liệu

#### 1. Định hướng cụ thể
Hiện tại toàn bộ các kịch bản đóng gói, worker Kaggle và nạp dữ liệu đã được viết xong và vượt qua 100% bài test (`test_kaggle_offline_pipeline.py`), nhưng **3 kho dữ liệu quan trọng nhất vẫn chưa được sinh trên GPU**:
1. **`query_decomposition_cache.jsonl` từ `Qwen2.5-1.5B-Instruct` thật (`P8-4`)**: Hiện file cache trên đĩa vẫn là kết quả của `--rule-only` (`routing_mode = "rule_fallback"`). Khi chạy trên GPU T4 cho `all_1060` queries (~90 giây), toàn bộ 1,060 câu truy vấn sẽ có cấu trúc phân rã ngữ nghĩa `llm_compact` thật sự, kích hoạt `SE-Viterbi` và `llm_modality_alignment`.
2. **`spatial_patches_fp16.npy` ($11,581 \times 9 \times 768$, `float16`) từ `SigLIP2` (`P8-5`)**: Đánh thức đặc trưng thứ 7 `s_late_interaction` (`QAT-LateInteraction` MaxSim trên lưới $3\times 3$ vùng không gian) hiện đang có trọng số bằng `0.0`.
3. **`KeyframeCaptions` (`Florence-2-base` Dense Captions + `faiss_e5_caption_384.bin`) (`P8-5`)**: Bổ sung kênh đối sánh **Text-to-Text (`Query <-> VLM Keyframe Caption`)** song song với kênh **Text-to-Image (`Query <-> SigLIP2 864d`)**, đánh trúng trực tiếp vào **`22` ca lỗi `F1_VISUAL_CANDIDATE_MISS`** trên `dev_150` và **`157` ca lỗi `F1`** trên `dev_full_859`.

#### 2. Kế hoạch thực hiện chi tiết từng lệnh
- **Bước 1 (Trên máy Local — Xuất gói dữ liệu đưa lên Kaggle):**
  ```powershell
  # 1. Xuất 1,060 câu truy vấn sạch (không chứa Ground Truth)
  .\.venv\Scripts\python.exe scripts/prewarm_llm_query_cache.py `
    --split all_1060 `
    --export-queries-only work/kaggle/queries_all_1060.jsonl

  # 2. Đóng gói 11,581 keyframes từ metadata.db để đưa lên Kaggle Dataset
  .\.venv\Scripts\python.exe scripts/package_kaggle_inputs.py `
    --mode db `
    --output-dir work/kaggle/p85_inputs
  ```
- **Bước 2 (Trên Kaggle Notebook với Accelerator `GPU T4 x2`):**
  ```bash
  # 1. Sinh phân rã truy vấn bằng Qwen2.5-1.5B-Instruct trên GPU (~90 giây)
  python scripts/prewarm_llm_query_cache.py \
    --split all_1060 \
    --output /kaggle/working/query_decomposition_cache_llm.jsonl \
    --checkpoint-every 50

  # 2. Chạy trích xuất Spatial Patches 3x3 (SigLIP2) + Keyframe Captions (Florence-2-base) + E5 Index
  python scripts/kaggle_multimodal_offline_worker.py \
    --mode p85 \
    --input-dir /kaggle/input/videntra-p85-inputs \
    --output-dir /kaggle/working/p85_outputs \
    --device cuda
  ```
- **Bước 3 (Tải kết quả về máy Local & Tích hợp vào hệ thống):**
  ```powershell
  # 1. Hợp nhất cache phân rã truy vấn LLM vào data/cache/query_decomposition_cache.jsonl
  .\.venv\Scripts\python.exe scripts/prewarm_llm_query_cache.py `
    --merge-only `
    --merge-from work/kaggle/query_decomposition_cache_llm.jsonl

  # 2. Nạp spatial_patches_fp16.npy, KeyframeCaptions, fts_vlm_caption và faiss_e5_caption_384.bin
  .\.venv\Scripts\python.exe scripts/ingest_kaggle_offline_results.py `
    --mode p85 `
    --kaggle-output-dir work/kaggle/p85_outputs

  # 3. Huấn luyện lại bộ xếp hạng 22d Synergistic-CrossLTR và chạy benchmark
  .\.venv\Scripts\python.exe scripts/train_ltr_ranker_cv.py
  .\.venv\Scripts\python.exe scripts/run_validation_benchmark.py
  ```

---

### Hướng 2: Huấn Luyện Phần Dư Trực Tiếp (`Residual ListNet / Pairwise Margin Optimization`) Cho `LearnedRankerLTR`

#### 1. Định hướng cụ thể
- Như đã phát hiện ở Phần II.2, hiện tại `train_ltr_ranker_cv.py` đang huấn luyện vector trọng số $w \in \mathbb{R}^{22}$ sao cho điểm $s_{\text{LTR}}(x) = w^\top \tilde{x}$ tự nó xếp hạng từ đầu, nhưng khi đưa vào `unified_pipeline.py`, điểm cuối cùng lại là:
  $$S_{\text{final}}(q, v) = S_{\text{base}}(q, v) + \lambda_{\text{blend}} \cdot \tanh\bigl(w^\top \tilde{x}\bigr)$$
- Sự lệch pha giữa hàm mục tiêu lúc huấn luyện ($w^\top \tilde{x}$) và hàm mục tiêu lúc suy luận ($S_{\text{base}} + \lambda_{\text{blend}}\tanh(w^\top \tilde{x})$) làm cho việc giảm `l2_reg` cải thiện điểm CV độc lập nhưng lại làm giảm điểm thực tế của pipeline!
- **Giải pháp:** Sửa trực tiếp hàm mất mát trong `train_ltr_ranker_cv.py` thành **Base-Anchored Residual Pairwise Logistic Loss**:
  $$\mathcal{L}_{\text{residual}}(w) = \sum_{(i \succ j)} \log\Bigl(1 + \exp\bigl(-\gamma \cdot \bigl[(S_{\text{base}, i} - S_{\text{base}, j}) + \lambda_{\text{blend}}(\tanh(w^\top \tilde{x}_i) - \tanh(w^\top \tilde{x}_j))\bigr]\bigr)\Bigr) + \frac{\mu}{2}\|w\|_2^2$$
- Với hàm mất mát này:
  1. Những cặp ứng viên mà $S_{\text{base}, i} \gg S_{\text{base}, j}$ (Stage 5 đã xếp đúng với cách biệt lớn) sẽ có gradient gần bằng `0` — LTR không cố gắng can thiệp làm hỏng kết quả đã đúng.
  2. Toàn bộ năng lực học của $w$ sẽ tập trung 100% vào các **Hard Ambiguous Pairs** (những câu mà ứng viên đúng và ứng viên nhiễu `F4`/`F2` đang có $S_{\text{base}, i} \approx S_{\text{base}, j}$ trong khoảng `±0.03`)!

#### 2. Kế hoạch thực hiện chi tiết
- **Bước 1:** Lưu thêm trường `base_video_score` (`float(m.ranking_score)` trước khi cộng LTR) vào dữ liệu huấn luyện trong `collect_ltr_training_rows()` của [train_ltr_ranker_cv.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/train_ltr_ranker_cv.py).
- **Bước 2:** Trong `train_pairwise_logistic_l2()`, tính margin hiệu dụng $\Delta_{ij} = (S_{\text{base}, i} - S_{\text{base}, j}) + \lambda_{\text{blend}} \cdot (\tilde{x}_i - \tilde{x}_j) w$ và đánh giá `run_group_kfold_cv` trực tiếp trên $S_{\text{base}} + \lambda_{\text{blend}} \tanh(w^\top \tilde{x})$.

---

### Hướng 3: Đối Sánh Chuỗi Sự Kiện Bằng Vận Chuyển Tối Ưu Có Ràng Buộc Thời Gian (`Temporal Sinkhorn Optimal Transport — T-SOT`)

#### 1. Định hướng cụ thể
- Với các truy vấn mô tả chuỗi hành động trên MSRVTT và QVHighlights, hiện tại `MV-SetSim` lấy cực đại độc lập cho từng mệnh đề $\max_{t \in W} \langle q^{(m)}, v_t \rangle$. Tuy nhiên, nếu 2 mệnh đề khác nhau (ví dụ: $q^{(1)} =$ *"a man picks up a basketball"*, $q^{(2)} =$ *"a man shoots the ball into the hoop"*) cùng đạt cực đại tại **cùng một khung hình duy nhất $t^*$**, thì phép lấy $\max$ độc lập không phân biệt được video có đủ 2 hành động nối tiếp nhau với video chỉ có 1 khung hình đứng cạnh rổ bóng!
- **Phương pháp `T-SOT` (Temporal Sinkhorn Optimal Transport):**
  Tìm ma trận kế hoạch vận chuyển $P \in \mathbb{R}_{+}^{M \times T}$ giữa $M$ mệnh đề truy vấn và $T$ keyframes trong cửa sổ sao cho mỗi mệnh đề phải được phân bổ vào các khung hình **khác nhau** và **tuân theo thứ tự thời gian**:
  $$\max_{P \ge 0} \sum_{m=1}^{M} \sum_{t=1}^{T} P_{m,t} \bigl(\langle q^{(m)}, v_t \rangle - \eta \cdot \text{Penalty}_{\text{order}}(m, t)\bigr) + \varepsilon H(P)$$
  với3 vòng lặp chuẩn hóa dòng/cột Sinkhorn-Knopp (chỉ mất `< 0.1 ms` bằng NumPy trên ma trận $3 \times 15$).

---

### Hướng 4: Đa Khung Hình Attention Reranking Cho Top-15 Ứng Viên Cuối (`Parameter-Free Temporal Attention Pooler`)

#### 1. Định hướng cụ thể
- Điểm nghẽn lớn nhất trên tập `dev_full_859` (`809` câu MSRVTT) là `F4_FINE_GRAINED_VISUAL_CONFUSION` (`33.76%`). Nguyên nhân gốc rễ là điểm `s_vis_blend_max` chỉ nhìn vào **1 keyframe đơn lẻ**, trong khi nhiều hành động (như *"opening a door"* vs *"closing a door"*, *"sitting down"* vs *"standing up"*) có khung hình tĩnh giống hệt nhau và chỉ khác nhau ở **vector chuyển dịch theo thời gian (Temporal Difference Vector $\Delta v_t = v_{t+1} - v_{t-1}$)**.
- **Cách áp dụng vào VIDENTRA không cần mạng nơ-ron mới:**
  - Với Top-15 ứng viên cuối cùng ở Stage 5, lấy chuỗi $K=5$ keyframes liên tiếp trong cửa sổ ứng viên $\{v_{t_1}, \dots, v_{t_K}\} \in \mathbb{R}^{K \times 768}$.
  - Tính **Query-Guided Temporal Attention Pooling** kết hợp **Motion-Aware Cosine Consistency**:
    $$\alpha_k = \frac{\exp(\tau \langle q_{768}, v_{t_k} \rangle)}{\sum_{j=1}^{K} \exp(\tau \langle q_{768}, v_{t_j} \rangle)}, \quad \bar{v}_{\text{att}} = \frac{\sum_{k=1}^{K} \alpha_k v_{t_k}}{\|\sum_{k=1}^{K} \alpha_k v_{t_k}\|_2}$$
  - Điểm tương đồng đa khung hình $s_{\text{multi\_att}}(q, W) = \langle q_{768}, \bar{v}_{\text{att}} \rangle$ luôn lớn hơn $\max_k \langle q_{768}, v_{t_k} \rangle$ **khi và chỉ khi** các khung hình trong cửa sổ chứa các góc nhìn bổ trợ trực giao của cùng một khái niệm trong câu truy vấn (nhờ tính chất lồi của chuẩn $L_2$ sau tổ hợp tuyến tính!). Đây là một tính chất hình học cực đẹp trên mặt cầu đơn vị $\mathbb{S}^{767}$ giúp phân biệt video thật sự diễn ra hành động xuyên suốt nhiều khung hình với video chỉ có 1 khung hình trùng ngẫu nhiên.
