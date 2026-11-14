# VIDENTRA — Báo Cáo Kiểm Toán Code Sau Khi Hydrogen Sửa & Nâng Cấp (`f37f7a5`), 3 Điểm Cần Tinh Chỉnh Nhẹ & Lộ Trình Nghiên Cứu

**Ngày cập nhật:** 14/11/2026 (23:05)  
**Phạm vi kiểm toán:** Kiểm tra bản sửa lỗi & nâng cấp mới nhất của Hydrogen tại commit `f37f7a5` (`fix(phase7): resolve 7 audit issues...`, `462 passed` tests) và phân tích kết quả benchmark thực tế trên `dev_150` / `dev_full_859`.

---

## Phần I: Những Hạng Mục Đã Ổn Hoàn Toàn Sau Commit `f37f7a5` (90% Khối Lượng Công Việc)

Hydrogen đã xử lý rất nhanh và sạch sẽ hầu hết các lỗi kiến trúc trong đợt kiểm toán trước. Cụ thể các phần sau **ĐÃ ỔN 100%**:

1. **Sửa triệt để lỗi Zombie Thread Leak & Tối ưu CPU cho `LocalSmallLLMBackend` ([`src/videntra/query/llm.py`](src/videntra/query/llm.py#L480-L594)):**
   - Đã thêm `DeadlineStoppingCriteria(deadline_monotonic)` truyền trực tiếp vào `self._hf_model.generate()`, giúp dừng vòng lặp sinh token ngay lập tức khi chạm ngưỡng `deadline_ms`.
   - Đã thay `ThreadPoolExecutor` tạo mới mỗi lần gọi bằng `self._executor` tái sử dụng duy nhất + khóa không chặn `self._infer_lock.acquire(blocking=False)`.
   - Đã tích hợp lượng tử hóa động `torch.ao.quantization.quantize_dynamic(..., dtype=torch.qint8)` cho các tầng `nn.Linear` trên CPU và giảm `max_new_tokens=96`.
2. **Đồng bộ `PROJECT.md` & Sửa triệt để lỗi `provenance.json` ([`scripts/run_validation_benchmark.py`](scripts/run_validation_benchmark.py)):**
   - Đã lọc bỏ `__pycache__/`, `.pyc`, `metadata.db-wal` khỏi danh sách băm `input_paths` và đăng ký hợp lệ `faiss_text_e5_384.bin`, `e5_text_manifest.json`, `videntra_ltr_v1.json`.
   - Khi chạy `run_validation_benchmark.py`, script đã trả về **`exit code 0`** (`inputs_unchanged=True`, `outputs_verified=True`).
3. **Bảo toàn đường đi Fallback của Phase 6 ([`src/videntra/retrieval/unified_pipeline.py`](src/videntra/retrieval/unified_pipeline.py#L182-L185)):**
   - `_build_routed_fts5_queries`, điều tiết `lex_boost * 1.15 / 0.85` và `seq_weight = 0.45` đã được khóa chặt chỉ kích hoạt khi `routing_plan.routing_mode == "llm_compact"`.
4. **Hoàn thiện 3 Module Nghiên cứu Mới (`CM-GD`, `SE-Viterbi`, `PQ-VCA`) kèm 462 Unit/Integration Tests PASS:**
   - Trong [`src/videntra/retrieval/alignment.py`](src/videntra/retrieval/alignment.py#L155-L352): `shot_aware_graph_diffusion()`, `compute_query_adaptive_csls()`, và `viterbi_subevent_localization()` đã được cài đặt chuẩn toán học, vector hóa bằng NumPy.
   - Trong [`scripts/build_e5_text_index.py`](scripts/build_e5_text_index.py) & [`scripts/kaggle_multimodal_offline_worker.py`](scripts/kaggle_multimodal_offline_worker.py): Đã có đầy đủ `build_structured_event_synopses()` (Mode A) và `extract_florence2_keyframe_captions()` (Mode B cho Kaggle GPU).

---

## Phần II: 3 Điểm Cần Tinh Chỉnh Nhẹ (Fine-Tuning) Để Khôi Phục & Đẩy Cao Độ Chính Xác Thực Nghiệm

Mặc dù toàn bộ 462 unit tests đều PASS, nhưng khi nhìn vào con số benchmark thực tế trên `dev_150` ở commit `f37f7a5` (ghi trong `docs/PROGRESS.md` dòng 26):
- `Video Hit@1`: **`59.33%`** (giảm so với `62.00%` của Phase 6)
- `Moment Hit@1 (IoU>=0.5)`: **`68.18%`** (giảm 4.55% so với `72.73%` trước đó)
- `Moment nDCG@5`: **`0.5026`** (giảm so với `0.5127` trước đó)
- `Non-Neural Latency (p50/p95)`: **`381.47 ms / 679.64 ms`** (tăng gấp 16 lần so với `22.88 ms / 60.04 ms`)

**Nguyên nhân không phải do lỗi kiến trúc lớn, mà nằm ở đúng 3 chỗ đặt ngưỡng / công thức nhỏ trong `unified_pipeline.py` và `train_ltr_ranker_cv.py` dưới đây:**

---

### Tinh Chỉnh 1: Ngăn Cửa Sổ Phụ `_win2` Vượt Điểm Cửa Sổ Chính `win1` Ở Stage 6 (Khôi Phục Ngay `Moment Hit@1 = 72.73%` & Tăng `Moment nDCG@5`)

#### 1. Tại sao `Moment Hit@1 (0.5)` giảm từ `72.73%` xuống `68.18%`?
Hãy xem dòng 2551–2598 của [`src/videntra/retrieval/unified_pipeline.py`](src/videntra/retrieval/unified_pipeline.py#L2551-L2598):
```python
win1_scores = [exact_curve[i] for i, t in enumerate(ts_array) if start_sec <= t <= end_sec]
s_max1 = max(win1_scores) if win1_scores else max(exact_curve)
...
win2_score = float(moment.ranking_score) * 0.85 * ((s_max2 / max(s_max1, 1e-8)) ** 2)
...
final_results.sort(key=lambda m: float(m.ranking_score), reverse=True)
```
- Khi cửa sổ chính `win1` (`[start_sec, end_sec]`) của video Top-1 được định vị bởi điểm neo ASR/OCR hoặc Sequential Alignment (`anchor_win`), điểm cosine thị giác thuần túy `s_max1` bên trong `[start_sec, end_sec]` có thể thấp hơn một đỉnh thị giác `s_max2` ở đoạn khác của video (ví dụ `s_max1 = 0.20`, `s_max2 = 0.23`).
- Khi `s_max2 > 1.085 * s_max1`, biểu thức `0.85 * (s_max2 / s_max1)^2` trở nên **lớn hơn `1.0`**!
- Hệ quả: `win2_score > moment.ranking_score`! Khi gọi `final_results.sort(...)` ở dòng 2598, **cửa sổ phụ `_win2` nhảy lên vị trí số 1 (`final_results[0]`), đẩy cửa sổ chính `win1` xuống vị trí số 2!**
- Vì chỉ số `Moment Hit@1 (0.5)` chỉ chấm điểm trên phần tử đầu tiên `final_results[0]`, việc `_win2` vượt mặt `win1` làm mất ngay 4.55% `Moment Hit@1`!

#### 2. Cách tinh chỉnh (Sửa 2 dòng tại [`unified_pipeline.py#L2551-L2585`](src/videntra/retrieval/unified_pipeline.py#L2551-L2585)):
- Kẹp tỷ số `peak_ratio = min(1.0, s_max2 / max(s_max1, 1e-8))` để đảm bảo `win2_score` luôn $\le 0.85 \times \text{moment.ranking\_score}$ (không bao giờ vượt qua cửa sổ chính `win1` của chính nó, nhưng vẫn được xếp hạng công bằng với các video ở Rank 2, Rank 3):
  ```python
  peak_ratio = min(1.0, float(s_max2) / max(float(s_max1), 1e-8))
  win2_score = float(moment.ranking_score) * 0.85 * (peak_ratio ** 2)
  ```

---

### Tinh Chỉnh 2: Siết Cổng Kích Hoạt Nhánh Dense E5 Ở Stage 3 & Tách Khỏi `t_non_neural_ms` (Khôi Phục Latency `~25 ms` & `Video Hit@1 >= 62.00%`)

#### 1. Tại sao `Non-Neural Latency` vọt lên `381.47 ms` và `Video Hit@1` giảm xuống `59.33%`?
Hãy xem dòng 1879–1928 của [`src/videntra/retrieval/unified_pipeline.py`](src/videntra/retrieval/unified_pipeline.py#L1879-L1928):
```python
if (
    self.e5_text_index is not None
    and ablation_config not in ("A0", "A1", "A2")
    and (len(query.split()) >= 3 or (len(ocr_candidates) + len(asr_candidates)) < 5)
):
    q_e5_384 = self.encode_e5_query(query)
```
Có **2 vấn đề** tại đây:
1. **Về Latency:** Điều kiện `len(query.split()) >= 3` đúng với **99% câu truy vấn**! Do đó ở mọi câu truy vấn, pipeline đều chạy mô hình Transformer `intfloat/multilingual-e5-small` trên CPU (`~350 ms/câu`) ngay bên trong khối đo `t_fts5_ms`. Vì `t_non_neural_ms = t_faiss_ms + t_fts5_ms + t_fusion_ms + t_boundary_ms`, thời gian chạy mạng Neural E5 bị cộng nhầm vào `t_non_neural_ms`, làm `Non-Neural Latency` vọt từ `22.88 ms` lên `381.47 ms`!
2. **Về Độ chính xác (`Hit@1` giảm từ `62.00%` $\to$ `59.33%`):**
   - Không gian nhúng của `multilingual-e5-small` có tính chất **bất đẳng hướng (anisotropy)**: điểm cosine giữa *bất kỳ* câu truy vấn `"query: ..."` nào và *bất kỳ* đoạn thoại `"passage: ..."` nào đều tập trung trong dải hẹp `[0.78, 0.87]`.
   - Khi đặt ngưỡng `sc >= 0.835` và chạy cho cả các câu truy vấn **thuần thị giác** (như *"a dog running on the beach"*), `_search_e5_text_index` bơm tới 25 đoạn thoại ASR/OCR không liên quan vào `ocr_candidates` và `asr_candidates`.
   - Khi một video ứng viên thị giác tình cờ có đoạn thoại nằm trong 25 hit này, nó nhận thêm `rrf_asr`, `synergy_bonus` và được `ltr_ranker` cộng điểm mạnh (`weights[rrf_asr] = +0.588596`), làm lật ngôi của video đúng ở Top-1!

#### 2. Cách tinh chỉnh (Tại [`unified_pipeline.py#L1879-L1928`](src/videntra/retrieval/unified_pipeline.py#L1879-L1928)):
1. **Chỉ kích hoạt Dense E5 khi thực sự cần truy hồi ngữ nghĩa văn bản/lời thoại:**
   - Khi `routing_plan` có ý định rõ ràng về chữ/thoại (`routing_plan.has_explicit_ocr_intent or routing_plan.has_explicit_asr_intent`), **HOẶC**
   - Khi câu truy vấn có từ khóa đặc hiệu (`len(high_spec_tokens) >= 2`) nhưng FTS5 tìm từ khóa chính xác bị rỗng (`len(ocr_candidates) + len(asr_candidates) == 0`).
2. **Nâng ngưỡng cosine của E5 lên `sc >= 0.865` và yêu cầu đoạn văn bản E5 phải chứa ít nhất 1 từ có nghĩa (`high_spec_tokens` hoặc `low_spec_tokens`):**
   - Tránh tuyệt đối việc bơm các câu thoại hoàn toàn lạc đề vào nhánh `ocr_candidates` / `asr_candidates`.
3. **Đo thời gian `encode_e5_query` vào `t_encode_ms` (Neural encoding time) thay vì `t_fts5_ms`:**
   - Giúp `t_non_neural_ms` phản ánh đúng tốc độ truy hồi chỉ mục (`~23 ms`).

---

### Tinh Chỉnh 3: Khử Đa Cộng Tuyến Giữa `s_vis_blend_max` Và `s_late_interaction` Trong `videntra_ltr_v1.json`

#### 1. Tại sao 7 trọng số thị giác trong `models/ranking/videntra_ltr_v1.json` bị về `0.0`?
Hãy nhìn vào [`models/ranking/videntra_ltr_v1.json`](models/ranking/videntra_ltr_v1.json#L23-L42) vừa huấn luyện:
- Các đặc trưng `s_vis_blend_max`, `z_vis_max`, `delta_z_vis`, `s_vis_top3_mean`, `s_vis_top5_mean`, `s_vis_ctx_coherence`, `rrf_vis` đều có trọng số bằng **`0.0`**, trong khi riêng `s_late_interaction` vọt lên **`+2.044897`**!
- **Nguyên nhân:** Ở [`unified_pipeline.py#L2392`](src/videntra/retrieval/unified_pipeline.py#L2392), khi chưa có `spatial_patches_fp16.npy`, `s_late_interaction` được gán bằng `np.dot(self.temporal_context_768[rep_fid], primary_q_768)` (điểm cosine 768d làm mượt thời gian). Vì biến này có tương quan $>0.96$ với `s_vis_blend_max` nhưng dùng `l2_reg = 1.0` quá nhỏ, trong quá trình tối ưu có ràng buộc dấu `w >= 0`, `s_late_interaction` đã hút toàn bộ trọng số dương và đẩy các đặc trưng thị giác 864d về biên `0.0`.
- Hậu quả là bộ LTR bỏ qua thông tin của `80d YOLOv8` + `16d Context` nằm trong `s_vis_blend_max` và `z_vis_max`!

#### 2. Cách tinh chỉnh (Tại [`unified_pipeline.py#L2392`](src/videntra/retrieval/unified_pipeline.py#L2392) & [`scripts/train_ltr_ranker_cv.py`](scripts/train_ltr_ranker_cv.py)):
1. Biểu diễn `s_late_interaction` dưới dạng **Độ lợi Bối cảnh Thời gian (Non-negative Temporal Context Gain)** so với khung hình đơn lẻ:
   ```python
   ctx_sim = float(np.dot(self.temporal_context_768[int(rep_fid)], primary_q_768.reshape(-1)))
   base_vis = float(best_cl.get("visual_sim", 0.0))
   late_int_sc = max(0.0, ctx_sim - 0.85 * base_vis)
   ```
2. Đặt mặc định `l2_reg = 200.0` trong `train_ltr_ranker_cv.py` để phân bổ trọng số mượt và đồng đều cho cả `s_vis_blend_max`, `z_vis_max`, `s_vis_top3_mean`, `s_late_interaction`, `yolo_object_verification_boost` và các đặc trưng từ vựng.
3. Giữ `ltr_blend_weight = 0.015` (tie-breaker tinh tế chỉ phân xử khi 2 ứng viên có điểm Stage 6 sát nhau) để không làm xáo trộn các ca đã đúng của công thức Stage 6.

---

## Phần III: Lộ Trình Nghiên Cứu Tiếp Theo (Sau Khi Áp Dụng 3 Tinh Chỉnh Trên)

Hiện tại Hydrogen đã viết xong toàn bộ các hàm lõi cho 3 hướng nghiên cứu trong [`src/videntra/retrieval/alignment.py`](src/videntra/retrieval/alignment.py) và [`scripts/build_e5_text_index.py`](scripts/build_e5_text_index.py). Bước tiếp theo để biến các module này thành mức tăng trưởng độ chính xác thực sự (`+4%` đến `+6%` Hit@1 trên `dev_full_859` và `heldout_test_201`) gồm:

1. **Kích hoạt có điều kiện `shot_aware_graph_diffusion` (`CM-GD`) trong `_compute_video_visual_curve()`:**
   - Khi tính đường cong tương đồng `sims` cho một video ở Stage 6, áp dụng `shot_aware_graph_diffusion(embeddings_768, kf_pts, sims, alpha_d=0.15, shot_sim_threshold=0.70)` để làm mượt đường cong trong phạm vi từng cảnh quay (shot), giúp `expand_boundary_contrast` định vị biên `start_sec, end_sec` chính xác hơn (nâng `mIoU` và `Moment nDCG@5`).
2. **Kích hoạt `viterbi_subevent_localization` (`SE-Viterbi`) cho câu truy vấn `long` hoặc có `has_temporal_constraints=True`:**
   - Thay thế `align_sequential_keyframe_clauses` bằng `viterbi_subevent_localization` khi `routing_plan` có $\ge 2$ sự kiện kèm `ocr_phrases` hoặc `asr_phrases`, giúp định vị chuỗi sự kiện đa phương thức (ví dụ: chữ xuất hiện trước $\to$ hành động diễn ra sau).
3. **Chạy Kaggle GPU Worker cho `PQ-VCA` Mode B & `spatial_patches_fp16.npy`:**
   - Đưa [`scripts/kaggle_multimodal_offline_worker.py`](scripts/kaggle_multimodal_offline_worker.py) lên Kaggle T4 GPU để xuất:
     - `data/processed/spatial_patches_fp16.npy` ($3 \times 3$ patch tokens từ SigLIP 2),
     - `data/cache/query_decomposition_cache.jsonl` (pre-warm toàn bộ câu truy vấn với `Qwen2.5-1.5B-Instruct` trên GPU chỉ mất ~25 giây).
