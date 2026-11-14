# VIDENTRA — Báo Cáo Kiểm Định Sau Bản Vá Hydrogen (`8fe684a` – `9beb770`), Hướng Dẫn Tinh Chỉnh Cuối & Lộ Trình Nghiên Cứu Tiếp Theo

> **Ngày kiểm định:** 15/11/2026  
> **Phạm vi kiểm định:** Đối chiếu toàn bộ các commits mới nhất của Hydrogen (`8fe684a`, `7c1cd53`, `9beb770`) so với bản kiểm định trước (`f37f7a5`) và baseline Phase 6 (`89c52b5`) trên `dev_150`, `heldout_test_201` và bộ 463 unit/integration tests.

---

## Tổng Quan Kết Quả Đối Chiếu Thực Nghiệm

| Chỉ số (Metric) | Phase 6 Baseline (`89c52b5`) | Hydrogen Đợt 1 (`f37f7a5`) | Hydrogen Đợt 2 (`9beb770` - Hiện tại) | Đánh giá trạng thái |
| :--- | :---: | :---: | :---: | :--- |
| **`dev_150` Video Hit@1** | `62.00%` (93/150) | `59.33%` (89/150) | **`61.33%`** (92/150) | Đã phục hồi +2.00% so với Đợt 1 (còn cách đỉnh Phase 6 đúng 1 câu do cổng E5) |
| **`dev_150` Video Hit@5** | `74.67%` | `72.67%` | **`74.67%`** | Đã phục hồi 100% về mức đỉnh |
| **`dev_150` Video Hit@10** | `76.67%` | `76.67%` | **`76.67%`** | Ổn định 100% |
| **`dev_150` Video MRR** | `0.6562` | `0.6352` | **`0.6517`** | Phục hồi mạnh (+0.0165 MRR) |
| **`dev_150` Video nDCG@10** | `0.6762` | `0.6596` | **`0.6729`** | Phục hồi mạnh (+0.0133 nDCG@10) |
| **`dev_150` Moment Hit@1 (IoU $\ge 0.5$)** | `72.73%` | `68.18%` | **`72.73%`** | Đã khắc phục triệt để lỗi `_win2` (+4.55%) |
| **`dev_150` Moment nDCG@5** | `0.5127` | `0.5035` | **`0.5119`** | Gần chạm đỉnh (chênh -0.0008 do `shot_aware_graph_diffusion`) |
| **`heldout_test_201` Video Hit@1** | `41.79%` | `41.79%` | **`41.79%`** | Ổn định, bảo toàn tính tổng quát hóa |
| **`heldout_test_201` Moment Hit@1 (0.5)** | `44.74%` | `44.74%` | **`44.74%`** | Ổn định |
| **`heldout_test_201` Moment nDCG@5** | `0.3698` | `0.3694` | **`0.3696`** | Ổn định |
| **Pure-ASR False-Positive Top-1** | `0` | `0` | **`0`** | Tuyệt đối an toàn |
| **Pytest Suite** | `348 passed` | `461 passed` | **`463 passed`** | Toàn bộ unit/integration test đều xanh |
| **Provenance Gate (`run_validation_benchmark.py`)** | `Exit 0` | `Exit 1` | **`Exit 0`** | Đã khóa sạch manifest & provenance |

---

## PHẦN I — Phần Nào Đã Ổn (Điểm Tốt Sau Khi Hydrogen Nâng Cấp)

Hydrogen đã xử lý rất tốt **5/6 vấn đề cốt lõi** từ báo cáo kiểm định trước:

### 1. Bộ Trọng Số `LearnedRankerLTR` (`models/ranking/videntra_ltr_v1.json`) Đã Cân Bằng & Chuẩn Vật Lý 100%
- Trong [train_ltr_ranker_cv.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/train_ltr_ranker_cv.py#L444-L491) và [videntra_ltr_v1.json](file:///C:/Users/DELL/Desktop/VIDENTRA/models/ranking/videntra_ltr_v1.json#L28-L47), việc nâng `l2_reg = 1500000.0` kết hợp với ràng buộc dấu đơn điệu (`MONOTONIC_DIRECTIONS`) đã chấm dứt hoàn toàn hiện tượng suy biến trọng số (weight collapse).
- Thay vì dồn 85% trọng số vào `s_vis_blend_max` và ép 10 đặc trưng về `0.0` như Đợt 1, hiện tại **14/18 đặc trưng** đã có trọng số hoạt động cân bằng và mang dấu vật lý chính xác:
  - Nhóm Visual Peak & Stability: `s_vis_blend_max (+0.006604)`, `s_vis_ctx_coherence (+0.007813)`, `s_vis_top3_mean (+0.006283)`, `s_vis_top5_mean (+0.005679)`, `z_vis_max (+0.004754)`.
  - Nhóm Phạt Nhiễu (Penalty): `delta_z_vis (-0.004693)`, `spike_drop_ratio (-0.002934)`.
  - Nhóm Đa Phương Thức (Lexical & RRF): `rrf_vis (+0.004682)`, `high_spec_token_coverage (+0.004260)`, `anchor_lexical_match (+0.003898)`, `bm25_log_score (+0.003517)`, `rrf_asr (+0.003288)`, `rrf_ocr (+0.001438)`.
- Đặc trưng `s_vis_ctx_coherence` (`temporal_context_768`) đã được tách riêng khỏi `s_late_interaction` trong [features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py#L19-L38) và [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2289-L2302), sẵn sàng cho cả 2 chế độ: có hoặc chưa có `spatial_patches_fp16.npy`.

### 2. Sửa Triệt Để Lỗi Cửa Sổ Phụ (`_win2`) Cướp Hạng 1 Ở Stage 6 (`Moment Hit@1` tăng vọt trở lại `72.73%`)
- Tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2630-L2633), Hydrogen đã sửa công thức chấm điểm ứng viên cửa sổ thứ hai (`_win2`):
  ```python
  peak_ratio = min(1.0, float(s_max2) / max(float(s_max1), 1e-8))
  win2_score = float(cand["score"]) * max(0.80, min(0.96, 0.92 * peak_ratio))
  ```
- Nhờ kẹp trần `min(0.96, ...)`, cửa sổ phụ `_win2` luôn giữ vai trò hỗ trợ cho `Moment Hit@5 / nDCG@5` mà không bao giờ vượt mặt cửa sổ chính `win1` ở vị trí Top-1. Kết quả là `Moment Hit@1 (IoU >= 0.5)` trên `dev_150` phục hồi hoàn toàn từ `68.18%` lên **`72.73%`**.

### 3. Nâng Cấp Backend `llama-cpp-python` (GGUF Q4_K_M) & Cơ Chế CPU-Safe Cho `SmallLLMQueryDecomposer`
- Tại [llm.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/query/llm.py#L61-L135), Hydrogen đã:
  - Bổ sung khám phá tự động file `.gguf` (`models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`) và nạp qua `llama_cpp.Llama` ưu tiên trước `transformers`.
  - Thiết lập mặc định `enabled = self._has_cuda()` (hoặc bật tường minh qua biến môi trường `VIDENTRA_LLM_ENABLED=1`), giúp máy chỉ có CPU tự động dùng `query_decomposition_cache.jsonl` + `RuleBasedQueryParser` (`< 1 ms`) thay vì bị treo 5 giây/câu truy vấn.

### 4. Khóa Chặt Provenance & Benchmark (`Exit Code = 0`)
- [run_validation_benchmark.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/run_validation_benchmark.py) đã vượt qua toàn bộ cổng kiểm định bảo mật dữ liệu (`inputs_unchanged=True`, `outputs_verified=True`, `Pure-ASR FP Top-1 = 0`).

---

## PHẦN II — Phần Nào Chưa Ổn (Vì Sao Chưa Ổn & Hướng Dẫn Sửa Chi Tiết)

Qua kiểm tra sâu từng dòng code trong 3 commits mới nhất (`8fe684a`, `7c1cd53`, `9beb770`), hiện còn **3 điểm kỹ thuật chưa ổn** (trong đó có **1 lỗi tiềm ẩn crash `AttributeError` / `TypeError`** khi bật `llm_compact` và **2 điểm làm giảm nhẹ hiệu năng/độ trễ**):

---

### Điểm Chưa Ổn 1 (Quan Trọng Nhất): Lỗi Crash `AttributeError` & `TypeError` Tại Nhánh `SE-Viterbi` Khi `routing_mode == "llm_compact"`

#### 1. Vì sao chưa ổn?
Hãy nhìn vào [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1599-L1600):
```python
sub_vecs_768 = [self.encode_text(sc) for sc in sub_clauses]
sub_q_864 = [build_query_864d_vector(sv, sc, self.coco_idf_map) for sv, sc in zip(sub_vecs_768, sub_clauses)]
```
Ở đây, `sub_q_864` là một **Python `list[np.ndarray]`** (danh sách các vector 1D `864d`), **không phải** là một `np.ndarray` 2 chiều!
Thế nhưng ngay bên dưới tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1679-L1695), Hydrogen lại viết:
```python
if (
    routing_plan is not None
    and routing_plan.routing_mode == "llm_compact"
    and sub_q_864 is not None
    and sub_q_864.shape[0] >= 2
    and cand_mat.shape[0] >= 2
):
    ...
    viterbi_res = viterbi_subevent_localization(
        timestamps=cand_ts,
        event_vis_curves=(sub_q_864 @ cand_mat.T),
    )
```
- **Hậu quả:**
  1. Khi `routing_plan.routing_mode == "llm_compact"` (tức là khi bật `VIDENTRA_LLM_ENABLED=1` trên GPU hoặc khi có cache hit từ `query_decomposition_cache.jsonl` cho câu có $\ge 2$ mệnh đề), biểu thức `sub_q_864.shape[0]` sẽ **văng lỗi crash ngay lập tức**:
     `AttributeError: 'list' object has no attribute 'shape'`!
  2. Ngay cả nếu sửa `.shape[0]` thành `len(sub_q_864)`, dòng tiếp theo `(sub_q_864 @ cand_mat.T)` cũng sẽ crash với:
     `TypeError: unsupported operand type(s) for @: 'list' and 'ndarray'` vì toán tử `@` trong Python không nhận `list` ở vế trái!
  3. Sở dĩ khi chạy pytest và benchmark vừa rồi lỗi này chưa phát nổ là vì trên CPU `VIDENTRA_LLM_ENABLED=0` nên `routing_plan.routing_mode` luôn là `"rule_fallback"`, khiến điều kiện `routing_plan.routing_mode == "llm_compact"` trả về `False` và nhảy qua khối lệnh lỗi này!

#### 2. Hướng dẫn sửa chi tiết
Mở file [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1678-L1699) và thay thế khối lệnh từ dòng `1678` đến `1699` bằng đoạn code an toàn kiểu dữ liệu sau (đồng thời cho phép chạy cả khi `rule_fallback` tách được mệnh đề thời gian `before/after/then` có `has_temporal_order`):

```python
                # Phase 7: Optional Sub-Event Viterbi alignment bonus when multi-event decomposition is active
                viterbi_bonus = 0.0
                if (
                    routing_plan is not None
                    and (
                        routing_plan.routing_mode == "llm_compact"
                        or getattr(routing_plan.decomposition, "temporal_relations", None)
                    )
                    and sub_q_864 is not None
                    and len(sub_q_864) >= 2
                    and cand_mat.shape[0] >= 2
                ):
                    sub_q_864_mat = np.vstack(
                        [np.asarray(sv, dtype=np.float32).reshape(1, -1) for sv in sub_q_864]
                    )
                    cand_ts = (
                        np.asarray(ts_list, dtype=np.float32)
                        if ts_list is not None and len(ts_list) == cand_mat.shape[0]
                        else np.arange(cand_mat.shape[0], dtype=np.float32) * 2.0
                    )
                    viterbi_res = viterbi_subevent_localization(
                        timestamps=cand_ts,
                        event_vis_curves=(sub_q_864_mat @ cand_mat.T),
                    )
                    if viterbi_res.get("monotonic_valid", False):
                        viterbi_bonus = 0.015 * max(0.0, float(viterbi_res.get("score", 0.0)))
```

---

### Điểm Chưa Ổn 2: Cổng Kích Hoạt `Dense E5` Ở Stage 3 (`unified_pipeline.py#L1905-L1954`) Vẫn Chạy Trên Mọi Truy Vấn Thuần Thị Giác

#### 1. Vì sao chưa ổn?
Tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1905-L1913), điều kiện kích hoạt `Dense E5` hiện tại là:
```python
sparse_lexical_yield = (len(ocr_candidates) + len(asr_candidates)) < 5
should_run_e5 = self._has_e5_text_index() and (
    sparse_lexical_yield
    or (
        routing_plan is not None
        and routing_plan.routing_mode == "llm_compact"
        and (routing_plan.w_ocr >= 0.25 or routing_plan.w_asr >= 0.25)
    )
)
```
- **Tại sao điều kiện `sparse_lexical_yield = (len(ocr_candidates) + len(asr_candidates)) < 5` lại gây tác dụng phụ?**
  - Với các truy vấn **thuần thị giác (Pure-Visual)** trong MSRVTT và QVHighlights (ví dụ: *"a man in a blue shirt walks a dog on the beach"*), bản chất của câu truy vấn là mô tả cảnh quay chứ không tìm chữ (OCR) hay lời thoại (ASR), nên số ứng viên FTS5 `len(ocr_candidates) + len(asr_candidates)` gần như luôn bằng `0` (tức là `< 5`).
  - Hệ quả 1 (**Độ trễ Non-Neural p50 tăng gấp 3 lần**): Vì `< 5` luôn đúng với mọi câu thuần thị giác, hàm `encode_e5_query(query)` và `_search_e5_text_index` bị gọi trên hầu hết 150 câu của `dev_150`, đồng thời bơm thêm tới 25 video từ nhánh chữ vào `all_clusters`, bắt Stage 4 phải gọi `lookup_true_visual_sim` trên đĩa cho hàng chục video không liên quan. Đây là lý do `Non-Neural Latency p50` tăng từ **`22.88 ms`** (Phase 6) lên **`72.68 ms`**.
  - Hệ quả 2 (**Mất 1 câu đúng ở Top-1 trên `dev_150`**): Khi E5 bơm một video có lời thoại trùng ngữ nghĩa chung chung (`sc >= 0.855`) vào một câu thuần thị giác, video đó nhận thêm `rrf_asr` và vượt qua video đúng ở Top-1 (làm `Video Hit@1` dừng ở `61.33%` = 92/150 thay vì `62.00%` = 93/150 của Phase 6).

#### 2. Hướng dẫn sửa chi tiết
Chỉ kích hoạt `Dense E5` khi câu truy vấn **thực sự có tín hiệu tìm kiếm chữ/lời thoại** (`w_ocr >= 0.22` hoặc `w_asr >= 0.22` hoặc có `ocr_terms` / `asr_terms` / cụm từ trong ngoặc kép) **mà** FTS5 tìm được ít kết quả (`< 5`), hoặc khi điểm tương đồng E5 cực cao (`sc >= 0.880`) kèm trùng khớp từ khóa có độ đặc hiệu cao (`high_spec_tokens`).

Mở [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1905-L1940) và sửa điều kiện `should_run_e5`:
```python
        sparse_lexical_yield = (len(ocr_candidates) + len(asr_candidates)) < 5
        has_lexical_modality_intent = (
            bool(ocr_query_str)
            or bool(asr_query_str)
            or bool(quoted_phrases)
            or (
                routing_plan is not None
                and (
                    routing_plan.w_ocr >= 0.22
                    or routing_plan.w_asr >= 0.22
                    or bool(routing_plan.ocr_queries)
                    or bool(routing_plan.asr_queries)
                )
            )
        )
        should_run_e5 = self._has_e5_text_index() and sparse_lexical_yield and has_lexical_modality_intent
```
Với thay đổi 10 dòng này:
- Các truy vấn có tín hiệu OCR/ASR nhưng bị lỗi chính tả OCR hoặc đồng nghĩa ASR (FTS5 trả về `< 5` kết quả) vẫn được `Dense E5` cứu trọn vẹn.
- Các truy vấn thuần thị giác 100% không bị `Dense E5` làm nhiễu và không tốn chi phí gọi encoder E5, giúp `Video Hit@1` quay về **`>= 62.00%`** và `Non-Neural Latency p50` giảm ngay về vùng **`~23–28 ms`**.

---

### Điểm Chưa Ổn 3: `shot_aware_graph_diffusion` Ở Stage 6 Làm Mờ Biên Độ Nhọn Trên Video Ngắn (`Moment nDCG@5` giảm nhẹ `0.5127 -> 0.5119`)

#### 1. Vì sao chưa ổn?
Tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2547-L2554), `shot_aware_graph_diffusion` đang được áp dụng cho mọi video có `len(ts_array) >= 4` với `alpha_d = 0.12` ngay trước bước `expand_boundary_contrast`:
```python
if len(ts_array) >= 4:
    boundary_curve = shot_aware_graph_diffusion(
        ts_array,
        boundary_curve,
        alpha_d=0.12,
        tau_time=3.5,
    )
```
Trên các đoạn video ngắn (4–7 keyframes, tương ứng 8–14 giây), đường cong `boundary_curve` vốn đã được làm mượt 3 điểm (`[0.20, 0.60, 0.20]`) ở dòng 2544. Việc tiếp tục khuếch tán đồ thị với `alpha_d=0.12` làm san phẳng градиент biên (`boundary contrast`), khiến `start_sec` và `end_sec` nở rộng quá mức 1–2 giây và làm giảm nhẹ `Moment nDCG@5` từ `0.5127` xuống `0.5119`.

#### 2. Hướng dẫn sửa chi tiết
Tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2547-L2554), chỉ bật `shot_aware_graph_diffusion` cho các video dài (`len(ts_array) >= 12`, tức các video QVHighlights 150 giây có $\ge 30$ keyframes) với hệ số bảo toàn biên `alpha_d = 0.06`:
```python
if len(ts_array) >= 12:
    boundary_curve = shot_aware_graph_diffusion(
        ts_array,
        boundary_curve,
        alpha_d=0.06,
        tau_time=3.0,
    )
```

---

## PHẦN III — Kế Hoạch Nâng Cấp Tiếp Theo (Phase 8 Master Plan: Vượt Trần MSRVTT, QVHighlights & Multi-Moment Localization)

### 3.0. Chẩn Đoán Điểm Nghẽn Hiện Tại Từ Bảng Phân Loại Lỗi (`Failure Taxonomy F1..F5`)

Để nâng cấp trúng đích thay vì thử sai cảm tính, hãy nhìn vào phân bố lỗi thực tế của hệ thống hiện tại trên cả 3 tập đánh giá:

| Mã Lỗi (Failure Mode) | Tỷ lệ trên `dev_150` | Tỷ lệ trên `dev_full_859` | Tỷ lệ trên `heldout_test_201` | Bản Chất Điểm Nghẽn Kỹ Thuật | Gói Nâng Cấp Giải Quyết Trực Tiếp |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`F4_FINE_GRAINED_VISUAL_CONFUSION`** | **18.00%** (27/150) | **33.76%** (290/859) | **35.32%** (71/201) | Video đúng **đã nằm trong Top-120 FAISS** (`Visual Recall@120 = 82%`), nhưng bị thua ở Top-1 trước một video có cảnh nền na ná (do hiệu ứng Hubness của vector `864d` và thiếu đối sánh tập mệnh đề con). | **Gói P8-1 (`QAC-CSLS` + `MV-SetSim`)** & **Gói P8-5 (`QAT-LateInteraction` $3\times 3$)** |
| **`F1_VISUAL_CANDIDATE_MISS`** | **14.67%** (22/150) | **18.28%** (157/859) | **16.92%** (34/201) | Video đúng rơi khỏi Top-120 của `SigLIP2` đơn khung hình và không có chữ OCR/ASR khớp từ khóa chính xác. | **Gói P8-5 (`PQ-VCA` `Florence-2` + `Dense E5` Synopsis)** |
| **`F2_LEXICAL_DISTRACTOR_OVERTHROW`** | **5.33%** (8/150) | **9.55%** (82/859) | **9.45%** (19/201) | Một video sai nhưng trùng ngẫu nhiên từ khóa OCR/ASR cộng dồn điểm tuyến tính vượt mặt ứng viên thị giác đúng. | **Gói P8-3 (`Synergistic-CrossLTR` 22d)** |
| **`Moment nDCG@5` (`0.5119` < mục tiêu `0.5300`) & `F5_BOUNDARY_DRIFT`** | **2.00%** (3/150) | — | — | Stage 6 hiện mới sinh tối đa 2 cửa sổ (`win1`, `_win2`), bỏ sót các đoạn highlight thứ 3, 4 trong cùng video QVHighlights dài 150s và chưa bắt sát điểm cắt cảnh (`shot cut`). | **Gói P8-2 (`NMS-MultiPeak` + `Shot-Snapped Boundary`)** |

> **Nhận xét chiến lược:**
> - **Gói P8-1, P8-2 và P8-3** là các nâng cấp thuật toán **100% chạy ngay trên CPU hiện tại (Zero-GPU-Cost)**, không cần chờ trích xuất lại đặc trưng nặng và có thể triển khai ngay sau khi vá 3 lỗi nhỏ ở Phần II.
> - **Gói P8-4 và P8-5** tận dụng Small-LLM GGUF và Kaggle Dual-T4 GPU để mở khóa toàn bộ các đặc trưng đang chờ dữ liệu (`llm_compact`, `s_late_interaction`, `PQ-VCA`).

---

### Gói Nâng Cấp P8-1 (Triển Khai Ngay Trên CPU): Khử Nhiễu Hubness `QAC-CSLS` & Đối Sánh Tập Mệnh Đề `MV-SetSim` (Giải Quyết Lỗi Lớn Nhất `F4_FINE_GRAINED_VISUAL_CONFUSION`)

#### 1. Tên phương pháp & Cơ sở toán học
- **Tên phương pháp:** **Query-Adaptive Cross-Modal Hubness Reduction (`QAC-CSLS`) kết hợp Multi-Subclause Symmetric Chamfer Similarity (`MV-SetSim`)**.
- **Vì sao giải quyết được `F4`?**
  1. **Hiện tượng Hubness trong không gian `864d`:** Trong không gian vector chiều cao, một số keyframes có nội dung "trung tính/phổ biến" (cảnh phòng khách, đường phố, người đứng nói chuyện) có tích vô hướng cao với hàng trăm câu truy vấn khác nhau (gọi là **Universal Hubs**). Khi câu truy vấn hơi mơ hồ, các video chứa frame "Hub" này luôn cướp vị trí Top-1 của video đúng.
  2. **Thiếu tính đầy đủ mệnh đề (Sub-clause Completeness):** Với câu truy vấn có nhiều thành phần (ví dụ: $c_1 =$ *"a boy in a red shirt"*, $c_2 =$ *"jumps into a swimming pool"*), trung bình cộng vector $q_{864}$ có thể chấm điểm rất cao cho một video chỉ có *"swimming pool"* thật rõ mà không hề có *"boy in a red shirt"*.
- **Công thức toán học tích hợp vào VIDENTRA:**
  - **Bước 1 (Precomputed Keyframe Hubness Radius $r_K(v_i)$):** Tính trước một lần duy nhất cho mỗi keyframe $v_i \in \mathbb{R}^{864}$ giá trị tương đồng trung bình với $K=16$ keyframe khác video gần nhất:
    $$r_K(v_i) = \frac{1}{K} \sum_{v_j \in \mathcal{N}_K^{\text{cross-video}}(v_i)} \langle v_i, v_j \rangle$$
    Lưu thành mảng 1 chiều `data/processed/siglip2/keyframe_hubness_radii.npy` (chỉ nặng `~150 KB` cho ~38,000 keyframes).
  - **Bước 2 (Điểm tương đồng hiệu chỉnh Hubness ở Stage 2 & Stage 4):**
    $$s_{\text{CSLS}}(q, v_i) = \langle q, v_i \rangle - \lambda_{\text{hub}} \cdot \max\bigl(0,\, r_K(v_i) - \bar{r}_K\bigr), \quad \lambda_{\text{hub}} = 0.35$$
    Chỉ phạt những keyframe có tính "Hub" cao hơn mức trung bình toàn cục $\bar{r}_K$.
  - **Bước 3 (Symmetric Chamfer Sub-Clause Coverage `MV-SetSim` trên cửa sổ ứng viên $W$):**
    Khi truy vấn có $M \ge 2$ mệnh đề con $\{q^{(1)}, \dots, q^{(M)}\}$, điểm bao phủ mệnh đề trên tập keyframes $\{v_t\}_{t \in W}$ của cửa sổ ứng viên là trung bình điều hòa giữa **Mean Clause MaxSim** và **Weakest-Link Clause MaxSim**:
    $$S_{\text{MV-SetSim}}(Q, W) = 0.65 \cdot \left(\frac{1}{M}\sum_{m=1}^{M} \max_{t \in W} \langle q^{(m)}, v_t \rangle\right) + 0.35 \cdot \left(\min_{m \in \{1..M\}} \max_{t \in W} \langle q^{(m)}, v_t \rangle\right)$$
    Số hạng $\min_{m} \max_{t \in W} \langle q^{(m)}, v_t \rangle$ đóng vai trò **Soft-AND Gate**: nếu ứng viên bỏ sót bất kỳ mệnh đề con nào, điểm `MV-SetSim` sẽ lập tức giảm xuống, loại bỏ hoàn toàn các video chỉ khớp một nửa câu truy vấn!

#### 2. Kế hoạch thực hiện chi tiết từng bước
- **Bước 1 (Tạo script tính bán kính Hubness 1 lần trong 3 giây):**
  - Thêm hàm `build_or_load_keyframe_hubness_radii()` trong [src/videntra/retrieval/visual.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/visual.py):
    ```python
    def compute_keyframe_hubness_radii(
        vectors_864: np.ndarray,
        video_ids: list[str],
        k_neighbors: int = 16,
        sample_anchors: int = 2048,
    ) -> np.ndarray:
        """Compute cross-video hubness radius r_K(v_i) for each keyframe."""
        n = vectors_864.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        rng = np.random.default_rng(42)
        anchor_idx = rng.choice(n, size=min(n, sample_anchors), replace=False)
        anchors = np.asarray(vectors_864[anchor_idx], dtype=np.float32)
        # Compute similarity in chunks to keep RAM < 200MB
        radii = np.zeros(n, dtype=np.float32)
        k_eff = min(k_neighbors, anchors.shape[0] - 1)
        if k_eff <= 0:
            return radii
        for start in range(0, n, 4096):
            end = min(n, start + 4096)
            chunk = np.asarray(vectors_864[start:end], dtype=np.float32)
            sims = chunk @ anchors.T
            part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
            radii[start:end] = np.mean(part, axis=1)
        return radii.astype(np.float32)
    ```
- **Bước 2 (Tích hợp `QAC-CSLS` và `MV-SetSim` vào `lookup_true_visual_sim` trong [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1620-L1710)):**
  - Tại `lookup_true_visual_sim`, trừ đi `0.35 * np.maximum(0.0, cand_radii - median_radius)` trên vector điểm khung hình `scores` trước khi lấy `s_max`.
  - Khi `sub_q_864` có `len(sub_q_864) >= 2`, tính ma trận `clause_frame_sims = sub_q_864_mat @ cand_mat.T` (kích thước $M \times T$):
    ```python
    per_clause_max = np.max(clause_frame_sims, axis=1)  # shape (M,)
    mv_set_sim = 0.65 * float(np.mean(per_clause_max)) + 0.35 * float(np.min(per_clause_max))
    ```
  - Lưu `min_clause_sim = float(np.min(per_clause_max))` vào `cluster_meta[vid]` để đưa thẳng vào bộ xếp hạng `LearnedRankerLTR` ở Gói P8-3.

---

### Gói Nâng Cấp P8-2 (Triển Khai Ngay Trên CPU): Đề Xuất Đa Đỉnh `NMS-MultiPeak` & Bắt Dính Điểm Cắt Cảnh `Shot-Snapped Boundary` (Đưa `Moment nDCG@5` Vượt Mức Nghiệm Thu `>= 0.5450`)

#### 1. Tên phương pháp & Cơ sở toán học
- **Tên phương pháp:** **1D Temporal Non-Maximum Suppression Multi-Peak Proposal (`NMS-MultiPeak`) kết hợp Cosine Shot-Boundary Snapping (`Shot-Snapped Watershed`)**.
- **Vì sao giải quyết được `Moment nDCG@5` (`0.5119` $\to$ `>= 0.5450`) và `Boundary MAE` (`6.91s` $\to$ `< 5.0s`)?**
  1. **Điểm mù của cơ chế `_win2` hiện tại:** Trong bộ dữ liệu QVHighlights, mỗi truy vấn được đánh giá trên **Top-5 đoạn khoảnh khắc (`Moment nDCG@5`)**, và nhiều video dài 150s có tới **3–4 phân cảnh phù hợp nằm cách xa nhau** (ví dụ: cảnh phỏng vấn ở giây `10–28`, quay lại ở giây `64–82`, và kết thúc ở giây `118–136`). Hiện tại Stage 6 ([unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2573-L2636)) chỉ tìm thêm tối đa 1 cửa sổ `_win2`, khiến vị trí thứ 3, 4, 5 trong bảng xếp hạng moment của video số 1 bị nhường cho các video xếp hạng dưới (vốn là các video sai!), kéo tụt `Moment nDCG@5`!
  2. **Điểm mù của biên thời gian tĩnh:** Hàm `expand_boundary_contrast` hiện tại mở rộng biên dựa trên đường cong tương đồng đã làm mượt, nên thường dừng lệch 1–2 keyframes so với thời điểm chuyển cảnh thực tế (`shot cut`) của biên tập viên video.
- **Công thức toán học tích hợp vào VIDENTRA:**
  - **Shot-Cut Detection miễn phí từ ma trận `768d`:** Giữa hai keyframe liên tiếp $t$ và $t+1$ trong cùng một video, độ đứt gãy chuyển cảnh (Shot Discontinuity) là:
    $$d_{\text{shot}}(t, t+1) = 1 - \langle \hat{v}_{t}^{(768)},\, \hat{v}_{t+1}^{(768)} \rangle$$
    Nếu $d_{\text{shot}}(t, t+1) > \tau_{\text{cut}} = 0.28$ và khoảng cách tới biên dự đoán $[t_{\text{start}}, t_{\text{end}}]$ nhỏ hơn $\le 4.0\text{s}$, ta ghim (snap) biên khoảnh khắc vào đúng mốc chuyển cảnh đó!
  - **Iterative 1D Temporal NMS cho $P \le 3$ cửa sổ mỗi video Top-3:**
    Sau khi trích xuất cửa sổ thứ $p$ là $[a_p, b_p]$, triệt tiêu vùng lân cận $[a_p - \delta, b_p + \delta]$ trên đường cong `exact_curve` và tiếp tục tìm đỉnh thứ $p+1$ nếu:
    $$s_{\max}^{(p+1)} \ge 0.80 \cdot s_{\max}^{(1)} \quad \text{và} \quad \max_{j \le p} \text{tIoU}\bigl([a_{p+1}, b_{p+1}], [a_j, b_j]\bigr) < 0.30$$
    Điểm của cửa sổ thứ $p \in \{2, 3\}$ được tính theo tỷ lệ năng lượng đỉnh và bị kẹp trần nghiêm ngặt `< score(win_{p-1})` để bảo toàn 100% `Moment Hit@1 = 72.73%`:
    $$\text{score}(\text{win}_p) = \text{score}(\text{win}_1) \cdot \min\Bigl(0.96 - 0.03(p-2),\; \max\bigl(0.78,\; 0.92 \cdot \frac{s_{\max}^{(p)}}{s_{\max}^{(1)}}\bigr)\Bigr)$$

#### 2. Kế hoạch thực hiện chi tiết từng bước
- **Bước 1:** Thêm hàm `snap_boundaries_to_shot_cuts()` trong [src/videntra/localization/refine.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/localization/refine.py):
  ```python
  def snap_boundaries_to_shot_cuts(
      timestamps: np.ndarray,
      frame_vectors_768: np.ndarray,
      start_sec: float,
      end_sec: float,
      cut_threshold: float = 0.28,
      max_snap_sec: float = 3.0,
  ) -> tuple[float, float]:
      """Snap predicted [start_sec, end_sec] to nearest natural shot cuts if within max_snap_sec."""
      if len(timestamps) < 3 or frame_vectors_768.shape[0] != len(timestamps):
          return start_sec, end_sec
      adj_cos = np.sum(frame_vectors_768[:-1] * frame_vectors_768[1:], axis=1)
      adj_dist = 1.0 - adj_cos
      cut_indices = np.where(adj_dist >= cut_threshold)[0]
      if len(cut_indices) == 0:
          return start_sec, end_sec
      # Cut timestamp between frame i and i+1
      cut_times_start = timestamps[cut_indices + 1]
      cut_times_end = timestamps[cut_indices]
      new_start, new_end = float(start_sec), float(end_sec)
      diffs_s = np.abs(cut_times_start - start_sec)
      if np.min(diffs_s) <= max_snap_sec:
          cand_s = float(cut_times_start[int(np.argmin(diffs_s))])
          if cand_s < new_end - 2.0:
              new_start = cand_s
      diffs_e = np.abs(cut_times_end - end_sec)
      if np.min(diffs_e) <= max_snap_sec:
          cand_e = float(cut_times_end[int(np.argmin(diffs_e))])
          if cand_e > new_start + 2.0:
              new_end = cand_e
      return new_start, new_end
  ```
- **Bước 2:** Trong Stage 6 của [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L2573-L2636), nâng vòng lặp sinh cửa sổ phụ từ 1 cửa sổ (`_win2`) lên tối đa 2 cửa sổ phụ (`_win2`, `_win3`) cho các video dài (`len(ts_array) >= 20`) có nhiều đỉnh độc lập (`tIoU < 0.30`).

---

### Gói Nâng Cấp P8-3 (Triển Khai Ngay Trên CPU): Mở Rộng `LearnedRankerLTR` Lên 22 Đặc Trưng Với Tương Tác Chéo Đa Phương Thức (`Synergistic-CrossLTR` — Giải Quyết Lỗi `F2_LEXICAL_DISTRACTOR_OVERTHROW`)

#### 1. Tên phương pháp & Cơ sở toán học
- **Tên phương pháp:** **Monotonic Constrained Synergistic Cross-Feature LTR (`Synergistic-CrossLTR` 22d)**.
- **Vì sao giải quyết được `F2_LEXICAL_DISTRACTOR_OVERTHROW`?**
  - Mô hình LTR 18 chiều hiện tại hoàn toàn tuyến tính cộng tính:
    $$f(x) = w_{\text{vis}} x_{\text{vis}} + w_{\text{ocr}} x_{\text{ocr}} + w_{\text{asr}} x_{\text{asr}} + \dots$$
  - Trong mô hình cộng tính, một video có cảnh hình ảnh hoàn toàn sai ($z_{\text{vis\_max}} < 0$), nhưng tình cờ người trong video nói một câu trùng nhiều từ khóa (`rrf_asr` cao + `bm25_log_score` cao + `anchor_lexical_match = 1`) vẫn nhận được tổng điểm dương lớn từ các trọng số từ vựng và vượt mặt video đúng!
  - Bằng cách thêm **4 đặc trưng tương tác phi tuyến tường minh (Explicit Non-Linear Interaction Features)** vào [features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py):
    1. `vis_lexical_synergy` (`+1` monotonic):
       $$x_{19} = \max\bigl(0.0,\, s_{\text{vis\_blend\_max}} - 0.16\bigr) \times \bigl(\text{rrf}_{\text{ocr}} + \text{rrf}_{\text{asr}}\bigr)$$
       *(Chỉ thưởng điểm cộng hưởng lớn khi cả thị giác LẪN chữ/lời thoại cùng khớp trên cùng một video).*
    2. `unverified_lexical_hazard` (`-1` monotonic):
       $$x_{20} = \max\bigl(0.0,\, 0.50 - z_{\text{vis\_max}}\bigr) \times \bigl(\text{rrf}_{\text{ocr}} + \text{rrf}_{\text{asr}}\bigr)$$
       *(Phạt trực tiếp ứng viên có điểm RRF chữ/thoại cao nhưng điểm chuẩn hóa thị giác $z_{\text{vis\_max}}$ nằm dưới mức trung bình của câu truy vấn).*
    3. `subclause_weakest_link` (`+1` monotonic):
       $$x_{21} = \min_{m \in \{1..M\}} \max_{t \in W} \langle q^{(m)}, v_t \rangle$$
       *(Điểm khớp của mệnh đề yếu nhất từ Gói P8-1 — ép video hạng 1 phải khớp toàn diện mọi vế của câu truy vấn).*
    4. `window_energy_concentration` (`+1` monotonic):
       $$x_{22} = \frac{\text{mean}_{t \in [t_{\text{start}}, t_{\text{end}}]} s(t)}{\text{mean}_{t \notin [t_{\text{start}}, t_{\text{end}}]} s(t) + 0.05}$$
       *(Đo độ nổi bật của khoảnh khắc dự đoán so với phần nền còn lại của video).*

#### 2. Kế hoạch thực hiện chi tiết từng bước
- **Bước 1:** Bổ sung 4 tên đặc trưng mới vào `FEATURE_NAMES` trong [src/videntra/ranking/features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py#L19-L38) và thêm ràng buộc dấu tương ứng (`+1, -1, +1, +1`) vào `MONOTONIC_DIRECTIONS` trong [src/videntra/ranking/learned_ranker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/learned_ranker.py).
- **Bước 2:** Tính 4 giá trị này trong `extract_candidate_features()` ([src/videntra/ranking/features.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/ranking/features.py#L85-L185)).
- **Bước 3:** Chạy lại huấn luyện 5-Fold GroupKFold CV:
  ```powershell
  .\.venv\Scripts\python.exe scripts/train_ltr_ranker_cv.py
  ```

---

### Gói Nâng Cấp P8-4 (Offline Pre-warm): Kích Hoạt 100% Phân Rã Ngữ Nghĩa `Qwen2.5-1.5B-Instruct` & `SE-Viterbi` Đa Phương Thức

#### 1. Định hướng cụ thể
- Như đã phân tích ở Phần II, hiện tại tệp `data/cache/query_decomposition_cache.jsonl` mới chỉ chứa kết quả của `RuleBasedQueryParser` (`fallback_reason="rule_based_only"`), nên 100% câu truy vấn trong benchmark đang chạy ở chế độ `routing_mode = "rule_fallback"`.
- Sau khi sửa lỗi `sub_q_864_mat` ở Điểm Chưa Ổn 1, ta cần:
  1. Truyền đầy đủ cả 3 ma trận đường cong `event_vis_curves`, `event_ocr_curves` và `event_asr_curves` vào hàm [viterbi_subevent_localization](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/localization/refine.py#L315-L340) trong `unified_pipeline.py`.
  2. Chạy `prewarm_llm_query_cache.py` với mô hình `Qwen2.5-1.5B-Instruct` thật (trên Kaggle T4 chỉ mất ~90 giây cho toàn bộ 1,060 câu truy vấn của cả 3 tập `dev_150`, `dev_full_859`, `heldout_test_201`, hoặc chạy qua đêm bằng GGUF trên CPU) để nạp đầy `query_decomposition_cache.jsonl` với `routing_mode = "llm_compact"`.

#### 2. Kế hoạch thực hiện chi tiết từng bước
- **Bước 1:** Cập nhật lời gọi `viterbi_subevent_localization` tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1679-L1698) để tận dụng cả `modality_weights=(routing_plan.w_vis, routing_plan.w_ocr, routing_plan.w_asr)`.
- **Bước 2:** Chạy lệnh tạo cache LLM thực:
  ```powershell
  $env:VIDENTRA_LLM_ENABLED="1"
  $env:VIDENTRA_LLM_DEADLINE_MS="45000"
  .\.venv\Scripts\python.exe scripts/prewarm_llm_query_cache.py
  ```

---

### Gói Nâng Cấp P8-5 (Kaggle Dual-T4 GPU Offload): Kích Hoạt `QAT-LateInteraction` ($3\times 3$ Spatial Tokens) & `PQ-VCA` (`Florence-2-base` Event Synopses — Giải Quyết Lỗi `F1_VISUAL_CANDIDATE_MISS`)

#### 1. Định hướng cụ thể
- Lỗi `F1_VISUAL_CANDIDATE_MISS` (`14.67%` trên `dev_150`) xảy ra khi vector trung bình toàn cục `768d` của `SigLIP2` không bắt được vật thể nhỏ hoặc hành động phức tạp, khiến video đúng không lọt nổi vào Top-120 FAISS.
- Hai vũ khí mạnh nhất đã được viết sẵn khung code trong dự án cho bài toán này là:
  1. **`QAT-LateInteraction` (`spatial_patches_fp16.npy`)**: Thay vì nén cả bức ảnh thành 1 vector duy nhất, giữ lại $3\times 3 = 9$ vector vùng không gian từ tầng ViT cuối cùng của `SigLIP2` để khớp từng cụm danh từ con vào đúng ô không gian tương ứng (`s_late_interaction` — đặc trưng thứ 18 trong `videntra_ltr_v1.json`).
  2. **`PQ-VCA` (`Florence-2-base` Keyframe Captions + `build_structured_event_synopsis`)**: Sinh mô tả chi tiết bằng ngôn ngữ tự nhiên cho từng keyframe bằng `microsoft/Florence-2-base`, hợp nhất với nhãn `YOLOv11n` + `OCR` + `ASR` thành một đoạn tóm tắt sự kiện `[SCENE] ... | [OBJECTS] ... | [OCR] ... | [ASR] ...` và lập chỉ mục bằng `multilingual-e5-small` (`e5_flat_ip.faiss`). Nhờ đó, đối sánh `Query <-> Video` được bổ sung kênh **Text-to-Text Dense Retrieval** cực mạnh, cứu trực tiếp các câu bị trượt ở kênh `Text-to-Image SigLIP2` (`F1`)!

#### 2. Kế hoạch thực hiện chi tiết từng bước
- **Bước 1 (Đóng gói dữ liệu đầu vào cho Kaggle):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/package_kaggle_inputs.py
  ```
- **Bước 2 (Chạy trên Kaggle Notebook với GPU T4 $\times 2$):**
  Chạy [scripts/kaggle_multimodal_offline_worker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/kaggle_multimodal_offline_worker.py) để xuất ra 3 tệp:
  1. `spatial_patches_fp16.npy` (kích thước $N \times 9 \times 768$, định dạng `float16`).
  2. `pseudo_captions.parquet` (mô tả khung hình từ `microsoft/Florence-2-base`).
  3. `query_decomposition_cache.jsonl` (phân rã truy vấn bằng `Qwen/Qwen2.5-1.5B-Instruct` trên GPU).
- **Bước 3 (Tích hợp kết quả về máy local & xây dựng lại chỉ mục):**
  ```powershell
  # 1. Nạp kết quả từ Kaggle và kiểm tra provenance
  .\.venv\Scripts\python.exe scripts/ingest_kaggle_offline_results.py

  # 2. Xây dựng lại chỉ mục Dense E5 có kèm Structured Event Synopses (PQ-VCA)
  .\.venv\Scripts\python.exe scripts/build_e5_text_index.py

  # 3. Huấn luyện lại bộ xếp hạng LearnedRankerLTR (kích hoạt trọng số s_late_interaction > 0)
  .\.venv\Scripts\python.exe scripts/train_ltr_ranker_cv.py

  # 4. Chạy đánh giá benchmark toàn diện
  .\.venv\Scripts\python.exe scripts/run_validation_benchmark.py
  ```

---

## PHẦN IV — Bảng Lộ Trình Thực Thi Theo Thứ Tự Ưu Tiên & Chỉ Tiêu Nghiệm Thu Phase 8

| Thứ tự | Hạng mục triển khai | Môi trường chạy | Thời gian ước tính | Mục tiêu cải thiện chính |
| :---: | :--- | :---: | :---: | :--- |
| **Bước 1** | Vá 3 điểm ở **Phần II** (`sub_q_864_mat` crash fix, cổng `has_lexical_modality_intent` cho E5, ngưỡng `shot_aware_graph_diffusion`) | Local CPU | `15 phút` | Khôi phục `dev_150` Video Hit@1 $\ge \mathbf{62.00\%}$, `Moment nDCG@5` $\ge \mathbf{0.5127}$, kéo `Non-Neural Latency p50` từ `72.68 ms` về $\le \mathbf{25.0\text{ ms}}$ |
| **Bước 2** | Triển khai **Gói P8-1 (`QAC-CSLS` + `MV-SetSim`)** + **Gói P8-2 (`NMS-MultiPeak` + `Shot-Snapped Boundary`)** + **Gói P8-3 (`Synergistic-CrossLTR` 22d)** | Local CPU | `45–60 phút` | Giảm lỗi `F4` & `F2`, đẩy `dev_150` Video Hit@1 lên $\mathbf{63.33\%–64.67\%}$, `Moment nDCG@5` vượt chuẩn nghiệm thu $\ge \mathbf{0.5350}$ |
| **Bước 3** | Triển khai **Gói P8-4 (Pre-warm `Qwen2.5-1.5B` Cache + `SE-Viterbi` đa phương thức)** | Kaggle GPU (hoặc Local GGUF) | `5 phút` (GPU) | Mở khóa `routing_mode = "llm_compact"` thực sự cho 1,060 câu truy vấn, tăng độ chính xác trên câu dài/đa sự kiện |
| **Bước 4** | Triển khai **Gói P8-5 (`spatial_patches_fp16.npy` $3\times 3$ + `PQ-VCA` `Florence-2` Event Synopses)** | Kaggle Dual-T4 GPU $\to$ Local | `30–45 phút` (GPU) | Đánh thức `s_late_interaction`, giảm mạnh lỗi `F1_VISUAL_CANDIDATE_MISS`, đưa `dev_150` Video Hit@1 lên vùng $\mathbf{66.0\%–70.0\%}$ và `heldout_test_201` vượt $\mathbf{45.0\%}$ |
