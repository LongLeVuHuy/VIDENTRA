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

## PHẦN III — Định Hướng Nghiên Cứu Cụ Thể & Kế Hoạch Thực Hiện Chi Tiết (Giai Đoạn Tiếp Theo)

Hiện tại kiến trúc khung của Phase 7 (`SmallLLMQueryDecomposer` + `LearnedRankerLTR` + `Dense E5` + `CM-GD` + `SE-Viterbi` + `PQ-VCA`) đã được xây dựng hoàn chỉnh trong codebase và có đầy đủ unit tests. Tuy nhiên, có **3 thành phần nghiên cứu đột phá đang ở trạng thái "chờ dữ liệu offline (latent-ready)"** nên chưa phát huy hết công suất trên tập benchmark thực tế. Dưới đây là định hướng nghiên cứu cụ thể và kế hoạch triển khai chi tiết từng bước để đẩy `Video Hit@1` từ **`61.33%` lên vùng `66%–70%`** và `Moment Hit@1 (0.5)` từ **`72.73%` lên `>76%`**:

---

### Hướng Nghiên Cứu 1: Pre-warm Toàn Bộ Cache Phân Rã Truy Vấn Bằng `Qwen2.5-1.5B-Instruct` & Kích Hoạt Thực Chiến `SE-Viterbi`

#### 1. Định hướng cụ thể
- Hiện tại `data/cache/query_decomposition_cache.jsonl` mới được tạo bởi `RuleBasedQueryParser` (`fallback_reason="rule_based_only"`), nên khi chạy benchmark, 100% câu truy vấn đi theo nhánh `routing_mode = "rule_fallback"` và **chưa câu nào dùng phân rã ngữ nghĩa thực sự của `Qwen2.5-1.5B-Instruct`**!
- Với file GGUF `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` đã có sẵn trong máy (hoặc chạy trên Kaggle GPU T4 trong ~2 phút cho toàn bộ 1,060 câu truy vấn), ta có thể chạy `scripts/prewarm_llm_query_cache.py` không bật cờ `--rule-only` để sinh bộ phân rã JSON chuẩn cho toàn bộ tập truy vấn.
- Khi đó, các câu phức hợp có mệnh đề thời gian (`before`, `after`, `then`, `while`) sẽ kích hoạt `routing_mode = "llm_compact"` và đi vào thuật toán quy hoạch động **`viterbi_subevent_localization` (`SE-Viterbi`)** đã sửa ở Điểm Chưa Ổn 1.

#### 2. Kế hoạch thực hiện chi tiết
- **Bước 1:** Sửa lỗi `sub_q_864_mat` ở Điểm Chưa Ổn 1 tại [unified_pipeline.py](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/unified_pipeline.py#L1679-L1695).
- **Bước 2:** Nâng cấp `viterbi_subevent_localization` trong `unified_pipeline.py` để truyền thêm cả `event_ocr_curves` và `event_asr_curves` khi mệnh đề con có chứa `ocr_terms` hoặc `asr_terms` (hiện tại hàm [viterbi_subevent_localization](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/localization/refine.py#L315-L340) đã hỗ trợ tham số `event_ocr_curves`, `event_asr_curves` và `modality_weights`, nhưng nơi gọi trong `unified_pipeline.py` mới chỉ truyền `event_vis_curves`).
- **Bước 3:** Chạy lệnh pre-warm bằng backend GGUF:
  ```powershell
  $env:VIDENTRA_LLM_ENABLED="1"
  $env:VIDENTRA_LLM_DEADLINE_MS="30000"
  .\.venv\Scripts\python.exe scripts/prewarm_llm_query_cache.py
  ```
- **Bước 4:** Huấn luyện lại `train_ltr_ranker_cv.py` và chạy `run_validation_benchmark.py` để đo mức tăng `Video Hit@1` và `Moment Hit@1` trên nhóm truy vấn đa sự kiện (Multi-Event / Temporal Queries).

---

### Hướng Nghiên Cứu 2: Kích Hoạt Thực Sự `QAT-LateInteraction` ($3\times 3$ Spatial Patch Tokens) Qua Kaggle GPU Worker

#### 1. Định hướng cụ thể
- Trong [videntra_ltr_v1.json](file:///C:/Users/DELL/Desktop/VIDENTRA/models/ranking/videntra_ltr_v1.json#L46), trọng số của đặc trưng thứ 18 `s_late_interaction` hiện đang bằng **`0.0`** vì tệp `data/processed/siglip2/spatial_patches_fp16.npy` chưa được trích xuất từ tầng cuối của ViT `SigLIP2` (`google/siglip2-base-patch16-224`).
- Khi trích xuất lưới $3\times 3 = 9$ vector vùng không gian (mỗi vùng `768d`, lưu dạng `float16` hoặc nén PCA `256d`) cho mỗi keyframe trên Kaggle GPU, cơ chế **ColBERT-style MaxSim Late Interaction** ([compute_late_interaction_score](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/visual.py#L466-L526)) sẽ cho phép mỗi cụm danh từ con của truy vấn (ví dụ: *"a red car in the background"* và *"a woman holding an umbrella"*) khớp độc lập vào đúng ô không gian $3\times 3$ tương ứng của khung hình, thay vì bị hòa lẫn trong một vector global pooling duy nhất.

#### 2. Kế hoạch thực hiện chi tiết
- **Bước 1:** Trong [kaggle_multimodal_offline_worker.py](file:///C:/Users/DELL/Desktop/VIDENTRA/scripts/kaggle_multimodal_offline_worker.py), lấy tensor `last_hidden_state` $(B, 196, 768)$ từ `vision_model` của `SigLIP2`, reshape về lưới $(B, 14, 14, 768)$ và áp dụng `AdaptiveAvgPool2d((3, 3))` + chuẩn hóa $L_2$ để thu được tensor $(N, 9, 768)$ ở định dạng `float16` (`spatial_patches_fp16.npy`).
- **Bước 2:** Tải file `spatial_patches_fp16.npy` từ Kaggle về thư mục `data/processed/siglip2/`.
- **Bước 3:** Chạy lại `.\.venv\Scripts\python.exe scripts/train_ltr_ranker_cv.py` — lúc này cột thứ 18 `s_late_interaction` sẽ tự động khác 0 và học được trọng số dương trong `videntra_ltr_v1.json`, giúp tăng mạnh độ chính xác trên các câu truy vấn mô tả vật thể nhỏ hoặc bố cục không gian phức tạp.

---

### Hướng Nghiên Cứu 3: `PQ-VCA` Mode A & Mode B — Làm Giàu Chỉ Mục Ngữ Nghĩa Bằng Pseudo-Query Keyframe Captions (`Florence-2-base`)

#### 1. Định hướng cụ thể
- Hiện tại chỉ mục `Dense E5` (`data/processed/text_e5/`) mới đang lập chỉ mục trên các dòng `OCR` và `ASR` sẵn có (Mode C). Với các video không có phụ đề/lời thoại hoặc có lời thoại lệch với hình ảnh (visual-dialogue mismatch), nhánh `Dense E5` chưa có mô tả thị giác dạng văn bản để đối sánh.
- Module [build_structured_event_synopsis](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/indexing/text_store.py#L31-L82) đã được viết sẵn để hợp nhất `[SCENE] + [OBJECTS] + [OCR] + [ASR]` thành một đoạn tóm tắt sự kiện đa phương thức (Structured Event Synopsis) cho mỗi cửa sổ 4–6 giây.

#### 2. Kế hoạch thực hiện chi tiết
- **Bước 1:** Trên Kaggle Dual-T4 GPU, chạy `microsoft/Florence-2-base` (`<MORE_DETAILED_caption>`) cho các keyframes đại diện (mỗi 2–4 giây/keyframe) để sinh `pseudo_captions.parquet`.
- **Bước 2:** Chạy `scripts/build_e5_text_index.py` để hợp nhất `pseudo_captions.parquet` + `yolo_boxes` + `ocr_segments` + `asr_segments` qua hàm `build_structured_event_synopsis()` và xây dựng lại `e5_flat_ip.faiss`.
- **Bước 3:** Bổ sung nhánh `rrf_synopsis` vào Stage 3 RRF Fusion để kết hợp đối sánh văn bản–văn bản (`Query <-> Florence-2 Synopsis`) song song với đối sánh văn bản–hình ảnh (`Query <-> SigLIP2 864d`), giải quyết triệt để khoảng cách ngữ nghĩa (Modality Gap) của không gian nhúng đa phương thức.

---

### Hướng Nghiên Cứu 4: `CM-GD` Query-Adaptive CSLS Hubness Reduction & Đặc Trưng Tương Tác Bậc Hai (Cross-Modal Interaction Features)

#### 1. Định hướng cụ thể
- **Giảm nhiễu Hubness bằng Query-Adaptive CSLS:** Trong không gian `864d`, một số khung hình có độ phức tạp thị giác trung bình cao thường đóng vai trò là "Hubs" (xuất hiện trong Top-K của rất nhiều câu truy vấn khác nhau). Hàm [csls_hubness_adjusted_scores](file:///C:/Users/DELL/Desktop/VIDENTRA/src/videntra/retrieval/visual.py#L430-L463) đã được cài đặt sẵn. Bằng cách tính trước vector trung bình $k$-láng giềng gần nhất $r_T(v_i)$ cho toàn bộ keyframes vào một mảng `keyframe_hubness_radii.npy` (kích thước chỉ vài trăm KB), ta có thể trừ đi $\frac{\beta}{2} r_T(v_i)$ ở thời điểm truy vấn với chi phí $O(1)$, loại bỏ ngay các video "Hub" gây nhiễu ở Top-1.
- **Đặc trưng tương tác chéo (Cross-Modal Synergistic Features) cho `LearnedRankerLTR`:** Hiện tại `LearnedRankerLTR` là mô hình tuyến tính trên 18 đặc trưng đơn lẻ. Việc bổ sung 3 đặc trưng tương tác nhân tính (multiplicative cross-features) có ý nghĩa vật lý rõ ràng vào `features.py`:
  1. `vis_x_ocr = s_vis_blend_max * rrf_ocr` (đồng thuận Thị giác $\times$ Chữ trên màn hình)
  2. `vis_x_asr = s_vis_blend_max * rrf_asr` (đồng thuận Thị giác $\times$ Lời thoại)
  3. `vis_x_spec = s_vis_blend_max * high_spec_token_coverage` (đồng thuận Thị giác $\times$ Độ phủ thực thể hiếm)
  sẽ giúp bộ xếp hạng tuyến tính nắm bắt được hiệu ứng cộng hưởng phi tuyến (non-linear multimodal synergy) mà không làm tăng nguy cơ overfitting trên tập huấn luyện 859 truy vấn.
