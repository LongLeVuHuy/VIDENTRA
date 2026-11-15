# VIDENTRA — Báo Cáo Audit Code (`src/`, `benchmark/`, `docs/PROGRESS.md`), Điểm Mới & Hướng Nghiên Cứu

> Ngày audit: 15/11/2026 · HEAD audit: `8b4ac80` (Phase 8, vá audit 1–3, residual LTR, T-SOT, attention pooler).
> Phương pháp: đọc code `src/videntra/retrieval|ranking|ocr|asr`, `scripts/offline_processor.py`, `scripts/train_ltr_ranker_cv.py`, `benchmark/`, `docs/PROGRESS.md`; các nhận định thuật toán đều được **chạy thử số liệu** (mục B3, B4). Chưa chạy lại full benchmark/pytest trong lượt audit này — các số benchmark bên dưới lấy từ `PROGRESS.md`.

---

## A. Kết luận nhanh

| # | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| B1 | **CRITICAL** | `scripts/offline_processor.py` dùng **vector ngẫu nhiên** làm embedding và **ghi đè `data/processed/faiss_index.bin`** (kể cả khi không `--reset`) | `offline_processor.py:235-236, 364-365, 429-430` |
| B2 | **HIGH** | `VietOCRRecognizer` / `GeneralOCRRecognizer` là **stub trả chuỗi cố định**; `PhoWhisperTranscriber` khi lỗi **bịa transcript "Cảm ơn"**; dữ liệu thật dùng RapidOCR + Faster-Whisper | `ocr/recognition.py:47-78`, `asr/transcription.py:103-109`, `update_real_ocr.py`, `update_real_asr.py` |
| B3 | **HIGH** | `temporal_sinkhorn_alignment_score` (T-SOT) thực chất ≈ **trung bình toàn video**, không phát hiện sự kiện | thử số: có sự kiện 0.1105 vs không 0.1012 (đúng là 0.30) |
| B4 | **HIGH** | `query_guided_temporal_attention_score` **thưởng khung hình rời rạc** (cắt cảnh) thay vì hành động mạch lạc | thử số: coherent 0.30 vs diverse 0.575 |
| B5 | **HIGH** | **Rò rỉ đánh giá**: LTR fit trên `dev_full_859` ⊃ `dev_150`, nhưng số `dev_150` vẫn được báo như kết quả chính | `train_ltr_ranker_cv.py:427`, `run_validation_benchmark.py:218` |
| B6 | MEDIUM | Kênh `E5 caption` (P8-5) bị **chặn bởi cổng lexical** → chết với câu thuần thị giác (đúng đối tượng cần cứu) | `unified_pipeline.py:1969-1997, 2064` |
| B7 | MEDIUM | Phase 7–8 **không cải thiện** (thậm chí thấp hơn) Phase 6 trên cả 3 tập; latency tăng ~47% | bảng mục C |
| B8 | MEDIUM | Attention bonus chỉ cộng cho **Top-15** → gián đoạn xếp hạng tại rank 15, không có trong dữ liệu train LTR | `unified_pipeline.py:2656-2692` |
| B9 | MEDIUM | `benchmark/` **cũ** (Phase 6, SHA `4742a7e`, `completed_phase6`) nhưng README/PROGRESS trích số khác | `benchmark_summary.json` |
| B10 | LOW | README (do mình viết) có lệnh sai: `check-paths`, `uvicorn …:app`, `/api/search`, `search_with_metadata(query=…)`, quảng bá VietOCR/PhoWhisper như đã dùng → **đã sửa trong lượt này** | — |
| B11 | LOW | 34 chỗ `except Exception:`/`except:` nuốt lỗi; OCR/ASR gán cứng `language="en"` | grep `src/` |

---

## B. Chi tiết lỗi & cách sửa

### B1 — `offline_processor.py` phá chỉ mục thật (CRITICAL)
**Vì sao lỗi:**
- Dòng 235-236: `vec = np.random.randn(768)` "Visual embedding placeholder" → mọi keyframe nhận vector ngẫu nhiên.
- Dòng 364-365: `faiss.write_index(index, self.faiss_path)` ghi vô điều kiện vào mặc định `data/processed/faiss_index.bin` — **chính là file 768d thật** (35.58 MB = 11 581 × 768 × 4 B).
- Dòng 429-430: `--reset` còn `unlink()` cả `metadata.db` và FAISS. Vi phạm `AGENTS.md` ("không overwrite/xóa dữ liệu thật").
- README cũ hướng dẫn chạy file này → người dùng làm hỏng chỉ mục.

**Cách sửa (chọn 1):**
1. *Khuyến nghị*: xóa `scripts/offline_processor.py` (legacy scaffold), hoặc chuyển vào `scripts/legacy/` kèm banner.
2. Nếu giữ: (a) đổi `faiss_path` mặc định sang `work/offline_scratch/faiss_index.bin`; (b) thay placeholder bằng `SigLIP2Encoder` thật (`src/videntra/encoders/siglip2.py`) hoặc `raise NotImplementedError` khi không có encoder; (c) trước `write_index`/`unlink` kiểm tra file tồn tại và yêu cầu `--i-understand-overwrite` + sao lưu `*.bak-<timestamp>`.
3. Thêm test: gọi `OfflineProcessingPipeline(...).run_all` với `faiss_path` tồn tại phải **raise** nếu không có cờ ghi đè.

### B2 — OCR/ASR "chuyên gia tiếng Việt" chưa thực sự hoạt động (HIGH)
**Hiện trạng thật:**
- `GeneralOCRRecognizer.recognize_crop` luôn trả `"VIDENTRA 2026"` (conf 0.88); `VietOCRRecognizer` luôn trả `"Cửa hàng giảm giá"` (conf 0.92). File `models/ocr/vietocr/vgg_transformer.pth` **không bao giờ được nạp**.
- `OCRRouter` chỉ được dùng trong `cli.py:388` và `benchmark_smoke_evidence.py`, không nằm trong luồng tạo `metadata.db`.
- `PhoWhisperTranscriber` có nạp model thật, nhưng nhánh `except Exception` trả `"Cảm ơn"` conf 0.70 → **bằng chứng bịa**, trái nguyên tắc "missing evidence là unknown".
- DB thật (1 324 video, 6 912 OCR obs, 3 860 ASR seg) được tạo bởi `update_real_ocr.py` (RapidOCR) + `update_real_asr.py` (Faster-Whisper, mặc định `tiny`), gán cứng `language="en"`.

**Cách sửa:**
1. `VietOCRRecognizer`: nạp thật bằng `vietocr` (`Cfg.load_config_from_name('vgg_transformer')`, `weights=models/ocr/vietocr/vgg_transformer.pth`, `Predictor`), trả `TextRecognitionResult` từ ảnh crop; nếu thiếu thư viện/weights → `raise RuntimeError` (không trả chuỗi giả).
2. `GeneralOCRRecognizer`: bọc `RapidOCR` (đang dùng ở `update_real_ocr.py`) để cả luồng dùng chung một lớp.
3. `PhoWhisperTranscriber`: bỏ nhánh fallback bịa; trả `TranscriptResult(text="", confidence=0.0, status="error")` và log lỗi.
4. `update_real_ocr.py`: dùng `OCRRouter` (RapidOCR → nếu có dấu tiếng Việt/conf thấp → VietOCR); `update_real_asr.py`: thêm `--backend {faster_whisper,phowhisper}` và phát hiện ngôn ngữ (`info.language`) thay vì gán `"en"`.
5. Test: stub phải bị cấm trong code chạy thật (test `assert "VIDENTRA 2026" not in output`), và test fallback ASR trả rỗng.
6. Chạy lại OCR/ASR cho các video tiếng Việt, ghi `model_id` + `revision` vào receipt (đúng quy tắc pin revision/hash).

### B3 — T-SOT ≈ trung bình toàn video, không phải căn chỉnh sự kiện (HIGH)
**Vì sao lỗi** (`alignment.py:temporal_sinkhorn_alignment_score`, gọi ở `unified_pipeline.py:1767`):
- Biên cột đều `b = 1/T`: **mỗi** keyframe bắt buộc nhận khối lượng bằng nhau ⇒ `Σ P·sims` hội tụ về ≈ trung bình sims của cả video (mất tính "đỉnh").
- `clause_frame_sims` là ma trận **cả video** (không phải cửa sổ như docstring nói); phạt `η((m/(M-1))-(t/(T-1)))²` ép mệnh đề đầu về đầu video, mệnh đề cuối về cuối video — sai với sự kiện nằm giữa.
- Thử số (2 mệnh đề × 40 khung, nền 0.10, sự kiện 0.30): T-SOT **0.1105** (có sự kiện) vs **0.1012** (không có) — chênh 0.009, trong khi mean-of-max là 0.30. Kết quả đưa vào `seq_score` làm nhiễu `s_comb` của mọi keyframe của video.

**Cách sửa:**
1. Dùng **OT không cân bằng/partial OT**: bỏ ràng buộc cột (chỉ ràng buộc hàng `a=1/M`, cột `≤ ε`) hoặc thêm khung "dustbin"; bắt buộc ε nhỏ (0.02) và ≥ 20 vòng lặp, hoặc thay bằng DP đơn điệu đã có (`viterbi_subevent_localization`) — bản chất bài toán là chuỗi thứ tự, DP chính xác hơn.
2. Chỉ tính trên cửa sổ ứng viên (±`max_gap_sec` quanh peak) và phạt theo **khoảng cách thời gian thực** `Δt`, không theo chỉ số `t/(T-1)`.
3. Test chấp nhận: `score(event present) − score(no event) ≥ 0.5 × (mean-of-max gap)`; nếu không đạt thì tắt T-SOT (`0.30 → 0.0`).
4. Cho tới khi sửa: đặt hệ số T-SOT = 0 (`multi_clause_blend = mv_set_sim`) và chạy A/B trên dev_full_859.

### B4 — Attention pooler đảo ngược mục tiêu (HIGH)
**Vì sao lỗi** (`query_guided_temporal_attention_score`): điểm = `cos(q, v̂_att)` với `v̂ = v/‖v‖`. Khi các khung **khác nhau** (hướng phụ trực giao), `‖v‖` co lại ⇒ cos **tăng**. Thử số với cos từng khung = 0.30: 5 khung giống nhau → **0.30**; 5 khung khác nhau (cắt cảnh) → **0.575**. Tức bonus thưởng video nhiều cảnh cắt – ngược với kỳ vọng "hành động nhất quán". Tuyên bố trong roadmap ("lồi → thưởng góc nhìn bổ trợ") là sai về toán học.
**Cách sửa:**
1. Không chuẩn hóa lại: dùng `s = Σ αₖ·cos(q, vₖ)` (attention-weighted mean cos) kết hợp phạt phân tán `− λ·(1 − ‖v‖)`; hoặc dùng tối đa-thấp `min_k` trên Top-2.
2. Chuyển thành **đặc trưng LTR** (cột 23) thay vì cộng tay `+0.025`, để trọng số do dữ liệu quyết định (và có train/infer parity).
3. Áp dụng cho **mọi** ứng viên (hoặc cho tới Top-K cố định cho cả lúc train) để không gây gián đoạn (B8).
4. Test: cùng cos, `coherent ≥ diverse`.

### B5 — Rò rỉ đánh giá `dev_150` (HIGH)
`dev_150` = `validation/` với `max_msrvtt=100`; `dev_full_859` = cùng thư mục, không giới hạn ⇒ `dev_150 ⊂ dev_full_859`. LTR cuối (`train_ltr_ranker_cv.py:427`) fit trên toàn bộ 859 ⇒ mọi số `dev_150` là **in-sample** với LTR. CV 5-fold chỉ cho metric OOF, nhưng benchmark cuối dùng weight fit-all.
**Cách sửa:** (a) khi báo `dev_150`, dùng trọng số LTR của fold không chứa câu đó (OOF) hoặc loại 150 câu khỏi train rồi báo; (b) coi `heldout_test_201` là số chính thức duy nhất (hiện **38.81%**); (c) ghi `ltr_trained_on` + `overlap_with_eval` vào `provenance.json` và làm `provenance gate` fail khi tập đánh giá giao tập huấn luyện.

### B6 — E5 caption bị chặn bởi cổng lexical (MEDIUM)
`q_e5_384` chỉ được mã hóa khi `sparse_lexical_yield AND has_lexical_modality_intent` (dòng 1969-1997); khối caption (dòng 2064) dùng `q_e5_384`. Câu thuần thị giác (đối tượng chính của caption Florence-2) sẽ không bao giờ gọi `_search_e5_caption_index`. **Sửa:** tách mã hóa E5: `need_e5_text = gate`, `need_e5_caption = e5_caption_index is not None`; mã hóa một lần nếu một trong hai cần (chi phí E5 ~ vài chục ms trên CPU, đo lại bằng benchmark latency). Bổ sung `rrf_vis_caption` thành đặc trưng LTR riêng (hiện chỉ cộng `0.5×` vào `rrf_vis`).

### B7 — Phase 7–8 chưa cải thiện độ chính xác (MEDIUM)
Dữ liệu `PROGRESS.md`/`benchmark_summary.json` (Hit@1):

| Tập | Phase 6 | Phase 8 (`8b4ac80`) | Δ |
|---|---|---|---|
| `dev_150` | 62.00% | 61.33% | −0.67 |
| `dev_full_859` | 41.68% | 41.44% | −0.24 |
| `heldout_test_201` | 39.30% | 38.81% | −0.49 |
| Non-Neural p50 (`dev_150`) | 24.27 ms | 35.72 ms | +47% |

Hiện chưa có bằng chứng thống kê cho lợi ích của LTR 22d/T-SOT/attention (khoảng tin cậy dev_150 ±7 điểm). **Việc cần làm:** ablation từng thành phần Phase 7–8 trên `dev_full_859` OOF (bật/tắt: LTR, T-SOT, attention, E5 gate, MV-SetSim), dùng paired bootstrap; giữ lại thành phần nào có Δ>0 có ý nghĩa, gỡ phần còn lại.

### B8 — Gián đoạn Top-15 của attention bonus (MEDIUM)
`for idx_m in range(min(15, len(cand_moments)))` chỉ cộng bonus cho 15 ứng viên đầu; ứng viên 16+ không được cộng ⇒ thang điểm không liên tục, và đặc trưng này không nằm trong tập train LTR. Sửa như B4.3.

### B9 — Benchmark cũ / số liệu không nhất quán (MEDIUM)
`benchmark/benchmark_report.md` và `benchmark_summary.json` ghi `git_sha 4742a7e`, `status completed_phase6` (14/11 15:07) — còn `PROGRESS.md` báo số Phase 8. README từng trộn các khoảng ("61.33%–62.00%"). **Sửa:** `run_validation_benchmark.py` phải ghi `git_sha` hiện tại + `ltr_model_sha256` + `config` vào summary; README chỉ trích **một** run cụ thể (kèm `run_id`); nếu muốn lưu lịch sử thì để ở `work/experiments/<run_id>/` (đã có).

### B10 — Lỗi tài liệu đã sửa
Đã sửa trong lượt này: `check-paths` → `scripts/verify_asset_roots.py`; `uvicorn …:app` → `python -m videntra serve --backend unified` (không có `app` ở module-level, chỉ có `create_app`); `/api/search` → `POST /v1/search`; `videntra search` mặc định `--backend visual` → thêm `--backend unified`; `search_with_metadata` yêu cầu `snapshot_id` và trả `dict`; gỡ hướng dẫn chạy `offline_processor.py`; ghi đúng trạng thái VietOCR/PhoWhisper.

### B11 — Nuốt lỗi
34 chỗ `except Exception: pass/return` (ví dụ `TextEmbeddingDiskCache._load/save`, nạp E5/caption index). Cache hỏng hoặc index thiếu bị bỏ qua im lặng ⇒ benchmark chạy "xanh" nhưng thiếu tín hiệu. **Sửa:** log `warning` kèm exception, thêm trường `diagnostics["degraded"]=[...]` trong kết quả, và để provenance gate ghi lại các kênh đã bị tắt.

---

## C. Điểm mới & điểm bứt phá thật sự của hệ thống

Chỉ liệt kê những điểm đã có code + test + dữ liệu hỗ trợ:

1. **Không gian 864d (SigLIP2 768 + COCO-IDF 80 + lưới 3×3 16)**: A1→A3 trong ablation Phase 6 cho thấy hybrid FTS5 + gating nâng `dev_150` Hit@1 từ 48.67% → 61.33%; đây là đóng góp lớn nhất đã được đo.
2. **Cổng chặn Pure-ASR** (`pure_asr_fp_top1_count = 0` trên cả 3 tập) và quy tắc "thiếu bằng chứng = unknown": giải quyết lỗi phổ biến của hệ thống ghép ASR.
3. **Sổ cái thời gian PTS** (`timeline/ledger.py`, không `timestamp×FPS` cho VFR) + **provenance gate** khóa SHA-256 ground truth: nghiêm ngặt hơn đa số prototype retrieval.
4. **Định vị khoảnh khắc**: mở rộng biên tương phản + neo theo anchor; Moment Hit@1 (IoU≥0.5) từ 4.55% (A3) lên 72.73% (A4) trên 22 câu QVH — *lưu ý n nhỏ, CI [54.6%, 90.9%]*.
5. **Kiến trúc offline/online tách bạch**: Kaggle GPU cho embedding nặng, CPU `mmap` cho phục vụ, cache JSONL phân rã truy vấn (`<1 ms`) — phù hợp RTX 3060 6GB.
6. **Small-LLM routing có fallback an toàn** (GGUF Qwen2.5-1.5B + rule parser). *Chưa* được chứng minh có lợi vì cache hiện là rule-only (xem hướng 1 bên dưới).

Chưa nên gọi là "bứt phá": T-SOT, attention pooler, `QAC-CSLS` (đã tắt vì giảm Hit@1), `snap_boundaries_to_shot_cuts` (tắt vì giảm Moment Hit@1), LTR 22d (chưa có ablation).

---

## D. Hướng nghiên cứu tiếp theo (ưu tiên theo giá trị / rủi ro)

**D0. Làm sạch & đo lại (1 ngày, bắt buộc trước mọi thứ khác).**
Sửa B1–B6, loại `dev_150` khỏi train LTR (hoặc dùng OOF), chạy ablation Phase 7–8 (B7), ghi lại benchmark bằng 1 run duy nhất.

**D1. Kích hoạt tín hiệu offline còn thiếu (Kaggle T4).**
(a) Chạy Qwen2.5-1.5B thật cho 1 060 câu (`prewarm_llm_query_cache.py --split all_1060`), đo lợi ích `llm_compact` trên nhóm câu đa mệnh đề; (b) `spatial_patches_fp16.npy` để bật `s_late_interaction`; (c) Florence-2 captions + E5 caption (đã có pipeline, cần sửa B6). *Giả thuyết kiểm chứng được:* `F1_VISUAL_CANDIDATE_MISS` (14.7% dev_150) giảm ≥ 3 điểm.

**D2. Nâng chất lượng thị giác ở tầng gốc (nơi trần 38–48% MSRVTT đang nằm).**
SigLIP2 zero-shot trên 1 khung/2 s là điểm nghẽn. Hướng: (i) lấy mẫu dày hơn quanh peak ở bước rerank (extract thêm 8 khung quanh top-15 offline); (ii) rerank Top-15 bằng cross-encoder video-text nhẹ (InternVideo2-1B/Video-LLaVA INT4 chạy trên Kaggle để chấm sẵn cho tập đánh giá, hoặc BLIP-2 ITM trên 6GB); (iii) fine-tune nhẹ (LoRA) projection SigLIP2 trên `dev_full_859` với group-split theo `source_video_id`.

**D3. Học phần dư thay vì cộng tay (đúng hướng, cần kỷ luật).**
Giữ residual pairwise LTR (đã đúng ý tưởng), nhưng: dùng LightGBM LambdaMART (đã có nhánh `_train_lightgbm_fold`) so với tuyến tính; hiệu chỉnh isotonic; báo cáo trên `heldout_test_201` chỉ **một lần** sau khi đóng băng cấu hình.

**D4. OCR/ASR tiếng Việt thực sự.**
Hoàn thiện B2, rồi đo `Lexical Recall@30` trên subset tiếng Việt (TextVR/AVC/TVR) trước-sau khi dùng VietOCR+PhoWhisper; thêm lexicon sửa lỗi dấu (so khớp gập dấu + edit-distance ≤ 1) cho FTS5.

**D5. Định vị khoảnh khắc nghiêm ngặt.**
Dùng QVHighlights đủ lớn (hiện chỉ 22 câu dev) để đo mAP/Moment nDCG đáng tin; thử DP đơn điệu (SE-Viterbi) mở rộng có đa đỉnh và NMS theo `tIoU`, thay các heuristic ngưỡng cố định (0.80/0.88 `min_peak_ratio`).

**D6. Quy trình đánh giá.**
Bootstrap theo `source_video_id`, báo cáo CI cho mọi so sánh, tự động fail khi (i) tập đánh giá giao tập huấn luyện, (ii) kênh nào bị tắt do exception, (iii) benchmark cũ hơn commit.
