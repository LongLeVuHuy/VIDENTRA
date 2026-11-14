# Project: VIDENTRA — Video Intelligence, Evidence & Temporal Retrieval Architecture

## 1. Kiến trúc Tổng thể & Triết lý Thiết kế
- **Single Source of Truth (SSOT):** Toàn bộ logic tìm kiếm đa phương thức hợp nhất được tập trung tại [`src/videntra/retrieval/unified_pipeline.py`](src/videntra/retrieval/unified_pipeline.py) (`UnifiedMultimodalSearchPipeline`), phục vụ đồng thời cho REST API (`POST /v1/search`), CLI (`videntra search --backend unified`) và hệ thống Evaluator.
- **Cấu hình Môi trường Runtime (`VIDENTRA_*`):** Tất cả đường dẫn tới cơ sở dữ liệu, chỉ mục, mô hình và thiết bị (`VIDENTRA_DB_PATH`, `VIDENTRA_FAISS_PATH`, `VIDENTRA_MODEL_DIR`, `VIDENTRA_DEVICE`, `VIDENTRA_ASSETS_ROOT`) đều hỗ trợ cấu hình động qua biến môi trường hoặc tham số CLI; không phụ thuộc đường dẫn cứng máy cá nhân.
- **Timeline Ledger Chính xác:** Mọi frame index đều tuân theo presentation order (zero-based) được giải mã tuần tự qua PyAV và lưu vết `PTS` / `time_base`, tuyệt đối không ước lượng `timestamp × FPS` trên nguồn VFR.
- **Ngữ nghĩa Bằng chứng Tam trị:** Quan hệ ngữ nghĩa trong `QueryGraph v1` được kiểm chứng qua `GraphMatcher` (`SUPPORTED`, `PARTIAL`, `CONTRADICTED`, `UNKNOWN`). Thiếu bằng chứng trả về `UNKNOWN`; không suy đoán danh tính người nói từ ASR nếu thiếu căn cứ đối chiếu.
- **Tách biệt Ground Truth (Isolation):** Runtime bị cấm truy cập trực tiếp các đường dẫn nhãn thông qua `MediaResolver` (`denied_roots`) và được kiểm chứng tự động qua `tests/evaluation/test_isolation.py`.

---

## 2. Luồng Xử lý Dữ liệu (7-Stage Unified Pipeline)
1. **Stage 1 — Query Analysis & Vector Generation:**
   - Phân tích truy vấn (`RuleBasedQueryParser`, `decompose_query_clauses`, `extract_object_requirements`) có chuẩn hóa danh từ số nhiều/vai trò COCO (`cyclist`, `skier`, `surfer`, `skater`, `motorcyclist`), ánh xạ hành động–vật thể (`driving -> car`, `tennis -> tennis racket`, `cutting -> knife`, ...) và bộ lọc `_is_common_sentence_initial_word` (`_COMMON_INITIAL_OPENERS`, `_COCO_VOCAB_WORDS`) ngăn nhận nhầm từ viết hoa đầu câu thành danh từ riêng.
   - Encode văn bản qua SigLIP 2 (`google/siglip2-base-patch16-224`, 768d) kèm dual-prompt ensemble (`0.75 * raw + 0.25 * "a video frame showing ..."`), bounded LRU RAM cache (512 entries, chuẩn hóa NFC, khử trùng batch) và mở rộng thành vector **864d** (`src/videntra/indexing/augmented_vector.py`).
2. **Stage 2 — FAISS Retrieval, 768d+864d Multi-Clause Blending & Sequential Alignment:**
   - Truy xuất ứng viên qua FAISS `IndexFlatIP`, pha trộn đồng nhất **768d Raw SigLIP 2 + 864d Augmented (`0.70 / 0.30`)** cho cả câu đơn lẫn đa mệnh đề (`sub_clauses`), và căn chỉnh tuần tự đa mệnh đề (`align_sequential_keyframe_clauses`) có hệ số phạt đồng khung (`0.92`).
3. **Stage 3 — Specificity-Sorted SQLite FTS5 Lexical Search:**
   - Truy vấn kết nối SQLite Read-Only (`?mode=ro` được bảo vệ bởi `threading.RLock`), sắp xếp token theo độ đặc hiệu (cụm trích dẫn, danh từ riêng, độ dài từ, giới hạn 14 tokens), hỗ trợ sinh gốc từ bảo thủ (`_generate_morphological_stems` cho từ $\ge 8$ ký tự) và nối bigram liền kề chịu lỗi dính chữ OCR (`compound_tokens`).
4. **Stage 4–5 — Multimodal Temporal Clustering, Order-Invariant RRF & Visual-Floor Gate:**
   - Gom cụm bằng chứng theo thời gian cục bộ (`lookup_true_visual_sim`), hợp nhất điểm bằng Order-Invariant RRF và áp dụng cổng chặn **Pure-ASR False Positive Gate** (kèm kiểm tra nhất quán số học `q_numbers`), **Visual-Floor Lexical Gate** (`s_vis_floor = s_vis_max - 0.065`), `single_common_only` (câu đơn/dài khớp đơn `m_high == 1`) và giảm nhiễu `gap_damp` khi hụt tương đồng thị giác $> 0.020$.
5. **Stage 6–7 — Local Window Coherence, Outlier Spike Guard, Contrast Boundary Expansion & Relation Verification:**
   - Tích hợp độ nhất quán cửa sổ cục bộ (`local_window_mean` trong `±3.5s`), bộ lọc đỉnh ngoại lai 1 khung hình thuần thị giác (`_video_sorted_visual_scores` + `tail_adjustment`), chuẩn hóa `density_bonus` cho video dài tín hiệu yếu (`n_kf > 30 and s_max < 0.225`), tinh chỉnh biên neo đỉnh (`expand_boundary_contrast` trong `src/videntra/localization/refine.py`), và tổng hợp đa dạng (`diverse_video` / `moment`).
   - Khi bật `--verify-relations` (CLI) hoặc `verify_relations=True` (API), `_verify_relations()` truy xuất OCR/ASR kèm `FrameObjects` 2D bounding boxes trong khoảng moment, hiệu chuẩn `ASRSegments.confidence` qua `calibrate_asr_confidence()`, kiểm chứng quan hệ không gian 2D cùng frame (`left_of`, `right_of`, `above`, `below`, `inside`, `near`) và tùy chọn rerank qua `rerank_with_relations`.

---

## 3. Danh mục Modules Cốt lõi (`src/videntra/`)

| Package | Thành phần & Chức năng chính |
| :--- | :--- |
| `videntra.api` | `app.py`, `search.py`, `media.py` (HTTP 206 Range), `frames.py` (exact seek headers), `exports.py` (CSV safe), `ui_html.py` |
| `videntra.asr` | `vad.py` (Voice Activity Detection), `transcription.py` (`PhoWhisperTranscriber`), `alignment.py` (`TimelineAligner`) |
| `videntra.config` | Cấu hình Ingestion, Sampling và thiết lập runtime |
| `videntra.contracts` | `schemas/` (JSON Schemas) và bộ validator cho `dataset`, `media_asset`, `frame_record`, `query_graph`, `result` |
| `videntra.encoders` | `siglip2.py` (`SigLIP2Encoder`, `MockSigLIP2Encoder`) |
| `videntra.evaluation` | `evaluator.py` (Hit@K, MRR, nDCG, mAP, Bootstrap CI), `provenance.py`, `split_lock.py`, `corpus_gate.py` |
| `videntra.evidence` | `store.py` (đọc/ghi `OCRTrackRecord` và `ASRSegmentRecord` dạng Parquet) |
| `videntra.indexing` | `augmented_vector.py` (864d subspace, Temporal Motion Delta, Multi-scale Context, QAT-LateInteraction `MaxSim`), `text_store.py` (SQLite FTS5), `visual_store.py`, `snapshot.py` |
| `videntra.ingestion` | `probe.py`, `jobs.py` (`IngestionJobManager` với resume an toàn & atomic write) |
| `videntra.localization` | `refine.py` (`refine_moment_boundaries`, `expand_boundary_contrast`) |
| `videntra.ocr` | `detection.py`, `recognition.py` (`GeneralOCRRecognizer`, `VietOCRRecognizer`), `routing.py`, `tracking.py` |
| `videntra.query` | `schemas.py` (`QueryGraph v1`, `ModalityRoutingPlan`, `CompactDecomposedQuery`), `rules.py`, `canonicalize.py`, `llm.py` (`AdaptiveQueryDecomposer`, `LocalSmallLLMBackend`, `QueryComplexityGate`, `SemanticDecomposerValidator`, RAM+Disk cache & `<1ms` fallback) |
| `videntra.ranking` | `ranker.py` (`RelationAwareRanker`), `learned_ranker.py` (`LearnedRankerLTR` pure-NumPy), `calibration.py` (`calibrate_asr_confidence`, `calibrate_multimodal_scores`), `features.py` (18d LTR features & `compute_query_z_scores`) |
| `videntra.registry` | `store.py`, `access.py` (`MediaResolver` kiểm soát allowlist, denylist, SHA-256 & `rights_status`) |
| `videntra.relations` | `matcher.py` (`GraphMatcher` logic tam trị), `predicates.py` (2D spatial cùng frame & `action_observation`), `entities.py`, `attributes.py`, `temporal.py` |
| `videntra.retrieval` | `unified_pipeline.py` (SSOT, wire E5 + Temporal Context + LTR residual + `final_results.sort`), `visual.py`, `branches.py`, `fusion.py`, `alignment.py` (Sequential DP, `CM-GD` diffusion/CSLS, `SE-Viterbi`), `union.py` |
| `videntra.runtime` | `AdmissionController`, `DeadlineBudget` bảo vệ bộ nhớ/SLA |
| `videntra.sampling` | Hybrid Adaptive Sampling |
| `videntra.storage` | Quản lý đọc/ghi snapshot và Parquet |
| `videntra.timeline` | `ledger.py` (presentation-order PTS ledger & exact seek) |

---

## 4. Các Công cụ Vận hành & Kiểm toán (`scripts/`)
- **Benchmark & Ablation:** `run_validation_benchmark.py`, `evaluate_multimodal_ablations.py`, `evaluate.py`, `profile_text_encoder.py`.
- **Kaggle GPU Offload & Ingestion:** `package_kaggle_inputs.py`, `kaggle_multimodal_offline_worker.py` (pin SigLIP revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weights SHA-256, PyAV, $3\times 3$ spatial patches `float16`, `multilingual-e5-small` 384d FAISS & `PQ-VCA` Mode B `Florence-2-base` captioning), `ingest_kaggle_offline_results.py` (chặn ghi đè, kiểm tra preflight PTS/confidence), `build_e5_text_index.py` (tạo `faiss_text_e5_384.bin` & `PQ-VCA` Mode A synopses), `train_ltr_ranker_cv.py` (5-fold `GroupKFold` CV theo `source_video_id` + Monotonic Sign Projection), `prewarm_llm_query_cache.py`.
- **Kiểm toán Provenance & Timeline (Read-Only):** `audit_model_index.py`, `audit_decoded_timeline.py`, `audit_kaggle_provenance.py`, `audit_qvh_coverage.py`, `audit_qvh_a5_readiness.py`, `audit_timeline_dependencies.py`, `propose_timeline_repairs.py`.
- **Quản lý Corpus & Split Lock:** `acquire_qvh_quarantine.py`, `freeze_qvh_source_split.py`, `export_qvh_inference_queries.py`, `evaluate_qvh_frozen_predictions.py`, `verify_asset_roots.py`.

---

## 5. Trạng thái Các Cột mốc (Milestones)

| Cột mốc | Phạm vi chính | Trạng thái |
| :--- | :--- | :--- |
| **M0–M2** | Contracts, Registry Isolation, Timeline Ledger, Hybrid Sampling, Visual SigLIP 2 & FAISS | **DONE** |
| **P0–P2 (Audit & SSOT)** | B1–B11, C1–C2, P3 (864d & FTS5 fallback), SSOT `UnifiedMultimodalSearchPipeline` | **DONE** |
| **P3–P4 (Evaluation Suite)** | E1–E5 (Multi-split `dev_150` / `dev_full_859` / `heldout_test_201`, Bootstrap CI, Taxonomy F1–F5, A0–A5 Ablations) | **DONE** (báo cáo tại `benchmark/benchmark_report.md`) |
| **Provenance & Hardening** | Thread-safe `search_with_metadata`, CLI/API parity (`--backend unified`, `--verify-relations`), `split_lock`, `corpus_gate`, `provenance`, Kaggle A5 pinned worker | **DONE** |
| **Production Cleanup & Repo Release** | Dọn dẹp file tạm & legacy scripts, thay thế mọi hardcoded personal paths bằng `VIDENTRA_*`, `requirements.txt`, `README.md`, `.gitkeep` + `.gitignore`, Security Scan & GitHub Push | **DONE** (381/381 tests PASS) |
| **Phase 5 — Accuracy & Relation Verifier Upgrade** | Stage 1/2 multi-clause 768d+864d blending, Stage 5 Visual-Floor Lexical Gate, Stage 6 local window coherence, 2D spatial/action grounding, `calibration.py`, `FrameObjects` evidence & optional `rerank_with_relations` | **DONE** (`dev_150` Hit@1 62.00%, 399/399 tests PASS) |
| **Phase 6 — De-Overfitting & Generalization Upgrade** | Chẩn đoán Overfitting (`dev_150` vs `dev_full_859` vs `heldout_test_201`), tổng quát hóa proper-noun/stemming/OCR-bigram, chặn Pure-ASR số học (`Pure-ASR FP = 0` trên cả 3 tập), chống đỉnh ngoại lai 1 khung hình Stage 6 & chuẩn hóa `requirements.txt` duy nhất | **DONE** (`dev_150` 62.00%, `dev_full_859` 41.68%, `heldout_test_201` 39.30%, 402/402 tests PASS) |
| **Phase 7 — Master Upgrade, Audit Fixes & Research Foundation** | Sửa 6 Bugs + 7 Vấn đề kiểm toán (Train–Inference Parity, Monotonic Sign Projection cho `videntra_ltr_v1.json`, `final_results.sort`, wire Dense E5 + `temporal_context_768`, `DeadlineStoppingCriteria` + Dynamic `qint8` cho `Qwen2.5-1.5B-Instruct`, fallback parity, provenance exit 0) & nền tảng `PQ-VCA` / `CM-GD` / `SE-Viterbi` | **DONE (Code + 462 Tests + Provenance Exit 0)** (`462/462` tests PASS; LTR 5-fold CV Hit@1 `0.4558`; `dev_150` exit 0: Video Hit@1 `59.33%`, `Pure-ASR FP = 0`; chờ chạy Kaggle GPU worker cho `spatial_patches_fp16.npy` & `PQ-VCA` / `Qwen2.5-1.5B` prewarm) |
