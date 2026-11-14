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
   - Phân tích truy vấn (`RuleBasedQueryParser`, `decompose_query_clauses`, `extract_object_requirements`) có chuẩn hóa danh từ số nhiều COCO và tách mệnh đề tuần tự thời gian (`after` / `sau khi`).
   - Encode văn bản qua SigLIP 2 (`google/siglip2-base-patch16-224`, 768d) kèm dual-prompt ensemble (`0.75 * raw + 0.25 * "a video frame showing ..."`), bounded LRU RAM cache (512 entries, chuẩn hóa NFC, khử trùng batch) và mở rộng thành vector **864d** (`src/videntra/indexing/augmented_vector.py`).
2. **Stage 2 — FAISS 864d Retrieval & Sequential Alignment:**
   - Truy xuất ứng viên qua FAISS `IndexFlatIP` 864d, tính điểm căn chỉnh tuần tự đa mệnh đề (`align_sequential_keyframe_clauses`) có hệ số phạt đồng khung (`0.92`).
3. **Stage 3 — Specificity-Sorted SQLite FTS5 Lexical Search:**
   - Truy vấn kết nối SQLite Read-Only (`?mode=ro` được bảo vệ bởi `threading.RLock`), sắp xếp token theo độ đặc hiệu (cụm trích dẫn, danh từ riêng, độ dài từ, giới hạn 14 tokens).
4. **Stage 4–5 — Multimodal Temporal Clustering & Gated RRF Fusion:**
   - Gom cụm bằng chứng theo thời gian cục bộ (`lookup_true_visual_sim`), hợp nhất điểm bằng Order-Invariant RRF và áp dụng bộ lọc chặn `Pure-ASR False Positive Gate`.
5. **Stage 6–7 — Contrast Boundary Expansion, Diversity Aggregation & Relation Verification:**
   - Tinh chỉnh biên thời gian neo đỉnh (`expand_boundary_contrast` trong `src/videntra/localization/refine.py`), tổng hợp đa dạng theo video (`diverse_video` / `moment`).
   - Khi bật `--verify-relations` (CLI) hoặc `verify_relations=True` (API) ở chế độ `accurate`, pipeline trả về `relation_assessment` đối soát OCR nằm trong khoảng moment và ASR nằm trọn trong khoảng moment kèm metadata provenance.

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
| `videntra.indexing` | `augmented_vector.py` (864d subspace), `text_store.py` (SQLite FTS5), `visual_store.py`, `snapshot.py` |
| `videntra.ingestion` | `probe.py`, `jobs.py` (`IngestionJobManager` với resume an toàn & atomic write) |
| `videntra.localization` | `refine.py` (`refine_moment_boundaries`, `expand_boundary_contrast`) |
| `videntra.ocr` | `detection.py`, `recognition.py` (`GeneralOCRRecognizer`, `VietOCRRecognizer`), `routing.py`, `tracking.py` |
| `videntra.query` | `schemas.py` (`QueryGraph v1`), `rules.py`, `llm_parser.py` (zero-crash fallback) |
| `videntra.ranking` | `ranker.py` (`RelationAwareRanker`) |
| `videntra.registry` | `store.py`, `access.py` (`MediaResolver` kiểm soát allowlist, denylist, SHA-256 & `rights_status`) |
| `videntra.relations` | `matcher.py` (`GraphMatcher` logic tam trị), `entities.py`, `attributes.py`, `temporal.py` |
| `videntra.retrieval` | `unified_pipeline.py` (SSOT), `visual.py`, `branches.py`, `fusion.py`, `alignment.py`, `union.py` |
| `videntra.runtime` | `AdmissionController`, `DeadlineBudget` bảo vệ bộ nhớ/SLA |
| `videntra.sampling` | Hybrid Adaptive Sampling |
| `videntra.storage` | Quản lý đọc/ghi snapshot và Parquet |
| `videntra.timeline` | `ledger.py` (presentation-order PTS ledger & exact seek) |

---

## 4. Các Công cụ Vận hành & Kiểm toán (`scripts/`)
- **Benchmark & Ablation:** `run_validation_benchmark.py`, `evaluate_multimodal_ablations.py`, `evaluate.py`, `profile_text_encoder.py`.
- **Kaggle GPU Offload & Ingestion:** `package_kaggle_inputs.py`, `kaggle_multimodal_offline_worker.py` (pin SigLIP revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weights SHA-256 & PyAV), `ingest_kaggle_offline_results.py` (chặn ghi đè, kiểm tra preflight PTS/confidence).
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
