# Đặc Tả Lưu Trữ Ngoại Tuyến (Offline Storage & Schema Specification)

Tài liệu này mô tả kiến trúc lưu trữ thực tế, phân tách thư mục tự mô tả (`self-documenting`) và cấu trúc các bảng SQLite WAL + FTS5 được triển khai tại [`src/videntra/storage/database.py`](../src/videntra/storage/database.py) và [`src/videntra/config/paths.py`](../src/videntra/config/paths.py).

## 1. Phân Tách Lưu Trữ Vật Lý (`data/`)

### 1.1. Kho Dataset Tự Chứa Theo Phân Vùng (`data/datasets/<dataset_id>/`)
Mỗi bộ dữ liệu được quản lý độc lập qua CLI `videntra dataset` ([`src/videntra/registry/datasets.py`](../src/videntra/registry/datasets.py)) và đồng bộ với `data/datasets.json`:
- `data/datasets/msrvtt/` — tập niêm phong `MSR-VTT 1K-A` (`1,000` videos, chỉ dùng đánh giá cuối cùng)
- `data/datasets/msrvtt_val_tune/` — tập `Valid-Tune` (`2,000` videos, chỉ dùng tune siêu tham số)
- `data/datasets/msrvtt_val_confirm/` — tập `Valid-Confirm` (`2,000` videos, kiểm chứng độc lập & khóa Champion)

Mỗi thư mục `data/datasets/<dataset_id>/` chứa đầy đủ phân vùng khép kín:
- `media/` — video `.mp4` gốc bất biến (hoặc symlink/pointer hợp lệ ngoài Git)
- `keyframes/<video_id>/` — keyframes trích xuất bằng chiến lược mặc định **`hybrid_adaptive`** (`63-AC pHash` + Composite Sharpness Lookahead, JPEG theo 0-based presentation order)
- `catalog/corpus_metadata.db` — SQLite catalog riêng của dataset (khi chạy khép kín)
- `embeddings/` — vector embeddings theo từng backbone (`SigLIP2`, `VideoPrism`, `PE-Core-bigG`, `DFN5B-CLIP-H`, `PE-Core-L`, `SigLIP2-SO400M`, `E5`)
- `evidence/` — trích xuất đa phương thức ngoại tuyến (`OCR`, `ASR`, `YOLO`, `Florence-2`)
- `provenance/` & `snapshots/` — manifest kiểm toán và snapshot đồng bộ

### 1.2. Kho Chỉ Mục Đã Xử Lý (`data/processed/`) & Cache (`data/cache/`)
Quản lý tập trung qua `VIDENTRA_*` env vars và [`src/videntra/config/paths.py`](../src/videntra/config/paths.py) (kèm fallback tương thích ngược):
- `data/processed/catalog/corpus_metadata.db` — SQLite Metadata & Evidence Store (WAL + FTS5)
- `data/processed/visual_keyframes/`
  - `siglip2_keyframes_768d.faiss` — FAISS index 768d (Metric: Inner Product / Cosine)
  - `siglip2_yolo_augmented_864d.npy` — ma trận 864d gắn đặc trưng vật thể `YOLOv8n` + NMS (`IoU > 0.65`)
- `data/processed/visual_clips/`
  - `videoprism_whole_clips_768d.faiss` — vector toàn clip `VideoPrism-LvT-B`
  - `videoprism_subclips_768d.npy` & manifest — vector các cửa sổ sub-clip chuyển động
- `data/processed/text_evidence/`
  - `e5_ocr_asr_384d.faiss` & manifest — FAISS dense index cho văn bản `OCR`/`ASR` đa ngôn ngữ (`multilingual-e5-small`)
- `data/cache/`
  - `query_embeddings/` — cache vector truy vấn theo từng backbone
  - `query_plans/` — cache kế hoạch phân rã câu hỏi (`QueryGraph`)
  - `calibration/` — thống kê chuẩn hóa hubness (`siglip2_keyframe_hubness_radii.npy`, Query-Bank `train_9k_clean` priors)

*Lưu ý Zero Leakage:* Nhãn truy vấn đánh giá nằm tách biệt trong `validation/msrvtt/{valid_tuning,valid_confirm}/` và `test_queries/msrvtt/test_1ka_benchmark/`; runtime qua `MediaResolver` bị cấm đọc Ground Truth.

---

## 2. Cấu Trúc Các Bảng Trong `corpus_metadata.db` (SQLite WAL)

Tất cả truy cập đều bật `PRAGMA foreign_keys = ON;`, `PRAGMA journal_mode = WAL;` và đóng kết nối sau từng thao tác (`OfflineMetadataStore` trong [`src/videntra/storage/database.py`](../src/videntra/storage/database.py)):

1. **`Videos`** — Metadata video bất biến & chuẩn thời gian (`video_id` PK, `dataset`, `file_path`, `duration_ms`, `fps`, `time_base_num`, `time_base_den`, `width`, `height`, `total_frames`, `has_audio`, `status`, `created_at`, `updated_at`).
2. **`Keyframes`** — Các frame đã trích xuất (`keyframe_id` PK, `video_id` FK, `keyframe_index`, `source_frame_index` 0-based presentation order, `pts`, `time_base`, `timestamp_ms`, `keyframe_path`, `thumbnail_path`, `faiss_id` UNIQUE, `phash`, `sampling_reason`, `created_at`).
3. **`KeyframeMap`** — Bảng ánh xạ nhanh phục vụ truy xuất & xuất chuẩn AIC (`faiss_id` PK, `keyframe_id` UNIQUE FK, `video_id` FK, `pts_time`, `source_frame_index`, `keyframe_path`).
4. **`ModelVersions`** & **`IndexManifest`** — Sổ đăng ký phiên bản mô hình (`model_id`, `model_name`, `version`, `modality`, `weights_hash`) và manifest FAISS (`vector_dim`, `metric`, `total_vectors`, `index_file_path`, `checksum`, `is_active`).
5. **`ProcessingJobs`** — Trạng thái job ngoại tuyến theo từng stage (`job_id` PK, `video_id`, `stage`, `status`, `progress_pct`, `error_message`, `worker`, `started_at`, `completed_at`).
6. **`FrameObjects`** & **`FrameObjectRuns`** — Kết quả detection `YOLOv8n` gắn tọa độ không gian (`detection_id`, `video_id`, `keyframe_id`, `timestamp_ms`, `class_id`, `class_name`, `bbox_x1..y2`, `confidence`, `image_width`, `image_height`) cùng log chứng minh detector đã chạy (`FrameObjectRuns`) để guard phát hiện thiếu vật thể.
7. **Nhánh OCR (`OCRObservations`, `OCREvidenceGroups`, `OCREvidenceMembers`)** — Lưu nguyên bản quan sát đa mô hình (`raw_text`, `normalized_text`, `bbox_x1..y2`, `confidence`, `language`) và nhóm hợp nhất theo track không-thời gian (`status`: `agreed | partial | disputed | single_model`, `track_id`, `estimated_start_ms`, `estimated_end_ms`).
8. **Nhánh ASR (`ASRSegments`, `ASRWords`, `ASREvidenceGroups`, `ASREvidenceMembers`)** — Lưu transcript thoại theo đoạn & từ (`start_ms`, `end_ms`, `raw_text`, `normalized_text`, `confidence`, `speaker_id`) và nhóm hợp nhất kèm độ chồng lấn thời gian.
9. **Tìm kiếm Toàn văn FTS5 (`fts_ocr`, `fts_asr`)** — Bảng ảo SQLite FTS5 (`tokenize='unicode61'`) lập chỉ mục đồng thời `normalized_text` (có dấu) và `unaccented_text` (bỏ dấu), gắn `UNINDEXED` `video_id`, `group_id`, `keyframe_id`/`segment_id`, mốc thời gian `ms`.

---

## 3. Chiến Lược Đảm Bảo Toàn Vẹn Dữ Liệu

1. **Đồng bộ FAISS ↔ SQLite:** Số vector trong `siglip2_keyframes_768d.faiss` luôn đối chiếu chặt với khóa `faiss_id` trong `Keyframes` và `KeyframeMap`; sai lệch cấu trúc bị chặn sớm qua `videntra doctor --strict` và `videntra dataset verify`.
2. **Không Suy Diễn Sai Thời Gian VFR:** Timestamp luôn tính từ `pts * time_base` của stream decoded theo presentation order; `source_frame_index` là duy nhất 0-based.
3. **Tính Bất Biến & Tái Lập:** Dữ liệu mới bổ sung hoặc tái trích xuất sử dụng ID quyết định dựa trên `compute_recipe_hash(recipe_spec)` (`make_deterministic_record_id`), chạy trong transaction `BEGIN IMMEDIATE` có rollback tự động khi lỗi.
