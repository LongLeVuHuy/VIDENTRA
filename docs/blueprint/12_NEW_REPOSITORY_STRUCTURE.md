# 12 — Cấu Trúc Repository & Chính Sách Mã Nguồn

Repository chính thức của **VIDENTRA** nằm trên nhánh `main`, tổ chức theo kiến trúc tự mô tả (`self-documenting`), tách biệt hoàn toàn mã nguồn thực thi, tập validation/test và dữ liệu media lớn ngoài Git.

## 1. Cấu Trúc Thư Mục Hiện Hành

```text
VIDENTRA/
├── pyproject.toml                   # Package metadata, entrypoint CLI `videntra`, pytest settings
├── requirements.txt                 # Thư viện Python khóa tương thích CPU & CUDA 12.6
├── .env.example                     # Mẫu biến môi trường VIDENTRA_* (Device, DB, FAISS, LLM, VLM)
├── AGENTS.md                        # Quy tắc Zero Leakage 3 phân vùng, Timeline VFR & Clean Naming
├── README.md                        # Kiến trúc tổng quan, sơ đồ Offline/Online, công thức & benchmark
├── PROJECT.md                       # Danh mục chi tiết các package, scripts và milestones đã hoàn thành
├── .github/workflows/ci.yml         # CI tự động chạy `videntra doctor` + `pytest -q` trên Python 3.12
│
├── src/videntra/
│   ├── __main__.py & cli.py         # CLI `videntra` (`doctor`, `dataset`, `serve`, `search`, ...)
│   ├── config/                      # SSOT paths.py, device.py (VIDENTRA_* auto-detect), ingestion.py (`hybrid_adaptive`)
│   ├── contracts/                   # Typed schemas & JSON validators (`dataset`, `media_asset`, `frame_record`, ...)
│   ├── registry/                    # DatasetRegistryManager (`data/datasets/<id>/`), MediaResolver (GT isolation)
│   ├── ingestion/                   # PyAV probe, presentation-order decode, IngestionJobManager atomic resume
│   ├── timeline/                    # 0-based presentation-order PTS ledger & time_base mapping
│   ├── sampling/                    # HybridAdaptiveSampler (`hybrid_adaptive` + 63-AC pHash), OCR probes, compare
│   ├── encoders/                    # SigLIP2 & encoder abstractions
│   ├── ocr/                         # Text detection, GeneralOCR / VietOCR routing & tracking
│   ├── asr/                         # Silero VAD, Faster-Whisper / PhoWhisper & timeline alignment
│   ├── evidence/                    # Chuẩn hóa NFC / unaccented text & evidence store
│   ├── indexing/                    # Augmented 864d vectors (SigLIP2+YOLO NMS), FAISS, FTS5, snapshots
│   ├── query/                       # QueryComplexityGate, RuleBasedParser (<1ms), Qwen2.5-1.5B + 5-step validator
│   ├── retrieval/                   # unified_pipeline.py (SSOT), msrvtt_matrix.py, videoprism_branch.py, alignment.py
│   ├── relations/                   # SpatialActionVerifier (2D bbox + DP), SORT/ByteTrack Kalman tracker
│   ├── localization/                # Temporal proposal, boundary contrast refinement & Soft-NMS
│   ├── ranking/                     # vlm_reranker.py (`Qwen2.5-VL-3B` NF4), learned_ranker.py (32d LTR), vedje_reranker.py, cross_modal_adapter.py
│   ├── runtime/                     # AdmissionController & DeadlineBudget (bảo toàn 6GB VRAM)
│   ├── storage/                     # SQLite WAL + FTS5 OfflineMetadataStore (`database.py`)
│   ├── api/                         # FastAPI (`search`, `media` Range 206, `frames`, `exports`, Web Inspector UI)
│   └── evaluation/                  # Evaluator độc lập, provenance SHA-256, OOF GroupKFold, split lock (cô lập khỏi runtime)
│
├── scripts/                         # Điều phối Kaggle GPU, rebuild_corpus_hybrid_adaptive.py, run_msrvtt_matrix.py, benchmark keyframes
├── tests/                           # Bộ kiểm thử unit & integration phân theo đúng module `tests/<module>/`
├── configs/
│   ├── experiments/                 # Cấu hình thí nghiệm retrieval & benchmark
│   ├── ingestion/                   # Cấu hình sampling & decode
│   └── local/asset_roots.example.json
├── models/
│   ├── ranking/videntra_ltr.json    # Trọng số 32d Monotonic LightGBM (phiên bản nhỏ tracked trong Git)
│   └── {asr,detection,llm,ocr,siglip2}/  # Weights nặng ngoài Git (`.gitignore`)
├── benchmark/                       # Báo cáo & receipt JSON chính thức (`msrvtt_1ka_benchmark_*`)
├── data/                            # Dữ liệu runtime (media/FAISS/DB ngoài Git)
│   ├── datasets/<id>/               # Tập khép kín (`msrvtt`, `msrvtt_val_tune`, `msrvtt_val_confirm`)
│   ├── processed/                   # `catalog/`, `visual_keyframes/`, `visual_clips/`, `text_evidence/`
│   └── cache/                       # `query_embeddings/`, `query_plans/`, `calibration/`
├── validation/msrvtt/               # Nhãn `valid_tuning/` (2k) & `valid_confirm/` (2k) — tách khỏi test
├── test_queries/msrvtt/             # Nhãn `test_1ka_benchmark/` (1k niêm phong) — chỉ chấm benchmark cuối
├── work/                            # Artifact thực nghiệm protocol MSR-VTT & benchmark keyframe (ngoài Git phần nặng)
└── docs/
    ├── PROGRESS.md                  # Nhật ký tiến độ & các kết quả benchmark đã chạy
    ├── OFFLINE_STORAGE_SPEC.md      # Đặc tả phân vùng lưu trữ & schema SQLite WAL / FTS5
    ├── schemas/                     # JSON Schema contracts & fixtures mẫu
    └── blueprint/                   # Tài liệu kiến trúc cốt lõi (`05`, `06`, `10`, `12`)
```

## 2. Chính Sách Tổ Chức & Đặt Tên

1. **Tên chuẩn sạch, không gắn hậu tố phiên bản dư thừa:** Các phương pháp và file đã chốt sử dụng tên chuẩn trực tiếp (`hybrid_adaptive`, `videntra_ltr.json`, `rebuild_corpus_hybrid_adaptive.py`, `run_msrvtt_matrix.py`, `test_ranking_upgrades.py`). Chỉ giữ alias tương thích ngược trong `config/` để không làm gãy snapshot cũ.
2. **Đường dẫn runtime qua `VIDENTRA_*` & `paths.py`:** Không hardcode đường dẫn cá nhân hay đường dẫn legacy; mọi artifact runtime đi qua `src/videntra/config/paths.py` và biến môi trường `.env`.
3. **Cách ly tuyệt đối Ground Truth & Heavy Binaries:**
   - Module `src/videntra/retrieval/`, `sampling/`, `ranking/` không bao giờ đọc thư mục `validation/` hay `test_queries/` hoặc nhận GT timestamp.
   - `raw media`, `.db`, `.faiss`, `.npy`, `.pt`, `.venv`, `.env`, `work/` nặng không đưa vào Git.
4. **Kiểm thử theo module:** Mọi thay đổi mã nguồn phải giữ mã gọn, không lặp logic, xóa sạch file/thư mục tạm phát sinh sau khi chạy, và kiểm chứng bằng `python -m videntra doctor --strict`, `python -m videntra dataset verify`, `pytest -q`.
