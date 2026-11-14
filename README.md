# VIDENTRA

**Video Intelligence, Evidence & Temporal Retrieval Architecture**  
*Find the moment. Understand the relation.*

VIDENTRA là hệ thống truy hồi khoảnh khắc video đa phương thức (Multimodal Video Moment Retrieval) tối ưu cho tiếng Việt và tiếng Anh, hỗ trợ phân tích quan hệ ngữ nghĩa (`QueryGraph v1`), trích xuất bằng chứng độc lập (**Visual**, **OCR**, **ASR**, **Object Detection**), suy luận logic tam trị (`SUPPORTED` / `CONTRADICTED` / `UNKNOWN`), phát lại HTML5 HTTP 206 Range Streaming và đối soát khung hình chính xác theo timeline (`presentation-order decoded frame ledger`).

---

## Mục lục

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc Thư mục Dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Hướng dẫn Cài đặt từ `requirements.txt`](#3-hướng-dẫn-cài-đặt-từ-requirementstxt)
4. [Cấu hình Dữ liệu, Trọng số Mô hình & Biến môi trường](#4-cấu-hình-dữ-liệu-trọng-số-mô-hình--biến-môi-trường)
5. [Hướng dẫn Vận hành Hệ thống (CLI, API & Web UI)](#5-hướng-dẫn-vận-hành-hệ-thống-cli-api--web-ui)
6. [Kiểm thử Tự động & Công cụ Kiểm toán](#6-kiểm-thử-tự-động--công-cụ-kiểm-toán)
7. [Hiệu năng & Giới hạn Kỹ thuật đã Ghi nhận](#7-hiệu-năng--giới-hạn-kỹ-thuật-đã-ghi-nhận)

---

## 1. Tổng quan Kiến trúc

Hệ thống vận hành theo mô hình **Single Source of Truth (SSOT)** thông qua `UnifiedMultimodalSearchPipeline` (`src/videntra/retrieval/unified_pipeline.py`), dùng chung xuyên suốt giữa CLI, FastAPI Server và bộ Evaluator:

1. **Ingestion & Presentation-Order Timeline Ledger (`src/videntra/timeline/`, `src/videntra/ingestion/`):**
   - Giải mã video tuần tự theo thứ tự hiển thị (zero-based presentation ordinal), ghi nhận `PTS` và `time_base` gốc qua PyAV (`av==19.0.2`), tuyệt đối không suy diễn `timestamp × FPS` trên video VFR.
   - Lấy mẫu khung hình thích ứng lai (`Hybrid Adaptive Sampling`) và trích xuất khung hình chính xác mức bit.
2. **Trích xuất Bằng chứng Đa phương thức Độc lập (`src/videntra/ocr/`, `src/videntra/asr/`, `src/videntra/evidence/`):**
   - **Visual:** `SigLIP 2` (`google/siglip2-base-patch16-224`, 768d) mã hóa hình ảnh và truy vấn văn bản song ngữ Anh–Việt.
   - **OCR:** Phát hiện vùng chữ kết hợp `OCRRouter` tự động định tuyến sang `VietOCRRecognizer` (khi có ký tự tiếng Việt có dấu) hoặc `GeneralOCRRecognizer` (`rapidocr-onnxruntime`), gom track theo không gian–thời gian (`OCRTrackRecord`).
   - **ASR:** Phát hiện tiếng nói (VAD) kết hợp `PhoWhisperTranscriber` / `faster-whisper`, căn chỉnh thời gian hai chiều (`TimelineAligner`).
   - **Object Detection & Subspace 864d (`src/videntra/indexing/augmented_vector.py`):** Chiếu xuất đặc trưng đối tượng COCO IDF và tín hiệu văn bản/metadata để mở rộng vector 768d thành không gian **864 chiều chuẩn hóa (Decoupled Subspace Scaling)**, lập chỉ mục trên **FAISS `IndexFlatIP`** và **SQLite FTS5** (chế độ persistent read-only).
3. **Pipeline Tìm kiếm 7 Chặng (`UnifiedMultimodalSearchPipeline`):**
   - **Stage 1:** Phân tích truy vấn (`RuleBasedQueryParser` + plural/COCO role & supplemental action-object mappings) & tạo vector 864d với dual-prompt ensemble và LRU RAM cache (512 entries, chuẩn hóa NFC).
   - **Stage 2:** Truy xuất FAISS IP, pha trộn đồng nhất **768d + 864d (`0.70 / 0.30`)** cho cả câu đơn và đa mệnh đề (`sub_clauses`), kết hợp căn chỉnh tuần tự đa mệnh đề (`align_sequential_keyframe_clauses`).
   - **Stage 3:** Truy vấn từ vựng SQLite FTS5 theo độ đặc hiệu token (quoted phrases, proper nouns, length-sorted tokens, giới hạn tối đa 14 tokens).
   - **Stage 4–5:** Gom cụm đa phương thức cục bộ theo thời gian, tính điểm hợp nhất bất biến thứ tự (Order-Invariant RRF), cổng chặn **Pure-ASR False Positive Gate** và **Visual-Floor Lexical Gate** (`s_vis_floor = s_vis_max - 0.065`).
   - **Stage 6–7:** Tính độ nhất quán cửa sổ cục bộ (`local_window_mean` `±3.5s`), mở rộng biên khoảnh khắc dựa trên neo đỉnh tương đồng (`expand_boundary_contrast`), tổng hợp đa dạng (`diverse_video` / `moment`), và đối chiếu đồ thị quan hệ (`GraphMatcher`, `verify_relations` tích hợp OCR, ASR calibrated + 2D `FrameObjects` cùng frame, hỗ trợ tùy chọn `rerank_with_relations`).
4. **Logic Tam trị & Bảo vệ Ground Truth Isolation:**
   - Thiếu bằng chứng trả về **`UNKNOWN`** (không bao giờ tự động coi là `FALSE`); lời thoại ASR không tự chứng minh danh tính người nói nếu thiếu neo đối soát.
   - `MediaResolver` (`src/videntra/registry/access.py`) thực thi allowlist cho thư mục media và chặn truy cập (`denied_roots`) đối với các thư mục đánh giá/ground truth (`validation/`, `test_queries/`, `work/evaluation-only/`).

---

## 2. Cấu trúc Thư mục Dự án

> **Lưu ý về cấu trúc GitHub:** Để giữ repository gọn nhẹ chuẩn production, toàn bộ dữ liệu nặng (video thô, trọng số mô hình, chỉ mục FAISS/SQLite, cache, kết quả chạy thử, bộ phân tách ground truth) được **bỏ qua khỏi Git**, nhưng cấu trúc thư mục vẫn được duy trì sẵn thông qua các tệp `.gitkeep`.

```text
VIDENTRA/
├── src/videntra/               # Mã nguồn lõi Python package
│   ├── api/                    # FastAPI REST API routes, HTTP 206 streaming, Web UI HTML, CSV/JSON export
│   ├── asr/                    # Voice Activity Detection, PhoWhisper/Faster-Whisper ASR & TimelineAligner
│   ├── config/                 # Cấu hình ingestion, sampling và tham số hệ thống
│   ├── contracts/              # JSON Schema validators cho dataset, media_asset, frame_record, query_graph, result
│   ├── encoders/               # SigLIP 2 encoder (real CPU/GPU + deterministic mock cho test)
│   ├── evaluation/             # Evaluator IR metrics, Bootstrap CI, provenance, split_lock & corpus_gate
│   ├── evidence/               # Lưu trữ & quản lý Parquet/SQLite cho bằng chứng OCR/ASR
│   ├── indexing/               # Vector FAISS store, 864d augmented subspace, SQLite FTS5 & SnapshotManifest
│   ├── ingestion/              # Pipeline ingest media, probe FFmpeg/PyAV, job manager hỗ trợ resume
│   ├── localization/           # Tinh chỉnh biên thời gian & contrast boundary expansion
│   ├── ocr/                    # Text detection, tracking & routing (RapidOCR / VietOCR)
│   ├── query/                  # RuleBasedQueryParser, LLMQueryParser (fallback an toàn) & QueryGraph v1
│   ├── ranking/                # RelationAwareRanker & Score Calibration (`calibration.py`: ASR logprob/temperature sigmoid)
│   ├── registry/               # MediaResolver (allowlist/denylist), registry SQLite & SHA-256 verification
│   ├── relations/              # GraphMatcher tam trị, 2D spatial cùng frame, action_observation & đối tượng COCO
│   ├── retrieval/              # UnifiedMultimodalSearchPipeline (SSOT), branches, fusion & alignment
│   ├── runtime/                # AdmissionController, DeadlineBudget & telemetry
│   ├── sampling/               # Hybrid Adaptive Sampling
│   ├── storage/                # Quản lý lưu trữ Parquet / snapshot / ZSTD
│   ├── timeline/               # Presentation-order frame ledger & bit-exact seek
│   └── cli.py                  # Điểm vào dòng lệnh `videntra` / `python -m videntra`
├── configs/
│   ├── experiments/            # Cấu hình thực nghiệm (e01.json, e02.json)
│   ├── ingestion/              # Cấu hình ingestion chuẩn (m1_default.json)
│   └── local/
│       └── asset_roots.example.json  # Mẫu ánh xạ thư mục dữ liệu cục bộ (copy thành asset_roots.json)
├── manifests/                  # Manifest chuẩn định danh tập dữ liệu và clip mẫu
├── scripts/                    # Công cụ vận hành: Kaggle worker, ingest kết quả, benchmark, audit provenance/timeline
├── tests/                      # Bộ kiểm thử tự động toàn diện (unit, integration, parity, isolation)
├── benchmark/                  # Báo cáo kết quả benchmark tổng hợp (benchmark_report.md, benchmark_summary.json)
├── docs/                       # Tài liệu thiết kế Blueprint, JSON Schemas, Audit reports & PROGRESS.md
├── data/                       # [Local Only - .gitkeep] Dữ liệu raw/, processed/ (metadata.db, faiss_index.bin), cache/
├── models/                     # [Local Only - .gitkeep] Trọng số cục bộ: siglip2/, ocr/, asr/, detection/
├── work/                       # [Local Only - .gitkeep] Dữ liệu ingest trung gian, fixtures & benchmark runs
├── validation/                 # [Local Only - .gitkeep] Bộ ground truth đánh giá (tách biệt khỏi runtime)
├── test_queries/               # [Local Only - .gitkeep] Bộ truy vấn kiểm thử giữ kín (heldout)
├── requirements.txt            # Danh sách thư viện Python Production đã pin phiên bản
├── pyproject.toml              # Cấu hình đóng gói & entry point CLI (`videntra`)
├── PROJECT.md                  # Tài liệu kiến trúc kỹ thuật & danh mục module/feature
└── README.md                   # Hướng dẫn cài đặt và vận hành hệ thống
```

---

## 3. Hướng dẫn Cài đặt từ `requirements.txt`

### 3.1. Yêu cầu tiên quyết
- **Hệ điều hành:** Windows 10/11 (64-bit) hoặc Linux (x86_64).
- **Python:** Phiên bản **3.12** (`>=3.12,<3.13`).
- **FFmpeg & FFprobe:** Đã cài đặt và thêm vào biến môi trường `PATH` (`ffmpeg -version` chạy thành công).
- **Phần cứng:**
  - Mục tiêu đề xuất: NVIDIA GPU >= 6GB VRAM (ví dụ RTX 3060), RAM >= 16–32 GB.
  - Chế độ suy luận CPU-only vẫn hỗ trợ đầy đủ với SigLIP 2, FAISS CPU và SQLite FTS5 (RAM ~420–580 MB khi truy vấn).

### 3.2. Các bước cài đặt trên Windows (PowerShell)

```powershell
# 1. Clone repository từ GitHub
git clone https://github.com/LongLeVuHuy/VIDENTRA.git
cd VIDENTRA

# 2. Tạo và kích hoạt môi trường ảo Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Nâng cấp pip và cài đặt toàn bộ thư viện từ requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Cài đặt gói VIDENTRA ở chế độ editable (kích hoạt lệnh `videntra`)
pip install --no-build-isolation -e .

# 5. Kiểm tra xung đột thư viện và CLI entry point
pip check
videntra --help
```

### 3.3. Các bước cài đặt trên Linux / macOS

```bash
git clone https://github.com/LongLeVuHuy/VIDENTRA.git
cd VIDENTRA

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation -e .

pip check
videntra --help
```

> **Ghi chú về GPU (CUDA):** Mặc định `requirements.txt` sử dụng `torch` tiêu chuẩn tương thích cả CPU/GPU theo môi trường pip. Nếu chạy suy luận hoặc embedding quy mô lớn trên máy có card NVIDIA, hãy cài `torch` bản CUDA (ví dụ `--index-url https://download.pytorch.org/whl/cu124`) trước khi chạy `pip install -r requirements.txt`, hoặc sử dụng GPU worker Kaggle kèm sẵn trong `scripts/kaggle_multimodal_offline_worker.py`.

---

## 4. Cấu hình Dữ liệu, Trọng số Mô hình & Biến môi trường

### 4.1. Tạo tệp cấu hình thư mục dữ liệu (`configs/local/asset_roots.json`)
Sao chép tệp mẫu và tùy chỉnh đường dẫn nếu thư mục dữ liệu nằm ở vị trí khác:

```powershell
Copy-Item configs\local\asset_roots.example.json configs\local\asset_roots.json
```

Mặc định tệp cấu hình trỏ vào cấu trúc thư mục chuẩn bên trong `VIDENTRA`:
- `roots`: `data/raw/<dataset>/media`, `work/m1_fixtures`
- `denied_roots`: `["work/evaluation-only", "validation", "test_queries"]` (đảm bảo runtime tìm kiếm không bao giờ đọc nhầm dữ liệu ground truth).

### 4.2. Chuẩn bị Trọng số Mô hình (Model Checkpoints)
Đặt trọng số mô hình vào thư mục `models/` tương ứng trước khi chạy suy luận thật (không cần tải weights nếu chỉ chạy unit tests bằng mock hoặc kiểm thử fixture):
- `models/siglip2/` — SigLIP 2 Base Patch16-224 (`google/siglip2-base-patch16-224`, revision pin `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`).
- `models/ocr/` — Mô hình nhận diện ký tự (`paddleocr_en/`, `vietocr/`).
- `models/asr/` — Mô hình chuyển giọng nói thành văn bản (`phowhisper_small/` hoặc `faster-whisper-small`).
- `models/detection/` — Trọng số phát hiện đối tượng (`yolov8n.pt`).

### 4.3. Biến môi trường hỗ trợ (`VIDENTRA_*`)
Hệ thống cho phép ghi đè toàn bộ đường dẫn runtime qua biến môi trường thay cho tham số dòng lệnh:

| Biến môi trường | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- |
| `VIDENTRA_DB_PATH` | `data/processed/metadata.db` | Đường dẫn tới cơ sở dữ liệu SQLite FTS5 & metadata chỉ mục |
| `VIDENTRA_FAISS_PATH` | `data/processed/faiss_index.bin` | Đường dẫn tới tệp chỉ mục vector FAISS |
| `VIDENTRA_MODEL_DIR` | `models/siglip2` | Thư mục chứa checkpoint mô hình SigLIP 2 |
| `VIDENTRA_DEVICE` | `cpu` | Thiết bị chạy encoder (`cpu` hoặc `cuda`) |
| `VIDENTRA_ASSETS_ROOT` | `data/raw` | Thư mục gốc chứa raw datasets cho các script chuẩn bị dữ liệu |
| `VIDENTRA_QVH_MEDIA_DIR` | `data/raw/qvhighlights/media` | Thư mục chứa video media của QVHighlights |
| `VIDENTRA_ASR_MODEL_DIR` | `models/asr/phowhisper_small` | Thư mục chứa checkpoint mô hình ASR |

---

## 5. Hướng dẫn Vận hành Hệ thống (CLI, API & Web UI)

Tất cả câu lệnh đều có thể gọi qua `videntra ...` hoặc `python -m videntra ...`.

### 5.1. Kiểm tra nhanh bằng Dữ liệu Fixture (Không cần tải dataset/model nặng)

```powershell
# Kiểm tra toàn bộ manifest và JSON schema
videntra manifests validate --all

# Tạo bộ video fixture tổng hợp để kiểm thử end-to-end
videntra fixtures build --output work/fixtures
```

### 5.2. Quy trình Xây dựng Dữ liệu & Chỉ mục (Ingest $\to$ Evidence $\to$ Index)

```powershell
# Bước 1: Ingest video và dựng Presentation-Order Timeline Ledger
videntra ingest `
  --manifest manifests/qvh24.media.json `
  --config configs/ingestion/m1_default.json `
  --output work/ingest `
  --asset-roots configs/local/asset_roots.json `
  --resume

# Bước 2: Trích xuất bằng chứng đa phương thức (OCR & ASR)
videntra evidence build `
  --manifest manifests/qvh24.media.json `
  --ingest-dir work/ingest `
  --output work/evidence `
  --modalities ocr,asr `
  --asset-roots configs/local/asset_roots.json

# Bước 3: Lập chỉ mục Visual Vector Index
videntra index build `
  --manifest manifests/qvh24.media.json `
  --ingest-dir work/ingest `
  --output work/index `
  --model-path models/siglip2 `
  --device cpu `
  --asset-roots configs/local/asset_roots.json
```

> **Offload Embedding & Extraction nặng lên Kaggle GPU:** Với tập dữ liệu hàng nghìn video, sử dụng quy trình đóng gói & đồng bộ có kiểm tra SHA-256 + timeline ledger:
> 1. Đóng gói keyframe input: `python scripts/package_kaggle_inputs.py`
> 2. Chạy worker GPU trên Kaggle: `scripts/kaggle_multimodal_offline_worker.py` (đã pin model revisions & PyAV)
> 3. Kiểm tra provenance & nạp kết quả vào snapshot mới (không ghi đè lịch sử): `python scripts/ingest_kaggle_offline_results.py`

### 5.3. Tìm kiếm Khoảnh khắc qua CLI (`videntra search`)

**Chế độ Production (`--backend unified` — khuyến nghị):** Sử dụng cùng pipeline 7 chặng của benchmark và API, hỗ trợ bật kiểm chứng quan hệ `--verify-relations`:

```powershell
videntra search `
  --backend unified `
  --db-path data/processed/metadata.db `
  --faiss-path data/processed/faiss_index.bin `
  --model-path models/siglip2 `
  --snapshot-id local-prod-v1 `
  --mode accurate `
  --ranking-mode diverse_video `
  --verify-relations `
  --query "a person riding a bicycle near a shop sign" `
  --top-k 10 `
  --output work/predictions.jsonl
```

**Chế độ Visual Cơ bản (`--backend visual`):** Dùng cho index cục bộ tạo trực tiếp bởi `videntra index build`:

```powershell
videntra search `
  --index-dir work/index `
  --query "màn hình kiểm tra màu sắc chuẩn" `
  --top-k 10 `
  --output work/predictions_visual.jsonl
```

### 5.4. Khởi chạy Server REST API & Web UI Inspector (`videntra serve`)

```powershell
videntra serve `
  --backend unified `
  --host 127.0.0.1 `
  --port 8000 `
  --db-path data/processed/metadata.db `
  --faiss-path data/processed/faiss_index.bin `
  --model-path models/siglip2 `
  --ingest-dir work/ingest `
  --asset-roots configs/local/asset_roots.json `
  --snapshot-id local-prod-v1
```

Sau khi khởi động, mở trình duyệt tại `http://127.0.0.1:8000/` để sử dụng **Web UI Inspector** (tài liệu OpenAPI tương tác có tại `http://127.0.0.1:8000/docs`).

#### Ví dụ gọi API `POST /v1/search`:
```powershell
curl -X POST "http://127.0.0.1:8000/v1/search" `
  -H "Content-Type: application/json" `
  -d '{"query": "person wearing red jacket", "top_k": 5, "mode": "accurate", "ranking_mode": "diverse_video", "verify_relations": true}'
```

#### Phím tắt trên trình phát Web UI Inspector:
- `Space`: Phát / Tạm dừng (Play / Pause)
- `R`: Phát lại đúng khoảng thời gian khoảnh khắc tìm được
- `,` / `.`: Lùi 1 khung hình / Tiến 1 khung hình (đối soát `X-Frame-Index`, `X-PTS`, `X-Timestamp-Us`)
- `Esc`: Đóng trình phát

### 5.5. Đánh giá Offline (`videntra evaluate`)
Chỉ chạy ở pha Evaluator sau khi toàn bộ predictions đã được xuất cố định:

```powershell
videntra evaluate `
  --predictions work/predictions.jsonl `
  --ground-truth validation/qvhighlights/dev_labels.jsonl `
  --output work/evaluation_report.json
```

---

## 6. Kiểm thử Tự động & Công cụ Kiểm toán

### 6.1. Chạy toàn bộ bộ kiểm thử (`pytest`)
```powershell
python -B -m pytest -q -p no:cacheprovider
```
Kết quả kiểm thử toàn diện gần nhất sau khi hoàn thiện codebase (Windows CPU): **399 passed (0 failed)** trong ~62.1s.

Hoặc kiểm tra riêng từng lớp thành phần:
```powershell
# Kiểm tra Ground Truth Isolation (E12)
pytest -q tests/evaluation/test_isolation.py

# Kiểm tra đồng nhất giữa API, CLI Unified & Relation Verifier
pytest -q tests/api/test_unified_parity.py

# Kiểm tra cấu hình đường dẫn và bảo vệ denied_roots
pytest -q tests/registry/test_asset_roots.py
```

### 6.2. Các Script Kiểm toán Read-Only
- `python scripts/audit_model_index.py --db data/processed/metadata.db --index data/processed/faiss_index.bin --model-dir models/siglip2` — Đối soát cấu trúc DB, số lượng vector FAISS và hash checkpoint mô hình.
- `python scripts/audit_decoded_timeline.py` — Đối chiếu PTS giải mã thực tế từng frame với metadata lưu trữ.
- `python scripts/audit_kaggle_provenance.py` — Kiểm toán SHA-256 đầu ra Kaggle worker và `keyframe_manifest.json`.
- `python scripts/run_validation_benchmark.py` — Chạy bộ benchmark đánh giá đa tập (yêu cầu dữ liệu chỉ mục và split đánh giá đầy đủ trong máy cục bộ).

---

## 7. Hiệu năng & Giới hạn Kỹ thuật đã Ghi nhận

Số liệu tham khảo từ báo cáo kiểm toán nội bộ [`benchmark/benchmark_report.md`](benchmark/benchmark_report.md) (đo trên kho 1,324 video / 11,179 keyframes 864d, phần cứng **Intel Core i5-7300U CPU, 32GB RAM, Local SSD**):

- **Tập `dev_150` (150 queries):**
  - Mốc kiểm toán gốc tại [`benchmark/benchmark_report.md`](benchmark/benchmark_report.md): Video Hit@1 **61.33%** (92/150), MRR **0.6540**, Video nDCG@10 **0.6763**, Moment Hit@1 (tIoU $\ge 0.5$) **72.73%** (16/22), **0 ca** Pure-ASR False Positive Top-1.
  - Sau nâng cấp Phase 5 (đo read-only trực tiếp trên cùng tập `dev_150` với `A5`): Video Hit@1 đạt **62.00%** (93/150), Hit@5 **71.33%**, Hit@10 **74.67%**, MRR **0.6584**, Video nDCG@10 **0.6796**, Moment Hit@1 (tIoU $\ge 0.5$) **72.73%**, Moment nDCG@5 **0.5303**.
- **Độ trễ truy vấn (CPU-only, tham chiếu [`benchmark/benchmark_report.md`](benchmark/benchmark_report.md)):** Non-neural retrieval trung vị `p50 = 26.46 ms` (`p95 = 46.92 ms`); Cold query (bao gồm SigLIP 2 Text Encoding trên CPU) `p50 = 250.93 ms` (`p95 = 317.82 ms`).
- **Bộ nhớ tiêu thụ:** RSS **420.0 MB**, Peak Working Set **580.0 MB**.
- **Tập mở rộng (mốc kiểm toán trước Phase 5):** `dev_full_859` đạt Video Hit@1 **41.68%**, MRR **0.5012**; `heldout_test_201` đạt Video Hit@1 **38.81%**, MRR **0.4733** (chi tiết phân tích sai số `F1..F5` xem tại [`benchmark/benchmark_report.md`](benchmark/benchmark_report.md)).

**Giới hạn trung thực:**
1. Chưa công bố số liệu benchmark chính thức trên phần cứng đích NVIDIA RTX 3060 6GB VRAM vì quá trình đo vừa qua thực hiện trên máy CPU thử nghiệm và GPU Kaggle Tesla T4 cho tác vụ offline extraction.
2. Các bộ dữ liệu bên thứ ba nếu chưa có biên nhận cấp phép suy luận thương mại được giữ trạng thái `rights_status = "unknown"` và cách ly trong `quarantine`/`denied_roots`.
3. Chức năng `verify_relations` sử dụng OCR, ASR đã hiệu chuẩn và `FrameObjects` 2D bounding boxes nằm trong khoảng thời gian moment (`left_of`, `right_of`, `above`, `below`, `inside`, `near` cùng frame, hoặc `action_observation` tường minh); quan hệ 3D (`behind`/`in_front_of`), quan hệ khác frame, hoặc hành động chỉ có box tĩnh vẫn giữ trạng thái `UNKNOWN` theo logic tam trị.
