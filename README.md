<div align="center">

# VIDENTRA

**Hệ Thống Truy Xuất & Định Vị Khoảnh Khắc Video Đa Phương Thức Chống Rò Rỉ Dữ Liệu**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![MSR-VTT 1K-A Hit@1](https://img.shields.io/badge/MSR--VTT%201K--A%20Hit%401-57.10%25-success)](benchmark/msrvtt_1ka_benchmark_report.md)
[![MRR@10](https://img.shields.io/badge/MSR--VTT%201K--A%20MRR%4010-0.6670-blue)](benchmark/msrvtt_1ka_benchmark_results.json)
[![Protocol](https://img.shields.io/badge/Protocol-3--Split%20Zero%20Leakage-informational)](AGENTS.md)

</div>

---

**VIDENTRA** là hệ thống truy xuất video theo truy vấn văn bản (*Text-to-Video Retrieval*) và định vị khoảnh khắc (*Moment Localization*), được thiết kế trên hợp đồng thời gian chuẩn (`0-based presentation-order` + `PTS × time_base` gốc, không phụ thuộc FPS giả định), kho dữ liệu tự mô tả theo từng dataset và quy trình đánh giá 3 phân vùng chống rò rỉ dữ liệu (**Zero Leakage Rule**).

Trên tập kiểm chuẩn niêm phong chuẩn quốc tế **MSR-VTT 1K-A** (`1.000` video / `1.000` truy vấn đánh giá một lần duy nhất sau khi khóa cấu hình), **VIDENTRA v2 (`R9d_videntra_v2_champion`)** đạt:

- **Hit@1 (R@1):** **`57.10%`** (`571 / 1.000`) — vượt baseline `SigLIP2-Base` (`42.30%`) thêm **`+14.80 pp`**, vượt `VideoPrism-LvT-B` (`48.70%`) thêm **`+8.40 pp`** (`95% Bootstrap CI [+6.02, +10.80] pp`, `+119` câu cứu đúng / `-35` câu tụt hạng, ròng `+84` câu), và vượt VIDENTRA v1 `R8` (`55.60%`) thêm **`+1.50 pp`**.
- **R@5:** **`79.90%`** | **R@10:** **`87.40%`** | **Stage-1 R@20:** **`91.80%`** | **MRR@10:** **`0.6670`**.

---

## Mục Lục

1. [Tính Mới & Khác Biệt Kiến Trúc của VIDENTRA so với Các Dự Án Khác](#tính-mới--khác-biệt-kiến-trúc-của-videntra-so-với-các-dự-án-khác)
2. [Kiến Trúc & Pipeline VIDENTRA v2](#kiến-trúc--pipeline-videntra-v2)
3. [Cấu Trúc Thư Mục Tự Mô Tả & Quy Tắc 3 Phân Vùng Zero Leakage](#cấu-trúc-thư-mục-tự-mô-tả--quy-tắc-3-phân-vùng-zero-leakage)
4. [Kết Quả Benchmark Chính Thức](#kết-quả-benchmark-chính-thức)
   - [Benchmark MSR-VTT 1K-A (Tune → Confirm → Sealed Test)](#benchmark-msr-vtt-1k-a-tune--confirm--sealed-test)
   - [Benchmark 5 Chiến Lược Trích Xuất Keyframes](#benchmark-5-chi-ến-lược-trích-xuất-keyframes)
5. [Cài Đặt, Quản Lý Dataset & Khởi Chạy](#cài-đặt-quản-lý-dataset--khởi-chạy)
6. [Tài Liệu Dự Án Liên Quan](#tài-liệu-dự-án-liên-quan)

---

## Tính Mới & Khác Biệt Kiến Trúc của VIDENTRA so với Các Dự Án Khác

Khác với các repo truy xuất video thường dừng ở việc so khớp cosine 1 backbone CLIP đơn lẻ trên vài frame lấy mẫu đều hoặc hiệu chỉnh *transductive* dựa trên toàn bộ tập truy vấn test, VIDENTRA tập trung giải quyết đồng thời 4 bài toán thực tế:

1. **Hợp Đồng Thời Gian VFR Chính Xác (`Presentation-Order PTS Ledger`):**
   - Toàn bộ quá trình giải mã bằng `PyAV` lập chỉ mục frame theo đúng thứ tự hiển thị (`0-based presentation-order`) và tính timestamp trực tiếp từ `PTS × time_base`, tuyệt đối không tính `frame_idx / FPS` vốn sai lệch nghiêm trọng trên video *Variable Frame Rate (VFR)*.
2. **Trích Xuất Keyframe Thích Ứng `hybrid_v2_adaptive` với `63-AC pHash` C++ SIMD:**
   - Thay vì lấy mẫu đều (*uniform sampling*) làm bỏ lỡ các điểm cắt cảnh (`Cut Recall` chỉ `32.4%`) hoặc giữ frame trùng lặp (`35.1%–39.3%` redundancy), `hybrid_v2_adaptive` dùng `63-AC pHash`, bộ lọc nét (*Composite Sharpness Lookahead*) và bộ khử cắt giả (chuyển động mờ/chớp sáng), đạt **`100.0%` Cut Recall**, **`1.67%` Redundancy**, **`0` vi phạm max-gap**, giảm số keyframe cần lưu mà vẫn bảo toàn độ phủ ngữ nghĩa.
3. **Khử Hubness Qui Nạp Không Rò Rỉ bằng Train-9K Query Bank (`21.219` câu sạch):**
   - Các phương pháp chuẩn hóa hubness kiểu *transductive* (chỉ dùng tốt khi biết trước toàn bộ query của tập test) không phù hợp cho truy vấn đơn lẻ khi triển khai thực tế (*streaming inductive search*). VIDENTRA xây dựng **Query-Bank Inductive Calibration** (`QB-DIS` Dynamic Inverted Softmax + `QB-Z-Norm` + `QB-CSLS`) dựa hoàn toàn trên **`21.219` captions sạch thuộc `train_9k_clean`** (đã loại toàn bộ `source_group` trùng lặp với Valid/Test), cho phép hiệu chỉnh từng câu truy vấn đến riêng lẻ theo thời gian thực (`Zero Test Leakage`).
4. **Hợp Nhất 6 Backbone Đa Không-Thời Gian + `MultiBackboneCrossModalAdapter` + VLM Có Cổng Biên:**
   - Kết hợp sức mạnh bổ trợ giữa biểu diễn ảnh tĩnh phân giải cao (`PE-Core-bigG-14-448`, `DFN5B-CLIP-ViT-H-14-378`, `PE-Core-L-14-336`, `SigLIP2-SO400M-384`, `SigLIP2-Base-384`), biểu diễn không-thời gian cấp clip/sub-clip (`VideoPrism-LvT-B`), cùng bằng chứng ngữ nghĩa rời rạc (`ASR` Faster-Whisper, `OCR` RapidOCR, `YOLOv8n + Kalman` tracks, `Florence-2` captions).
   - Ở tầng V2 (`R9d`), những truy vấn có biên phân tách Stage-1 còn mơ hồ (`stage1_margin_z < 1.2`) được đưa qua **Stage-2 VLM Storyboard Cross-Encoder (`Qwen2.5-VL-7B-Instruct` 4-bit NF4)** đọc lưới **2×4 (8 khung thời gian theo trục thời gian)** và khử thiên vị video (`debias_alpha=0.3`, không đụng nhãn thật).
5. **Kiến Trúc Kho Dữ Liệu Module Hóa & Quy Trình Đánh Giá 3 Phân Vùng Nghiêm Ngặt:**
   - Mỗi dataset là một gói độc lập tại `data/datasets/<id>/` với CLI `videntra dataset list|add|remove|verify`, cho phép thêm hoặc xóa dataset (kèm cắt atomic khỏi SQLite & FAISS) dễ dàng mà không làm hỏng hệ thống.

---

## Kiến Trúc & Pipeline VIDENTRA v2

```text
  Video Corpus (.mp4)
       │
       ▼  [1. PyAV Decode + PTS Ledger]
  Presentation-Order Frames (0-based, exact PTS * time_base)
       │
       ▼  [2. HybridAdaptiveSampler (hybrid_v2_adaptive, 63-AC pHash)]
  Adaptive Keyframes (.jpg) + Shot Intervals
       │
       ├──► [3a. Visual / Spatio-Temporal Backbones (Kaggle Dual-T4 + Local FAISS)]
       │     ├─ PE-Core-bigG-14-448 (1280d, 12 frames + X-Pool)
       │     ├─ DFN5B-CLIP-ViT-H-14-378 (1024d, 12 frames)
       │     ├─ PE-Core-L-14-336 (1024d, 12 frames)
       │     ├─ SigLIP2-SO400M-384 (1152d, 12 frames)
       │     ├─ VideoPrism-LvT-B (768d whole + multi-window sub-clips)
       │     └─ SigLIP2-Base-384 (768d keyframe dense index)
       │
       └──► [3b. Explicit Multimodal Evidence]
             ├─ RapidOCR -> Text Evidence
             ├─ Faster-Whisper small + Silero VAD -> ASR Segments (BM25 + E5 384d)
             ├─ YOLOv8n + Kalman/SORT -> Object Tracks
             └─ Florence-2-base -> Visual Captions
       │
       ▼  [4. Query Processing & Inductive Calibration (21.2K Train-9K Query Bank)]
  Multi-Prompt Ensembling + QB-DIS / QB-Z-Norm / QB-CSLS + Boltzmann LogSumExp
       │
       ▼  [5. Stage-1 Multi-Backbone Fusion & Cross-Modal Adapter (R9c)]
  Top-20 Candidates + Stage-1 Uncertainty Margin (m1_z = s[0] - s[1])
       │
       ├──► [nếu m1_z >= 1.2] ──► Giữ nguyên xếp hạng Stage-1 (bảo toàn truy vấn chắc chắn)
       └──► [nếu m1_z <  1.2] ──► [6. Stage-2 VLM Storyboard Cross-Encoder]
                                     Qwen2.5-VL-7B-Instruct (NF4, 2×4 chronological grid, Top-5)
                                     + Video Prior Debiasing (alpha=0.3) & margin_raw fusion
                                           │
                                           ▼
                                  [7. Ranked Results (`R9d_videntra_v2_champion`), Provenance & Web Inspector UI]
```

*Ghi chú kỹ thuật về Stage-2 VLM:* Khi chạy `Qwen/Qwen2.5-VL-7B-Instruct` (`tie_word_embeddings=False`) trên Kaggle, hàm `_tie_vlm_weights` đã tạo phép chiếu đầu ra lên không gian vector `input_embeddings[Yes] - input_embeddings[No]`, làm độ phù hợp giữa truy vấn và video tương quan nghịch tuyến ổn định với logit gốc (được kiểm chứng độc lập cả hai chiều trên `Valid-Tune`: Hit@1 thuần tăng `+1.00 pp`, `w_vlm = -1.0` chọn hoàn toàn trên `Valid-Tune` và cải thiện đồng nhất trên `Sealed Test 1K-A` thêm `+1.10 pp` từ `56.00%` lên `57.10%`).

---

## Cấu Trúc Thư Mục Tự Mô Tả & Quy Tắc 3 Phân Vùng Zero Leakage

Toàn bộ dữ liệu và mã nguồn được tổ chức theo quy tắc đặt tên tự mô tả (**self-documenting**) ghi trong [`AGENTS.md`](AGENTS.md):

```text
VIDENTRA/
├── configs/local/                 # Cấu hình asset_roots cục bộ (chỉ track file mẫu .example.json)
├── data/
│   ├── datasets/                  # Kho dataset độc lập (tự động đồng bộ qua `videntra dataset`)
│   │   ├── msrvtt/                #   MSR-VTT 1K-A gallery (1.000 videos, keyframes, SQLite, metadata)
│   │   ├── msrvtt_val_tune/       #   MSR-VTT Valid-Tuning (2.000 videos khép kín: catalog, embeddings, evidence)
│   │   └── msrvtt_val_confirm/    #   MSR-VTT Valid-Confirm (2.000 videos khép kín: catalog, embeddings, evidence)
│   ├── processed/                 # Dữ liệu phục vụ runtime theo phân vùng rõ ràng
│   │   ├── catalog/               #   corpus_metadata.db (SQLite WAL)
│   │   ├── visual_keyframes/      #   siglip2_keyframes_768d.faiss & siglip2_yolo_augmented_864d.npy
│   │   ├── visual_clips/          #   VideoPrism 768d + PE-Core + SigLIP2-SO400M + DFN5B vectors
│   │   └── text_evidence/         #   e5_ocr_asr_384d.faiss & e5_ocr_asr_manifest.json
│   └── cache/
│       ├── query_embeddings/      #   Cache embedding truy vấn đã tính
│       ├── query_plans/           #   Cache kế hoạch phân rã câu truy vấn
│       └── calibration/           #   siglip2_keyframe_hubness_radii.npy
├── validation/msrvtt/             # Chỉ chứa nhãn & truy vấn validation (KHÔNG đụng tập test)
│   ├── valid_tuning/              #   2.000 videos (tune_g1 + tune_g2) — dùng để sweep/tune siêu tham số
│   └── valid_confirm/             #   2.000 videos (confirm_g1 + confirm_g2) — dùng để kiểm chứng độc lập & khóa Champion
├── test_queries/msrvtt/
│   └── test_1ka_benchmark/        # 1.000 videos MSR-VTT 1K-A niêm phong — CHỈ chấm benchmark cuối cùng 1 lần
├── src/videntra/                  # Mã nguồn lõi thư viện (ingestion, sampling, indexing, retrieval, ranking, registry, api)
├── scripts/                       # Công cụ huấn luyện, Kaggle extraction, V1/V2 evaluation matrix
├── benchmark/                     # Báo cáo và file JSON benchmark chính thức
└── tests/                         # Bộ unit & integration tests (cô lập hoàn toàn khỏi Ground Truth)
```

### Nguyên Tắc Chống Rò Rỉ 3 Phân Vùng (Zero Leakage Rule)
- **Huấn luyện (`train_9k_clean`, `4.585` videos):** Đã loại trừ hoàn toàn các clip trùng `source_group` với cả 3 tập eval; chỉ dùng để huấn luyện adapter/ranker và xây dựng `Train-9K Query Bank` (`21.219` captions sạch).
- **Tinh chỉnh (`Valid-Tuning`, `2.000` videos / `2.000` queries):** Phân vùng duy nhất được phép quét/tinh chỉnh trọng số backbone, siêu tham số `QB-DIS`/`QB-Z-Norm`, `margin_gate` và `w_vlm`.
- **Kiểm chứng & Khóa Champion (`Valid-Confirm`, `2.000` videos / `2.000` queries):** Đánh giá mù 1 lần cho ứng viên tốt nhất từ Tuning để quyết định khóa Champion (`51.45%` vs V1 `50.20%`).
- **Kiểm chuẩn niêm phong (`Sealed Test 1K-A`, `1.000` videos / `1.000` queries):** Tuyệt đối **không** dùng để huấn luyện, chọn lọc hay tune siêu tham số; chỉ mở niêm phong chấm đúng 1 lần sau khi Champion V2 đã khóa.

---

## Kết Quả Benchmark Chính Thức

### Benchmark MSR-VTT 1K-A (Tune → Confirm → Sealed Test)

Đánh giá bằng bộ nguồn chính thức tại [benchmark/msrvtt_1ka_benchmark_report.md](benchmark/msrvtt_1ka_benchmark_report.md) và [benchmark/msrvtt_1ka_benchmark_results.json](benchmark/msrvtt_1ka_benchmark_results.json) (cấu hình khóa tại `2026-11-23T07:54:01Z`, `seed=20261123`, `885` source groups trên `test_1ka`):

| Config ID | Mô Tả Kiến Trúc | Valid-Tune Hit@1 (`2.000` q) | Valid-Confirm Hit@1 (`2.000` q) | Test 1K-A **Hit@1** (`1.000` q) | Test 1K-A **R@5** | Test 1K-A **R@10** | Test 1K-A **MRR@10** | Δ Hit@1 vs B1 (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_siglip2_standalone` | Baseline `SigLIP2-Base-384` | — | 36.55% | 42.30% | 65.10% | 74.10% | 0.5207 | -6.40 pp `[-9.42, -3.44]` |
| `B1_videoprism_standalone` | Baseline `VideoPrism-LvT-B` (whole + clips) | — | 43.95% | 48.70% | 75.30% | 82.50% | 0.5972 | +0.00 pp (Baseline) |
| `R9_dfn5b_h_standalone` | Đơn lẻ `DFN5B-CLIP-ViT-H-14-378` | 37.95% | 37.70% | 41.60% | 66.70% | 75.50% | 0.5244 | -7.10 pp `[-10.00, -4.25]` |
| `R9_pe_bigg_standalone` | Đơn lẻ `PE-Core-bigG-14-448` (X-Pool) | 47.15% | 48.45% | 53.60% | 77.40% | 84.60% | 0.6386 | +4.90 pp `[+2.04, +7.73]` |
| `R8_videntra_v1_champion` | VIDENTRA v1 Champion (3 backbone + QB-CSLS + MP) | 49.25% | 50.20% | 55.60% | 79.00% | 86.80% | 0.6549 | +6.90 pp `[+4.68, +9.13]` |
| `R9a_6backbone_raw` | Hợp nhất thô 6 backbone | 49.35% | 49.75% | 54.30% | 79.10% | 86.50% | 0.6467 | +5.60 pp `[+3.19, +7.97]` |
| `R9b_6backbone_qb_dis_znorm_mp` | 6 backbone + 21.2K `QB-DIS`/`QB-Z-Norm` + Multi-Prompt | 49.85% | 51.15% | 55.90% | 79.70% | 87.40% | 0.6593 | +7.20 pp `[+4.82, +9.58]` |
| `R9c_6backbone_adapter_multimodal` | R9b + Adapter + Multimodal (`ASR` + `OCR`) | 49.85% | **51.55%** | 56.00% | **79.90%** | **87.40%** | 0.6603 | +7.30 pp `[+4.92, +9.71]` |
| **`R9d_videntra_v2_champion`** | **R9c + Stage-2 Margin-Gated `Qwen2.5-VL-7B` Reranker** | **50.95%** | **51.45%** | **57.10%** | **79.90%** | **87.40%** | **0.6670** | **+8.40 pp `[+6.02, +10.80]`** |

*Chi tiết từng bước đóng góp trên Sealed `test_1ka`:*
- Chuẩn hóa hubness qui nạp `21.2K` + Multi-Prompt (`R9a → R9b`): **`+1.60 pp`** (`54.30% → 55.90%`).
- Kết hợp tín hiệu đa phương thức `ASR` + `OCR` (`R9b → R9c`): **`+0.10 pp`** (`55.90% → 56.00%`, Stage-1 `R@20 = 91.80%`).
- Stage-2 Margin-Gated `Qwen2.5-VL-7B-Instruct` trên `328/1.000` truy vấn dưới ngưỡng `margin_gate=1.2` (`R9c → R9d`): **`+1.10 pp`** (`56.00% → 57.10%`, cứu ròng so với `B1`: `+119 / -35`, tổng ròng `+84` câu).

---

### Benchmark 5 Chiến Lược Trích Xuất Keyframes

Đánh giá đa tầng so sánh chất lượng đại diện hình ảnh (**Tier 2 Corpus**) và hiệu quả truy xuất hạ nguồn bằng **SigLIP2** (**Tier 3**) trên các chiến lược trích xuất keyframes (`composite_score` xét đồng thời độ phủ cắt cảnh, độ đại diện, tính đa dạng chống trùng lặp và zero max-gap violation):

| Chiến Lược (`Strategy`) | T3 Hit@1 | T3 MRR | T3 Peak Cosine | T2 Rep Score | T2 Cut Recall | T2 Blank Rate | T2 Redundancy | Vi Phạm Max-Gap | Composite Score | Đủ ĐK Mặc Định |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hybrid_v2_adaptive` (Mặc định)** | 66.7% | 0.7932 | 0.1329 | 0.9266 | **100.0%** | 0.83% | **1.67%** | **0** | 0.62874 | **Có** |
| `sharpness_lookahead` | 66.7% | **0.7944** | **0.1334** | **0.9341** | **100.0%** | 0.83% | **1.53%** | **0** | **0.63053** | Có |
| `shot_medoid_submodular` | 66.7% | 0.7913 | 0.1301 | 0.9129 | **100.0%** | 0.91% | **0.00%** | **0** | 0.62509 | Có |
| `hybrid_v1_legacy` | 61.1% | 0.7648 | 0.1334 | 0.9280 | 100.0% | 0.98% | 39.28% | 0 | 0.61622 | Có |
| `uniform_matched` | 77.8% | 0.8372 | 0.1318 | 0.9289 | 32.4% | 0.18% | 35.12% | 0 | 0.58135 | Không |
| `uniform_matched_budget` | 66.7% | 0.7927 | 0.1313 | 0.9313 | 44.0% | 0.02% | 33.17% | 0 | 0.57322 | Không |

- **Lý do chọn `hybrid_v2_adaptive` làm mặc định hệ thống:** Giảm dư thừa khung hình từ **`39.28%` (`hybrid_v1_legacy`) xuống chỉ còn `1.67%`**, giảm số keyframe trung bình từ `73.40` xuống `54.53`, đảm bảo `100.0%` điểm cắt cảnh và `0` vi phạm max-gap (các chiến lược *uniform* bị loại khỏi mặc định vì `Cut Recall` chỉ `32.4%–44.0%` làm mất sự kiện chuyển cảnh ngắn).

---

## Cài Đặt, Quản Lý Dataset & Khởi Chạy

### 1. Yêu Cầu Môi Trường
- **OS & Python:** Windows / Linux, **Python 3.12** (`>=3.12,<3.13`).
- **Phần cứng mục tiêu:** NVIDIA RTX 3060 6GB VRAM + 32GB RAM (các tác vụ encoding backbone lớn `PE-Core-bigG`, `DFN5B-CLIP-H`, `Qwen2.5-VL-7B` có script điều phối sẵn cho **Kaggle 2 × Tesla T4 GPU**).

```powershell
# Khởi tạo virtual environment (Python 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1. Cài PyTorch CUDA 12.6 (nếu dùng GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Cài toàn bộ dependencies & VIDENTRA ở chế độ editable
pip install -r requirements.txt
pip install --no-build-isolation --no-deps -e .
```

### 2. Kiểm Tra Sức Khỏe Hệ Thống & Quản Lý Dataset
```powershell
# Kiểm tra toàn diện biến môi trường, đường dẫn processed/cache và GPU/CPU
.\.venv\Scripts\python.exe -m videntra doctor --strict

# Xem các dataset tự mô tả đã đăng ký (msrvtt, msrvtt_val_tune, msrvtt_val_confirm)
.\.venv\Scripts\python.exe -m videntra dataset list

# Kiểm tra tính toàn vẹn media, keyframes, SQLite catalog và embeddings
.\.venv\Scripts\python.exe -m videntra dataset verify

# Thêm dataset mới hoặc gỡ bỏ an toàn một dataset (dry-run mặc định)
.\.venv\Scripts\python.exe -m videntra dataset add --id <dataset_id> ...
.\.venv\Scripts\python.exe -m videntra dataset remove --id <dataset_id> --yes --prune-processed
```

### 3. Chạy Giao Diện Web Inspector & Kiểm Thử Tự Động
```powershell
# Khởi động REST API & Web Inspector UI
.\.venv\Scripts\python.exe -m videntra serve

# Chạy toàn bộ bộ kiểm thử tự động
.\.venv\Scripts\python.exe -B -m pytest -q
```

### 4. Tái Lập Ma Trận Đánh Giá V2 & Kaggle Extraction (Dành Cho Nghiên Cứu)
```powershell
# Tinh chỉnh & đánh giá Stage-1 (Tune -> Confirm) + xuất pool Stage-2 zero-GT
.\.venv\Scripts\python.exe -B scripts/run_msrvtt_v2_matrix.py stage1-and-export

# Điều phối Stage-2 Qwen2.5-VL Reranker trên Kaggle GPU & khóa Champion V2
.\.venv\Scripts\python.exe -B scripts/run_msrvtt_kaggle_pipeline.py vlm-rerank-shard --target-label val_tune --candidates-file work/msrvtt_protocol_20261122/v2_run/vlm_candidates_tune.jsonl --push
.\.venv\Scripts\python.exe -B scripts/run_msrvtt_v2_matrix.py fuse-vlm-and-lock

# Chấm benchmark đơn lần trên tập niêm phong test_1ka sau khi đã khóa Champion
.\.venv\Scripts\python.exe -B scripts/run_msrvtt_v2_matrix.py eval-sealed-test --vlm-dir work/msrvtt_protocol_20261122/kaggle_out/vlm_test_1ka
```

---

## Tài Liệu Dự Án Liên Quan

- [PROJECT.md](PROJECT.md) — Sơ đồ cấu trúc module `src/videntra/`, `scripts/` và lịch sử cột mốc dự án.
- [AGENTS.md](AGENTS.md) — Quy tắc kỹ thuật, phân vùng Zero Leakage 3 tập dữ liệu và quy ước tổ chức thư mục tự mô tả.
- [docs/PROGRESS.md](docs/PROGRESS.md) — Nhật ký triển khai chi tiết từng bước, kết quả đo lường và xác minh.
- [benchmark/msrvtt_1ka_benchmark_report.md](benchmark/msrvtt_1ka_benchmark_report.md) — Báo cáo chi tiết kết quả đơn lần trên `MSR-VTT 1K-A`.
