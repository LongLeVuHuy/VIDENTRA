# VIDENTRA — Milestone 1 (M1) Implementation Contracts

**Phiên bản:** 1.0  
**Ngày hoàn tất:** 05/11/2026  
**Phạm vi:** Ingestion, canonical timeline, adaptive sampling & independent OCR probes theo blueprint đã duyệt.

---

## 1. Modules và Interfaces

### 1.1 Ingestion & Probe (`videntra.ingestion.probe`)
- `probe_media(asset: Path | str | dict, stream_selection: StreamSelectionConfig | dict | None = None, tools: dict | None = None, timeout_seconds: float = 30.0) -> ProbeResult`
  - Thực thi `ffprobe` bằng argv list trực tiếp (không shell concatenation), timeout và structured errors.
  - Phân tích thông tin container, video streams, audio streams, rational `time_base`.
  - Hỗ trợ stream selection tường minh (`video_stream_index`, `audio_stream_index`, `allow_missing_audio`).
  - Phân loại lỗi: `ProbeError`, `ProbeTimeoutError`, `StreamSelectionError`, `HashMismatchError`, `NoVideoStreamError`.

### 1.2 Sequential Decoding (`videntra.ingestion.decode`)
- `decode_frames(asset: Path | str | dict, probe_result: ProbeResult | None = None, recipe: DecodeConfig | dict | None = None, include_pixels: bool = True) -> Iterator[DecodedFrame]`
  - Giữ ordinal zero-based tăng dần tuần tự theo presentation order từ decoder, **không bao giờ suy ra từ nominal FPS**.
  - Authoritative timestamps từ `pts` và rational `time_base`. Microseconds tính bằng half-to-even rounding (`round()`).
  - First valid presentation time là `origin` chung cho video và audio.
  - Xử lý lặp PTS (`pts_occurrence` tăng dần) và gián đoạn thời gian / backward PTS (tự động phân tách `epoch_id`).
  - `DecodedFrame` là transient record; pixels dạng numpy ndarray (RGB uint8) được giải phóng sau xử lý, không serialize vào metadata registry.

### 1.3 Timeline Ledger & Maps (`videntra.timeline.ledger`, `videntra.timeline.maps`)
- `write_timeline_ledger(destination_path, frames, video_id, timeline_revision, row_group_size=1000, source_map=None, store_pixel_hashes=False) -> LedgerSummary`
  - Ghi Parquet phân trang theo row groups có giới hạn bộ nhớ qua `pyarrow.parquet.ParquetWriter`.
  - Schema tuân thủ normative contract `FrameRecord`.
- `read_frame_record(parquet_path, ordinal: int) -> dict`
- `get_ledger_endpoints(parquet_path, start_us: int, end_us: int) -> tuple[int, int]`
  - Xác định start ordinal (frame đầu có PTS $\ge$ start) và end ordinal (frame cuối có PTS $<$ end) theo khoảng nửa mở $[start, end)$.
- `exact_seek_frame(media_path, parquet_path, target_ordinal, video_stream_index=0) -> DecodedFrame`
  - Tìm keyframe gần nhất trước `target_ordinal` trong cùng epoch, seek container đến keyframe PTS và decode tiến tới frame đích.
  - Đạt bit-level parity tuyệt đối với sequential decode.
- `RationalClipMap`: ánh xạ rational PTS $\leftrightarrow$ clip microseconds.
- `PiecewiseSourceMap`: ánh xạ clip time $\rightarrow$ source time (hỗ trợ `constant_offset` và `piecewise` segments; trả `None` khi không xác định).
- `AudioMap` & `build_audio_map`: ánh xạ audio packets/chunks với shared origin của video, lưu trữ tại `audio_map.jsonl`.

### 1.4 Hybrid Sampling & Independent OCR Probes (`videntra.sampling`)
- `sample_frames_hybrid(frames, config, video_id, timeline_revision, sampler_revision) -> SampleManifest`
  - Phân tích ảnh low-res 256px max dim.
  - Luma difference $d_i$, histogram intersection distance $c_i$, rolling 2s median/MAD change score $r_i$.
  - Mandatory anchors: `first`, `last`, `max_gap` (mặc định $\le$ 2s), `cut_before`, `cut_after`. **Tuyệt đối không bị xóa bởi dedup hoặc quota.**
  - Change triggers & burst grid: $[t - 0.5s, t + 1.0s]$ tại 4 fps, snap chính xác vào frame thực tế đã decode.
  - pHash 64-bit DCT: Hamming distance $> 4$ mới chấp nhận soft sample.
  - Soft rolling quota: $\le 1$ fps trên cửa sổ 60s. Ghi nhận overflow nếu mandatory anchors vượt ngân sách.
- `select_ocr_probes(frames, video_id, probe_fps=1.0, shot_cut_ordinals=None) -> OCRProbeManifest`
  - Luồng probe OCR độc lập hoàn toàn với visual selection (mặc định 1 fps + shot cuts).
- `generate_uniform_baseline`, `compute_structural_metrics`: so sánh cấu trúc với uniform 1 fps và uniform budget-matched.

### 1.5 Resumable Ingestion Job Service (`videntra.ingestion.jobs`)
- Cache key: $\text{SHA-256}(asset\_hash : stage\_id : config\_hash)$.
- Trạng thái: `pending` $\rightarrow$ `running` $\rightarrow$ `validated` $\rightarrow$ `committed`.
- Staging tại `_attempts/`, verify toàn bộ Parquet/JSONL trước khi atomic move vào thư mục chính thức.
- Windows file handle safety: giải phóng stream/container trước thao tác rename/move.
- Idempotency: re-run kiểm tra hash cache, tái sử dụng artifact sẵn có, không sinh trùng vector/sample ID.

---

## 2. Đường dẫn và Artifacts đầu ra

Dưới thư mục chỉ định `--output <dir>`:
```text
<output_dir>/
  run_manifest.json                          # Tổng kết run, hashes, thống kê
  metrics.json                               # Chỉ số rút gọn
  timeline/
    <safe_video_id>/
      <timeline_revision>/
        frames.parquet                       # Parquet ledger chi tiết từng frame
        audio_map.jsonl                      # Ánh xạ audio chunks
  samples/
    <safe_video_id>/
      <sampler_revision>/
        samples.parquet                      # Danh sách sample thích ứng
        ocr_probes.parquet                   # Danh sách probe OCR độc lập
```

---

## 3. CLI Interfaces

```powershell
# 1. Ingest batch media
python -m videntra ingest --manifest <manifest_path> --config configs/ingestion/m1_default.json --output work/m1/<run_id> [--resume]

# 2. Inspect timeline frame
python -m videntra timeline inspect --video-id <video_id> --revision <revision> --frame-index <ordinal> [--ledger <path>]

# 3. Structural sampling comparison
python -m videntra sampling compare --config configs/experiments/e02.json --output work/m1/<comparison_run_id> [--manifest <manifest_path>]
```

---

## 4. Trạng thái Quyền dữ liệu (Rights Gate)

- **Synthetic Fixtures:** 100% allowed (nguồn gốc fixture harness nội bộ, không phụ thuộc bên thứ ba).
- **QVHighlights 24 real media clips:** Hiện tại giữ nguyên `rights_status = unknown`. Theo quy định blueprint, không tự động đổi sang `allowed` và không dùng cờ bypass. QVH pilot ghi nhận trạng thái **BLOCKED BY RIGHTS REVIEW** cho đến khi có xác nhận giấy phép nghiên cứu chính thức.
