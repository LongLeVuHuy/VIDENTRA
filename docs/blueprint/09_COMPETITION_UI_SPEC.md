# 09 — Competition UI và backend contracts

Đặc tả UI/API, **chưa có app triển khai**. Default Top10, Fast Mode, Moment Ranking. Card hiển thị rõ moment start/end, **video ID và original frame index**. Score là rank score, không phần trăm đúng. [Pipeline§13.14](05_VIDENTRA_DETAILED_PIPELINE.md), [timing contracts](06_DATA_SCHEMAS_AND_TIMESTAMPS.md).

## 1. Search screen wireframe

```text
VIDENTRA                                      Snapshot: corpus-2026-v1
Find the moment. Understand the relation.

[ Tìm người áo đỏ đưa túi ...                         ] [Tìm kiếm Enter]
Top-K [10 v]  Mode [Fast v]  Xếp hạng [Moment v]   [Bộ lọc]
Đang tìm... / 10 kết quả • 1.420 ms • 2 nhánh đủ, 1 nhánh thiếu
[Truy vấn đã hiểu ▾]  [Lần tìm trước]  [So sánh 2 kết quả]  [Export ▾]

┌ 1 ──────────────────────┐ ┌ 2 ──────────────────────┐
│ thumbnail               │ │ thumbnail               │
│ video_id                │ │ video_id                │
│ 00:12.000 → 00:17.200    │ │ 01:06.120 → 01:11.480    │
│ frame 300 → 429 (base 0) │ │ frame 1653 → 1788        │
│ keyframe42 / orig360    │ │ keyframe117 / orig1690  │
│ score0.73 • partial      │ │ score0.61 • unknown     │
│ [Visual] [OCR] [ASR]     │ │ [Visual] [OCR missing]  │
│ [Mở] [Copy ID+frame]     │ │ [Mở] [Copy ID+frame]     │
└─────────────────────────┘ └─────────────────────────┘
```

Số liệu wireframe **SYNTHETIC**, không UI screenshot hoặc measured run. Card2 badge “unknown” là relation-detail text; API match_status chuẩn supported/partial/contradicted. Layout adaptive2–5columns, fixed thumbnail aspect ratio với letterbox, không crop dấu/vật cần evidence. Top50 dùng virtual list, thumbnail lazy-load+prefetch bounded. Không load50 video players cùng lúc.

Search controls Top5/10/20/50; Mode Fast/Accurate; rank mode Moment/Diverse Video. Default moment allows repeated video IDs. Filters panel dataset/video/language/modality-status; không mặc định filter-out no-audio clips. Query text giữ nguyên editable; parser summary có roles/then/quoted text và uncertainty “xanh=blue/green” để user tự chỉnh.

## 2. Result inspector wireframe

```text
[Đóng Esc]  video_id  • source_video_id   [Copy]   [So sánh]
┌──────────────────── video source/proxy ─────────────────────┐
│                                                            │
└────────────────────────────────────────────────────────────┘
[Play Space] [Replay R] [Full video] [−1 frame ,] [+1 frame .]
Moment 12.000–17.200s | Current14.400s | original frame360 (zero-based)
Timeline: ──[ give A→B ]──[ speech B ]─────────────────────────
OCR: SALE @14.000,14.800 • ASR: cảm ơn @15.200–17.200
Relations: giver A → receiver B [supported] ; behind [unknown]
[Evidence frames] [Transcript] [Nguồn/độ tin cậy] [Copy current frame]
```

Evidence colors không chỉ là màu; dùng label+icon vì accessibility. Unknown/partial không được dùng badge “verified”. Hover score hiển thị “điểm xếp hạng, chưa hiệu chuẩn xác suất”. Confidence raw model chỉ advanced pane. Thiếu audio do media missing khác utterance không có speech.

## 3. Playback và frame accuracy

Click card→resolve đúng video URI trong snapshot→open one player→seek predicted start→play segment→pause tại end→replay. Browser seek thường snap keyframe và callbacks trễ; không hứa native HTML video frame-perfect boundary stop. Dùng `requestVideoFrameCallback` nếu available, fallback timeupdate; report displayed frame timing via proxy/source map.

Exact inspection là endpoint source-frame decoder: seek codec keyframe trước target, decode forward tới **ledger ordinal**, return frame image+PTS/index. Frame-step dùng ordinal±1, không timestamp±1/FPS. Loading frame không gọi models. Khi source clip chưa có full source ledger, label “frame gốc của clip”; source-video index hiển thị “chưa có mapping”, không tạo theo offset×fps.

Moment start/end half-open; end frame displayed inclusive. Copy payload gồm coordinate_system/frame_index_base/snapshot, không chỉ số mơ hồ. Source offset và clip time toggle giữ rõ nhãn. Audio-only/missing-frame results có nulls, UI không render0 giả.

Full video playback bỏ auto-pause tại moment end theo user toggle; replay trả moment start. Error404/codec unsupported cho proxy/retry action, vẫn giữ result metadata.

## 4. Interaction và speed

Keyboard: Enter search khi focus query; / focus query; arrows select card; Space play khi inspector focus; ,/. step frame; R replay; C copy current video/frame; Esc close/cancel. Không intercept khi đang gõ text hoặc browser shortcut reserved. Tooltip legend và screen-reader labels.

Selected card stable khi Accurate results stream: show “có thứ hạng cập nhật” và apply sau user action hoặc giữ inspector bound moment_id; không đổi video đang xem do reorder. Query mới cancel pending request và ignore stale response bằng client_request_id.

Compare tối đa2players, đồng bộ relative event start optional; không imply video timelines identical. Thumbnail cache by content hash, ETag; transcript/evidence fetched on expand. Cached query result key gồm snapshot/query/model versions/filter/mode/top_k; page navigation không rerun models.

## 5. Search API request/response

```json
{
  "query": "Tìm đoạn có chữ SALE và người nói cảm ơn",
  "top_k": 10,
  "mode": "fast",
  "ranking_mode": "moment",
  "filters": {
    "dataset_ids": [],
    "video_ids": [],
    "languages": []
  },
  "client_request_id": "synthetic-request-001"
}
```

```json
{
  "request_id": "synthetic-server-001",
  "snapshot_id": "synthetic-snapshot",
  "status": "partial",
  "parser_status": "fallback",
  "branch_statuses": {
    "visual": "complete",
    "ocr": "complete",
    "asr": "timeout"
  },
  "results": [],
  "warnings": ["ASR branch vượt deadline; kết quả hiện tại chưa đầy đủ."],
  "timings": {"elapsed_ms": 2000, "measured": false},
  "next_cursor": null,
  "truncated": true,
  "synthetic": true
}
```

Example deliberately zero results to show partial envelope; actual Result item đầy đủ tại [schema example](../schemas/result.synthetic.json). Production timings measured=true với real clock; synthetic envelope tuyệt đối không serve. Status enum complete/partial/no_supported_match/empty; branch complete/timeout/failed/not_requested/missing_index. HTTP200partial không đồng nghĩa không có matching videos.

Pagination cursor opaque contains request/snapshot/result-version/rank; filter đổi→new search. TopK50 limited server-side; no unlimited extraction bypass. Results ordered deterministic pipeline§13.12. Evidence endpoint projection phải resolve full archived provenance đúng snapshot.

## 6. Endpoint inventory

| Method/path | Contract |
|---|---|
| POST /v1/search | body trên; request ID+result items+warnings/timings |
| GET /v1/search/{id}/events | SSE initial/refined/complete/error; optional Accurate |
| DELETE /v1/search/{id} | cancel pending compute; không xóa search corpus |
| GET /v1/media/{video_id} | allowed media resolver, Range/206, source/proxy codec |
| GET /v1/media/{video_id}/timeline?around_us=N | neighboring ordinals/PTS và proxy mapping, bounded response |
| GET /v1/frames/{video_id}/{index} | exact frame bytes, X-Frame-Index/X-PTS/X-Timeline-Revision headers |
| GET /v1/thumbnails/{sampled_frame_id} | immutable image/ETag; placeholder on explicit error |
| GET /v1/evidence/{evidence_id}?snapshot_id=S | typed payload+provenance; no labels/GT |
| POST /v1/exports | selected moment IDs+snapshot+format, returns file; no external submission |

409 stale snapshot không silently remap selected moment vào new corpus. 429 capacity hiển thị retry;503 snapshotUnavailable không empty results giả. Backend file-path traversal denied; filenames escaped; OCR/ASR/query text rendered as text, không HTML; CSV cell bắt đầu =,+,−,@ được safe-escape theo export policy để tránh spreadsheet formula injection.

## 7. Export

Generic CSV columns:
`rank,video_id,source_video_id,start_seconds,end_seconds,start_original_frame_index,end_original_frame_index,representative_original_frame_index,frame_index_base,coordinate_system,ranking_score,score_type,match_status,snapshot_id`.
JSON includes full evidence/timing/provenance. UTF-8 BOM optional user config; NFC; use decimal dot; null fields blankCSV/nullJSON.

Official competition format **chưa biết**. Adapter accepts config + validator + official example supplied later; one-based conversion chỉ ở adapter, internal vẫn0-based. Không giả định họ cần frame ID, timestamp, clip ID hay source ID; không có auto-submit trongM7.

## 8. E11 acceptance

Real fixture VFR, nonzeroPTS, no-audio, corrupt thumbnail, media404, delayed ASR branch, stale response, reordered Accurate list. Assert displayed fields equal API; source frame image hash equals ledger; stepping to next ordinal exact; seeking approximate flagged; end playback pause tolerance measured by browser platform; copy/export include coordinate convention.

Opening card/replay/step must trigger0encoder/LLM/OCR/ASR calls. API integration test branch timeout shows partial; no-supported-match partial cards visible. Keyboard/accessibility smoke, Top50 scroll/network bytes and time-to-first-card measured. Không claim UI competition-ready tới khi chạy với real data và official export adapter được xác định.
