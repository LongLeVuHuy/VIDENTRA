# 10 — Roadmap triển khai

**Cập nhật 05/11/2026: blueprint đã được duyệt; M0-CODE-01 hoàn tất.** Xem [tiến độ](../PROGRESS.md). CLI manifests/registry/fixtures và tests M0 đã tồn tại và chạy; các lệnh retrieval/benchmark ở milestone sau vẫn là future contracts.

Bản kế hoạch gốc: **Planning đã tạo repository, tài liệu và audit. Chưa triển khai engine.** Chỉ bắt đầu coding sau review blueprint theo yêu cầu người dùng. Các lệnh `python -m videntra...`, pytest và npm bên dưới là **CLI/test contracts cần tạo ở milestone tương ứng**, chưa tồn tại và chưa chạy. Planning QA dùng script riêng đã có.

## Task coding đầu tiên sau design approval

**M0-CODE-01: minimal package, immutable media/dataset registry và timeline fixture harness.**

1. Tạo `pyproject.toml`, `src/videntra/contracts` và CLI skeleton theo [12](12_NEW_REPOSITORY_STRUCTURE.md). Pin dependencies sau environment smoke; không chép toàn bộ requirements cũ.
2. Implement schemas và semantic validators từ [06](06_DATA_SCHEMAS_AND_TIMESTAMPS.md); thêm split/exposure/rights manifests. Đăng ký 24 QVH assets hiện có theo hash; labels nằm riêng trong evaluation.
3. Tạo tiny fixtures CFR, VFR, nonzero PTS và audio delay bằng FFmpeg, kèm expected rational timeline. Task này không gọi OCR/ASR/LLM hoặc import old lab modules.
4. Chạy schema, registry, idempotency và GT-isolation tests; sau đó mới chuyển sang decode ledger ở M1.

Task đầu không cần ingest 10.000 giờ hoặc tải model mới. Backup cục bộ đã verify; việc tạo bản off-device và giải quyết rights/exposure tiếp tục ở M0. Chưa xóa tài sản cũ.

## M0 — Reproducibility, migration an toàn và registry (ước lượng 2–4 ngày kỹ sư)

- **Phụ thuộc:** blueprint được review; audit/backup receipts hiện có. Không cần chờ xóa dự án cũ.
- **Đầu vào:** 13 tài liệu, legacy manifest, raw media references, model revisions và dataset sources.
- **Công việc:** tạo package/config conventions, validators, model/data registries; map tài sản mà không copy source; audit split/exposure; thêm backup độc lập khi có destination.
- **Đầu ra/module:** `contracts/`, `config/`, `registry/`, `cli.py`; manifests, lockfiles và migration receipt.
- **Lệnh kiểm tra:** `python -m videntra manifests validate --all`; `pytest tests/contracts tests/registry tests/evaluation/test_isolation.py`.
- **Unit/integration cases:** stable IDs sau restart; duplicate hash alias; rights chưa rõ không được activate; runtime bị từ chối truy cập GT; source hash không khớp bị reject.
- **Metrics/tài nguyên:** số lỗi schema, rights/exposure chưa giải quyết, backup hash/restore; CPU và RAM 32 GB, chưa cần GPU.
- **Go:** active assets có định danh, quyền và exposure; manifest round-trip; không import old modules; backup reference hợp lệ. Off-device backup bắt buộc trước xóa, không phải trước viết pipeline mới.
- **No-go/rollback:** quarantine tài sản chưa rõ; restore registry revision trước; giữ nguyên dự án cũ.

## M1 — Ingestion, timeline và adaptive sampling (ước lượng 1–2 tuần)

- **Phụ thuộc:** M0; FFmpeg/PyAV đã pin và smoke thành công.
- **Đầu vào:** tiny media fixtures, QVH dev subset, media manifests.
- **Công việc:** O00–O02; sequential frame ledger; audio mapping; xử lý corrupt/duplicate/restart; hybrid sampler có max-gap, shot/burst và text probes.
- **Đầu ra/module:** `ingestion/probe.py, decode.py, jobs.py`; `timeline/maps.py, ledger.py`; `sampling/hybrid.py, probes.py`; Parquet manifests.
- **Lệnh kiểm tra:** `python -m videntra ingest --manifest manifests/dev_media.jsonl --resume`; `python -m videntra sampling compare --config configs/experiments/e02.yaml`; `pytest tests/timeline tests/sampling tests/ingestion`.
- **Cases:** CFR/VFR/B-frames/repeated PTS/nonzero start/audio delay/source offset; seek so với sequential frame hash; max-gap trừ decode holes đã ghi; kill/restart không tạo duplicate.
- **Metrics/tài nguyên:** E01/E02, decode fps, density, overhead và coverage; CPU/disk, chưa bắt buộc GPU.
- **Go:** ledger khớp fixtures; không bỏ qua decode failure; đủ trạng thái completeness của 24 clip thật; kiểm tra sampler bằng human events. Thiếu quality data thì dùng policy bảo thủ và gắn provisional.
- **No-go/rollback:** fallback max-gap; giữ media và failed attempts; không “sửa” timestamp bằng nominal FPS.

## M2 — SigLIP 2 và exact FAISS baseline (ước lượng 1 tuần)

- **Phụ thuộc:** M1 và sample frames đúng timeline.
- **Đầu vào:** Base224 checkpoint đã archive, chỉ reuse sau hash/license check; dev queries có provenance.
- **Công việc:** O03/O09 tối thiểu với Flat; normalization parity; stable vector registry; original-query search; provisional windows; tách predictions khỏi evaluator.
- **Đầu ra/module:** `encoders/siglip2.py`, `indexing/faiss_store.py, ids.py`, `retrieval/visual.py`, `evaluation/metrics.py`; baseline tái lập trên 24 video.
- **Lệnh kiểm tra:** `python -m videntra index build --config configs/baseline_flat.yaml`; `python -m videntra search --query-file manifests/dev_queries.jsonl --output experiments/m2/predictions.jsonl`; `python -m videntra evaluate --run experiments/m2 --protocol qvh_dev_known_positive`; `pytest tests/encoders tests/indexing tests/retrieval`.
- **Cases:** cosine/Flat parity; rebuild giữ IDs; dtype/norm/NaN; query dài được giữ; runtime không nhận qrels.
- **Metrics/tài nguyên:** E03, VideoHit/MomentHit/MRR đúng scope, FP16 loss, embedding throughput, peak VRAM; RTX 3060 với batch khởi đầu 16 và giảm thích nghi về 1.
- **Go:** query trả video/moment/frame thật, phát lại được, truy vết được. Báo baseline đo được; không bịa quality target đạt sẵn.
- **No-go/rollback:** FP32/CPU hoặc giảm batch; giữ exact baseline khi thử ANN.

## M3 — OCR và ASR có adaptive routing (ước lượng 2–3 tuần)

- **Phụ thuộc:** timeline M1, baseline M2, source/charset gates.
- **Đầu vào:** high-resolution probes và audio độc lập; OCR/ASR dev có annotation cùng joint annotation subset.
- **Công việc:** O04–O07; shared detector/crops; PP-OCRv5 general và VietOCR; PP-OCRv6/VieSVTR có gate; VAD, Whisper-small/PhoWhisper; routing, dedup và provenance.
- **Đầu ra/module:** `ocr/detection.py, recognition.py, routing.py, tracking.py`; `asr/vad.py, transcription.py, alignment.py`; `evidence/normalize.py, store.py`.
- **Lệnh kiểm tra:** `python -m videntra evidence build --manifest manifests/dev_media.jsonl --modalities ocr,asr`; `python -m videntra benchmark run --experiment E04`; `pytest tests/ocr tests/asr tests/evidence`.
- **Cases:** NFC/dấu Việt; singleton OCR là instant; cùng text ở hai thời điểm không gộp; VAD concatenation maps; no-speech khác missing; code-switch; confidence không phải probability.
- **Metrics/tài nguyên:** CER/WER, detection/timing errors, fallback ratio, false negatives có confidence cao; GPU jobs tuần tự, thử specialist trên CPU khi phù hợp.
- **Go:** hai candidate models được thử nhỏ có kiểm soát; chọn routing theo gain/cost; audio timeline đúng. Specialist không có lợi thì giữ interface và dùng primary-only.
- **No-go/rollback:** tắt recognizer/route lỗi; giữ recipes cũ. Không thông qua gate tiếng Việt của PP-OCRv6 bằng sửa dictionary đơn thuần.

## M4 — Multimodal candidate union độc lập (ước lượng 1 tuần)

- **Phụ thuộc:** visual M2 và evidence M3.
- **Đầu vào:** visual/text indexes có version; queries phân tầng OCR/ASR/visual.
- **Công việc:** FTS5 lexical, E5 tùy chọn; global modality searches; time-compatible proposals; grouped RRF, branch reservations và atomic snapshots.
- **Đầu ra/module:** `indexing/text_store.py, snapshots.py`; `retrieval/branches.py, union.py, fusion.py`; branch traces.
- **Lệnh kiểm tra:** `python -m videntra benchmark run --experiment E06`; `pytest tests/retrieval/test_union.py tests/retrieval/test_temporal_fusion.py tests/indexing/test_snapshots.py`.
- **Cases:** candidate ngoài visual Top-K vẫn được OCR/ASR tìm lại; SALE ở xa không hỗ trợ event hiện tại; snapshot nhất quán khi update; branch timeout trả partial.
- **Metrics/tài nguyên:** CandidateHit@K/tIoU, rescue rate, mất recall theo branch, latency và FTS size; CPU/RAM, text encoder có cap.
- **Go:** trước cap, union giữ constituent seeds; quota hoạt động; adaptive routing mất không quá 1 điểm phần trăm recall theo stratum là mục tiêu kiểm chứng.
- **No-go/rollback:** giữ parallel union/equal weights; không che recall loss bằng reranker.

## M5 — QueryGraph từ small LLM (ước lượng 1–2 tuần)

- **Phụ thuộc:** M4, human query-structure labels và deterministic fallbacks.
- **Đầu vào:** VI/EN query graphs; frozen prompt/schema revisions.
- **Công việc:** simple rules; so Qwen3-0.6B/1.7B; constrained compact JSON; schema/semantic checks; original-query floor; cache/deadline.
- **Đầu ra/module:** `query/rules.py, parser.py, graph.py, validation.py, cache.py`; E07 report.
- **Lệnh kiểm tra:** `python -m videntra benchmark run --experiment E07`; `pytest tests/query tests/retrieval/test_original_fallback.py`.
- **Cases:** đảo vai trò; negation scope; temporal cycle; span không có trong query; xanh blue/green; “orders coffee”; JSON lỗi; timeout; query dài nhiều mệnh đề.
- **Metrics/tài nguyên:** Predicate F1, binding/temporal/negation accuracy, hallucination, JSON validity, E2E accuracy, parser p95; đo memory quantized 0.6B/1.7B.
- **Go:** graph có ý nghĩa, không bỏ constraints âm thầm. Không đạt quality gate thì parser chỉ advisory; Fast rules/cache fallback vẫn hoạt động.
- **No-go/rollback:** original+rules không bị ảnh hưởng; không fine-tune ngay để che lỗi retrieval.

## M6 — Relation-aware temporal matching (ước lượng 3–5 tuần; rủi ro nghiên cứu cao)

- **Phụ thuộc:** M4/M5; human event/role/speaker labels độc lập.
- **Đầu vào:** candidate pools từ các retrieval runs không dùng GT; evidence graph; hard-negative pairs.
- **Công việc:** entity assignment, three-valued logic, temporal beam/DP; attributes/2D/ordering; candidate tracking/action adapters; native grounder và bounded VLM challengers.
- **Đầu ra/module:** `relations/entities.py, predicates.py, temporal.py, matcher.py, adapters/`; `localization/proposals.py, refine.py`; `ranking/features.py, ranker.py`.
- **Lệnh kiểm tra:** `python -m videntra benchmark run --experiment E08`; `python -m videntra benchmark run --experiment E09`; `pytest tests/relations tests/localization tests/ranking`.
- **Cases:** đủ 9 loại hard negative; transcript không chứng minh người nói; bbox không chứng minh give; track mất qua cut → unknown; beam truncation không có nghĩa unsatisfiable; phủ định giữ đúng scope.
- **Metrics/tài nguyên:** pair contrast accuracy/CI, false-supported rate/coverage, MomentHit và boundary MAE; optional model ≤5 GiB active, có thể không đạt Accurate 8 s.
- **Go:** B qua E08; mở từng relation riêng. Chỉ công bố hỗ trợ handover/speaker/behind khi từng adapter được kiểm chứng; còn lại UI báo partial.
- **No-go/rollback:** trả A hoặc B cơ bản với unknown; tắt C nếu quality/cost không đạt; lưu failures làm research record.

## M7 — Competition UI và API (ước lượng 1–2 tuần; UI cơ bản có thể bắt đầu sau M2)

- **Phụ thuộc:** Result contract ổn định; M2 đủ cho inspector đơn giản, M4–M6 bổ sung evidence.
- **Đầu vào:** API schema, real playable media và generic export settings.
- **Công việc:** FastAPI, React grid, Top-K/modes, source/proxy playback, exact-frame navigation, keyboard, comparison/export và errors.
- **Đầu ra/module:** `api/search.py, media.py, frames.py, exports.py`; `ui/`; integration fixtures.
- **Lệnh kiểm tra:** `pytest tests/api`; `npm --prefix ui run test`; `npm --prefix ui run test:e2e`; `python -m videntra benchmark run --experiment E11`.
- **Cases:** VFR frame-step parity; hủy stale response; mở/replay không gọi model; API errors; CSV formula-safe escaping.
- **Metrics/tài nguyên:** time-to-first-results, playback delay, số thao tác tới mục tiêu, frame correctness; một player và thumbnail LRU.
- **Go:** mặc định Top-10, đủ metadata, frame-index scope chính xác, replay được. Official exporter chờ competition schema thực tế.
- **No-go/rollback:** source-frame inspector nếu browser seek không chính xác; rollback API schema version khi cần.

## M8 — Optimization và scale (ước lượng 2–4 tuần; chưa tính thời gian full ingest)

- **Phụ thuộc:** baseline qua quality gates; máy đích/storage thực tế.
- **Đầu vào:** manifests ở 100k → 1M → 10M → target vectors; representative ingest tối thiểu 10 giờ.
- **Công việc:** E05 index grid, shard/delta/tombstones; E10 profiling; CPU threads/GPU admission; snapshot load/recovery; media/storage plan.
- **Đầu ra/module:** `indexing/shards.py, compaction.py`; `runtime/scheduler.py, budgets.py, telemetry.py`; báo cáo capacity đo được.
- **Lệnh kiểm tra:** `python -m videntra benchmark run --experiment E05 --scale 1000000`; `python -m videntra benchmark run --experiment E10 --hardware target_3060`; `pytest tests/runtime tests/indexing/test_recovery.py`.
- **Cases:** OOM/restart/quyền bị thu hồi/stale snapshot; exact NN audit; concurrent queries; cold/warm/cache states.
- **Metrics/tài nguyên:** p50/p95/deadline misses, RTF/fps, ANN recall, peak RAM/VRAM/disk. Không mặc định đã có external storage hàng TB.
- **Go:** cấu hình đo thực đáp ứng recall và mục tiêu warm p50 ≤1 s, p95 ≤3 s; báo riêng Fast allocation 2 s và Accurate 8 s. Dự báo 10.000 giờ kèm uncertainty trước full ingest.
- **No-go/rollback:** giảm active index, CPU shards, bố trí lịch offline. Nếu thiếu tài nguyên phải điều chỉnh phạm vi/hardware; không tuyên bố scale đạt rồi.

## M9 — Fine-tuning parser có điều kiện (ước lượng 1–3 tuần, tùy chọn)

- **Phụ thuộc:** E07 chứng minh parser gap đáng kể; có verified labels, lợi ích E2E dự kiến, quyền dữ liệu và compute. Không bắt buộc cho MVP.
- **Đầu vào:** human-reviewed graphs; train/dev/test tách source và paraphrase families; label provenance rõ.
- **Công việc:** frozen prompting baseline trước; LoRA so QLoRA bằng PEFT/TRL; context ≤2048, rank 8/16; teacher output là pseudo-label trước human review.
- **Đầu ra/module:** `training/parser_sft.py`, adapters/manifests, model card và so sánh frozen prompt baseline.
- **Lệnh kiểm tra:** `python -m videntra.training.parser_sft --config configs/train/parser_lora.yaml`; `python -m videntra benchmark run --experiment E07 --parser tuned`; `pytest tests/training/test_data_isolation.py`.
- **Cases:** không có holdout labels/video IDs trong training; adapter load/fallback parity; validity khi quantized inference.
- **Metrics/tài nguyên:** parser metrics cùng retrieval accuracy/latency; đo train feasibility trên 6 GB. External GPU chỉ dùng khi thực sự có và được phép.
- **Go:** cải thiện strata cần thiết và E2E, không regression hoặc vượt budget.
- **No-go/rollback:** chuyển active pointer về prompting/rules; archive checkpoint và report để truy vết.

## Lịch và checkpoints

Ước lượng effort dành cho một kỹ sư có hỗ trợ AI, không phải deadline cam kết hoặc GPU throughput. Data annotation/quyền truy cập có thể chi phối tiến độ. Critical path: M0 → M1 → M2 → M3 → M4 → M5 → M6. UI cơ bản có thể làm theo dependencies; M8 cần baseline hợp lệ; M9 tùy chọn.

Core có thể cần khoảng **12–20+ tuần kỹ sư**, tùy annotation và nghiên cứu; chưa tính full 10.000 giờ ingest. Lịch học/part-time cần quy đổi riêng.

Sau mỗi milestone, append quyết định, lệnh/kết quả thực chạy, run hashes, vấn đề chưa giải quyết và gate tiếp theo. Review trước scale-up; không sửa frozen labels để test pass. Failed stage không overwrite historical successful run. Coding agent đọc AGENTS, pipeline và milestone, làm đúng phạm vi rồi báo các kiểm tra thực sự đã chạy.

