# 12 — Cấu trúc repository và coding policies

Repository độc lập nằm tại `C:\Users\DELL\Documents\ChatGPT\Multimodal search\VIDENTRA`, branch `codex/blueprint`. Chưa có remote, publish hoặc deployment. Không copy source tree cũ vào repository mới.

## Các artifact đã tạo

```text
VIDENTRA/
  .git/
  .gitignore
  AGENTS.md
  README.md
  docs/
    blueprint/                    # 00–12 và README mục lục
    schemas/                      # JSON Schemas, synthetic examples
    audit/                        # inventory, sources, backup, QA receipts
  scripts/
    audit_legacy_assets.py
    backup_legacy_assets.py
    validate_blueprint.py
```

Script hiện có phục vụ audit và kiểm tra tài liệu; chưa có retrieval engine. Backup binary nằm ngoài repo. Không track raw media, checkpoints hoặc venv. Source cũ được archive riêng.

## Cấu trúc triển khai sau review

```text
VIDENTRA/
  pyproject.toml
  uv.lock                         # chọn lock tool sau smoke trên máy đích
  src/videntra/
    cli.py
    config/
    contracts/                    # typed records, semantic validators
    registry/                     # assets, models, datasets, stable IDs
    ingestion/                    # probe, decode, jobs, recovery
    timeline/                     # PTS, ordinals, source/proxy maps
    sampling/                     # hybrid sampler, text probes độc lập
    encoders/                     # SigLIP, text encoders, revisions
    ocr/                          # detector, recognizers, router, tracks
    asr/                          # audio, VAD, transcription, alignment
    evidence/                     # typed payloads, interval lookup
    indexing/                     # FAISS, text, shards, snapshots
    query/                        # rules, LLM graph, validation, cache
    retrieval/                    # independent branches, union, fusion
    relations/                    # roles, temporal constraints, adapters
    localization/                 # proposals, refinement, NMS
    ranking/                      # features, scoring, calibration
    runtime/                      # admission, deadlines, telemetry
    api/                          # search, media, frames, export
    evaluation/                   # inference không import module này
    training/                     # parser SFT tùy chọn sau M9 gate
  ui/                             # React + TypeScript ở M7
  configs/
    models/
    datasets/
    hardware/
    experiments/
    competitions/
    train/                        # chỉ thêm khi kích hoạt M9
  manifests/                      # manifests nhỏ, không credentials
  tests/
    contracts/
    registry/
    ingestion/
    timeline/
    sampling/
    encoders/
    ocr/
    asr/
    evidence/
    indexing/
    query/
    retrieval/
    relations/
    localization/
    ranking/
    api/
    runtime/
    evaluation/
    training/
  experiments/<run_id>/            # immutable outputs; file lớn ngoài Git
  data/                            # pointers tới external storage
  models/                          # pointers và manifests
```

Chỉ tạo module/test khi milestone cần đến. Blueprint là specification, không phải runtime dependency.

Thứ tự xây dựng: contracts → timeline → evidence → indexes → retrieval → relations → ranking → API. Mũi tên biểu thị prerequisite; module phía sau có thể import contract của phía trước. Evaluation tiêu thụ predictions và labels trong process riêng; runtime không import evaluator hoặc nhãn.

## Chính sách AGENTS.md

[AGENTS.md](../../AGENTS.md) quy định trao đổi tiếng Việt, đọc spec trước sửa code, planning cho tới review, bảo toàn tài sản, ghi rõ giả thuyết, GT ngoài inference, giới hạn tài nguyên và experiments bất biến. Xóa/deploy cần authorization riêng.

Chỉ thêm instructions thư mục con khi cần quy tắc riêng, như label isolation, UI API contracts hoặc timeline correctness. Không kế thừa roadmap B4/EVOI và không mặc định ủy quyền cho sub-agent.

## Dependency policy

Chọn Python, FFmpeg, PyAV, FAISS CPU và encoder runtime sau smoke test trên máy đích. Không import mọi model library vào API process. Paddle/PyTorch/CTranslate2 có thể cần worker environments riêng để tránh CUDA conflicts. Trao đổi typed messages/files, không dùng pickle làm giao thức giữa worker.

Ưu tiên native Windows. WSL/Linux là lựa chọn nếu dependency thực tế yêu cầu; không tự đổi OS. GPU jobs tuần tự trên 6 GB. Lưu lockfiles, model SHA, processor hash, CUDA/driver receipt. Planning không cài toàn bộ stack hoặc tải hàng loạt weights.

Credentials không nằm trong config. Environment files bị ignore; example config chỉ chứa placeholder cho secret. Runtime không hardcode đường dẫn cá nhân.

## Storage và experiments

`VIDENTRA_DATA_ROOT`, `VIDENTRA_MODEL_ROOT`, `VIDENTRA_INDEX_ROOT`, `VIDENTRA_RESULTS_ROOT` resolve qua config. Không import đường dẫn legacy.

Production IDs/locators không phụ thuộc absolute path; server registry resolve media URI. File lớn nằm ngoài Git; hashes/revisions nhỏ được version control. Không tự bật Git LFS để lưu hàng TB media.

Checkpoint có license receipt. Cache score cũ không phải benchmark mới. Retention của logs/query text phải được cấu hình; logs không chứa secrets. Auto-cleanup chỉ áp dụng intermediate files tái tạo được của chính hệ thống; không recursive cleanup dự án cũ.

## Testing và review

Pytest theo milestone; property tests cho timeline/ranking khi phù hợp; integration fixtures là media nhỏ kiểm tra được. Giữ exact baseline trước ANN. Relation adapter có positive, contradiction và unknown cases. Model smoke/costly tests nêu resource requirements riêng.

Review timebase/offset, stable IDs, label access, lỗi so với empty result, model/processor versions, fallback khi vượt cap và ý nghĩa score. PR ghi hành vi thay đổi, lệnh thực sự chạy và giới hạn còn lại. Không gọi schema QA là engine benchmark.

Task coding đầu tiên: M0-CODE-01 trong [roadmap](10_IMPLEMENTATION_ROADMAP.md). Source tree ở phần “Cấu trúc triển khai” vẫn là kế hoạch.

