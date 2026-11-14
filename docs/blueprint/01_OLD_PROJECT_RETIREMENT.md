# 01 — Audit, bảo toàn, di chuyển và retirement

## Phạm vi đã kiểm tra

Old root: `C:\Users\DELL\Desktop\Multimodal-Semantic-Search`. New root: `C:\Users\DELL\Documents\ChatGPT\Multimodal search\VIDENTRA`. Audit ngày04/11/2026 đọc AGENTS/README, source path và semantic parser, journal, dataset registry, chiến lược GT hiện có, model config/files, manifests và experiment/review directories. Không chạy engine cũ hoặc xác nhận lại mọi kết quả lịch sử.

[File inventory](../audit/legacy_inventory.json) có29.886 entries gồm29.885 files và1 environment directory chưa expand; tổng14.272.613.516 bytes ngoài venv và external junction target. Inventory không follow reparse point, không đếm alias hai lần. Classification file-level là provisional; MIGRATE không có nghĩa đã import hay đã đủ quyền.

| Asset canonical | Số file | Bytes | Classification và xử lý |
|---|---:|---:|---|
| data/raw |5693|8371941954|MIGRATE: giữ media/official labels/receipts; registry mới tham chiếu hash, không tái gọi old test thành fresh |
| data/processed |344|180262513|INVESTIGATE: frozen cache/provenance cần giữ; embeddings có thể REGENERATE sau parity |
| model/siglip2 |8|1539456768|MIGRATE checkpoint đã có, verify model revision trước reuse |
| model/asr |29|970705024|MIGRATE PhoWhisper; external Faster-Whisper thêm riêng |
| model/ocr |31|164776028|MIGRATE/benchmark VietOCR và Paddle reference, charset/license review |
| model/spatial |22|690312479|ARCHIVE, chưa mandatory new runtime |
| lab/results |22897|1940754099|ARCHIVE mọi historical results/source snapshots/labels, không import làm benchmark mới |
| lab/src, tests, tools |392|5514614|ARCHIVE; chỉ reimplement/port utility có justification |
| lab/notes |8|1005700|ARCHIVE journal và learning records |
| docs/research + rebuild_2026 |58|814921|ARCHIVE nghiên cứu, kể cả phương án bị bỏ |
| Desktop/Multimodal_Search_Lab.xlsx |1|92158|ARCHIVE canonical tracker ngoài old root;5 sheets, ZIP CRC checked |
| .venv |chưa đếm|chưa đo|REGENERATE sau lockfile; chưa đủ điều kiện xóa |

Các thư mục root configs, lab/configs, Kaggle bundles, notebooks, backup workbook, root requirements/AGENTS/README đều có trong archive; bảng trên chỉ nhóm quan trọng, không phải toàn listing. [Workbook audit](../audit/legacy_workbook.json) kiểm tra cấu trúc/CRC, không revalidate metrics/công thức.

### Junction và Git thực tế

- `lab/data`→old/data; `results`→old/lab/results: aliases cùng bytes.
- `model/asr/whisper_small`→Hugging Face snapshot `536b0662742c02347bc0e980a01041f333bce120` ngoài repo; target files được archive riêng.
- Không có `.git` độc lập ở old root. `git rev-parse` resolve tới `C:/Users/DELL`; ancestor branch master **chưa có commit**, path-scoped `git ls-files` không có tracked asset old project. Vì vậy không có lịch sử commit old-project tại máy để bundle. Không archive toàn home Git vì vượt phạm vi.
- Nếu tồn tại remote/ổ khác chưa biết: INVESTIGATE tại M0; không khẳng định lịch sử ở nơi khác không tồn tại.

## Backup đã thực hiện và kiểm chứng

Archive: `C:\Users\DELL\Documents\VIDENTRA-retirement-backups\20261104\legacy_assets.zip`.

- ZIP_STORED/ZIP64 chứa **29.890 files**, source bytes14.758.918.046; archive14.767.531.636 bytes.
- Bao gồm toàn bộ files đã inventory ngoài venv, external Faster-Whisper files và workbook canonical.
- Mỗi file được SHA-256 trong lúc copy; stat trước/sau để phát hiện thay đổi; đã đọc lại **toàn bộ archive members** và đối chiếu29.890 hashes.
- Archive SHA-256: `43b6207f97f99443d4c9ee6645a3e5bda44134261a80cf48cc6a5fa143866a75`.
- Free space trước92.113.473.536 bytes; sau archive + initial restore test75.907.973.120 bytes. Đây là snapshot tại thời điểm chạy, không dung lượng còn trống vĩnh viễn.
- Restore test thực có journal snapshot, JSON raw-progress, workbook, MP4 và external model.bin, bytes khớp. [Supplemental restore](../audit/restore_supplement.json) đã khôi phục canonical journal và official QVH labels (1.550 JSONL rows), đồng thời ffprobe đọc được MP4 H.264 dài 11 giây. Đây không phải kiểm tra decode mọi frame hoặc reload model.
- [Receipt chính](../audit/backup_verification.json); full per-file hash manifest nằm cạnh archive; scripts audit/backup trong repository mới chỉ là planning tooling.

**Giới hạn:** backup nằm cùng volume C:. Nó bảo vệ khỏi thay đổi/xóa nhầm project nhưng không bảo vệ hỏng/mất ổ máy. Trước retirement thật cần copy archive+manifest+receipt ra ổ ngoài hoặc destination độc lập, verify lại SHA-256, thử restore ở máy/path khác và kiểm tra quyền truy cập. Không có remote backup hay disaster-recovery guarantee được tuyên bố.

## Phân loại migration

| Category | Quy tắc |
|---|---|
| MIGRATE | raw official data, labels, provenance, receipts/checksums, checkpoint có license/revision; registry mới chỉ references immutable store |
| ARCHIVE | research/journals/workbooks, historical configs/runs/source; giữ byte-identical, có mapping old path |
| REGENERATE | venv, reconstructable thumbnails/ANN/embeddings **chỉ sau** recipe/source/license/revision pinned và verified rebuild |
| DELETE_AFTER_BACKUP | bytecode/cache được chứng minh disposable; vẫn cần explicit deletion authorization riêng |
| INVESTIGATE | labels/review provenance chưa rõ, old holdout, external caches/remote history, unknown licenses, processed artifacts chưa tái tạo được |

Human review packet trống/`review_pending` không được nâng thành GT. QVH24 local đã dùng development; gold-informed AVC300 gallery và cache229 chỉ diagnostic. Các labels official và human-reviewed có cùng mức ưu tiên bảo toàn nhưng provenance khác nhau.

## Deletion manifest và approval gate

[deletion_manifest.json](../audit/deletion_manifest.json) liệt kê canonical paths/categories, backup references, eligibility và `approved=false`. **Hiện không file nào được phê duyệt xóa.** Có thể đề xuất bytecode/venv/derived rebuildables trước, nhưng chỉ sau restore/rebuild evidence, không dựa vào tên “cache”.

Trình tự retirement sau planning:
1. Freeze old root writes; delta audit so manifest, backup phần thay đổi.
2. Verify independent backup location, permissions/free space; copy/archive hash, restore raw+labels+models+journal/workbook ở đường dẫn khác.
3. Dựng VIDENTRA asset registry với original IDs, hashes, exposure và license; smoke decode media, load compatible checkpoints và compare relevant utility outputs.
4. Thực hiện pilot E2E dùng new code, giữ old assets và rollback path; không alias runtime về old lab modules.
5. Trình người dùng manifest cụ thể: path, bytes, hash, backup location, reason, restore recipe, retention. Xin quyền xóa **sau** những kiểm chứng này.
6. Nếu được phép, xóa các targets được nêu bằng canonical path trong một shell, không traverse junction. Không dùng git clean -fdx hoặc recursive deletion root trong planning.
7. Reconcile deletion receipts và test restore; giữ backup theo retention được người dùng chọn, mặc định chưa đặt expiry.

Không có thao tác destructive, di chuyển raw data hoặc overwrite old research nào trong assignment này. Chính sách cập nhật journal/roadmap cũ không được mang sang ép tiếp tục B4; tiến độ greenfield ghi tại README/roadmap mới.
