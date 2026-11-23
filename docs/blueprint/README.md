# VIDENTRA Architecture & Specification Index

Thư mục `docs/blueprint/` lưu trữ các đặc tả kiến trúc, hợp đồng dữ liệu và chuẩn tổ chức mã nguồn cốt lõi của **VIDENTRA**.

## Tài liệu Kiến trúc Đang Áp Dụng

| Tài liệu | Nội dung chính |
| :--- | :--- |
| [`05_VIDENTRA_DETAILED_PIPELINE.md`](05_VIDENTRA_DETAILED_PIPELINE.md) | Đặc tả chi tiết luồng Offline Ingestion (`O01–O08`) & Online Retrieval/Evidence Alignment, hợp đồng lỗi và ngân sách tính toán 6GB VRAM |
| [`06_DATA_SCHEMAS_AND_TIMESTAMPS.md`](06_DATA_SCHEMAS_AND_TIMESTAMPS.md) | Hợp đồng schema JSON/SQLite, chuẩn hóa văn bản NFC, thời gian VFR (`0-based presentation-order`, `PTS × time_base`) và job consistency |
| [`10_IMPLEMENTATION_ROADMAP.md`](10_IMPLEMENTATION_ROADMAP.md) | Lộ trình triển khai các cột mốc M0–M9, ngưỡng kiểm định chất lượng (Go/No-Go) và kịch bản test cô lập nhãn |
| [`12_NEW_REPOSITORY_STRUCTURE.md`](12_NEW_REPOSITORY_STRUCTURE.md) | Cấu trúc thư mục chuẩn hiện hành của repository (`src/videntra/`, `scripts/`, `tests/`, `data/`, `validation/`, `test_queries/`), quy tắc đặt tên sạch và chính sách môi trường |

## Tài liệu Liên Quan Khác

- [`../../README.md`](../../README.md) — Tổng quan tính mới, workflow Mermaid Offline/Online, công thức toán học, kết quả benchmark `MSR-VTT 1K-A` & keyframe và hướng dẫn CLI.
- [`../../PROJECT.md`](../../PROJECT.md) — Danh mục chi tiết các module Python (`src/videntra/`), công cụ vận hành (`scripts/`) và lịch sử hoàn thiện từng giai đoạn.
- [`../../AGENTS.md`](../../AGENTS.md) — Quy tắc bắt buộc của dự án (Zero Leakage 3 phân vùng, timeline bất biến, tổ chức thư mục tự mô tả, đặt tên sạch).
- [`../OFFLINE_STORAGE_SPEC.md`](../OFFLINE_STORAGE_SPEC.md) — Đặc tả phân vùng lưu trữ dữ liệu & schema SQLite WAL / FTS5 thực tế.
- [`../PROGRESS.md`](../PROGRESS.md) — Nhật ký tiến độ triển khai và các mốc đo kiểm thực nghiệm.
- [`../schemas/`](../schemas/) — Bộ JSON Schema chuẩn (`dataset`, `media_asset`, `frame_record`, `query_graph`, `result`) và fixtures kiểm thử.
