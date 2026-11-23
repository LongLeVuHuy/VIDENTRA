# VIDENTRA — Project instructions


- Trao đổi và viết tài liệu chủ yếu bằng tiếng Việt; giữ identifiers/technical terms bằng English.
- Thư mục nhật kí ở C:\Users\DELL\Desktop\VIDENTRA\docs\PROGRESS.md. trước khi làm nhớ đọc qua. Sau khi làm xong nhớ cập nhật
- Tập validations nằm ở thư mục validations và tập test ở thư mục test_queries (tỷ lệ 80% validations và 20% test trên tổng số data videos)
- Đây là greenfield VIDENTRA. Không kế thừa B4 EVOI, Selective Verification hoặc Evidence Sufficiency Ledger mặc định.
- Blueprint đã được người dùng duyệt ngày 05/11/2026; M0-CODE-01 đã được phép triển khai. Tiếp tục theo milestone được giao, không tự mở rộng sang full engine.
- Trước coding, đọc README, docs/blueprint/05_VIDENTRA_DETAILED_PIPELINE.md, 06_DATA_SCHEMAS_AND_TIMESTAMPS.md và milestone hiện tại trong10_IMPLEMENTATION_ROADMAP.md.
- Target hardware: RTX3060 6GB VRAM, RAM32GB. Thông số CPU/SSD/TGP chưa biết; không báo tốc độ/latency chưa đo.
- Ground Truth chỉ evaluator đọc. Không dùng gold video IDs/timestamps để chọn candidates, sampling hoặc reranking.
- Frame indexes lấy decoded presentation-order ledger, zero-based; timestamps từ PTS/time_base và explicit mappings. Không timestamp×FPS cho VFR.
- Original query luôn giữ; OCR/ASR tìm độc lập corpus; missing evidence là unknown, không automatically false.
- Model outputs không là human truth; scores không calibrated probabilities nếu chưa calibration.
- Preserve raw assets, annotations, source receipts, human reviews, checkpoints và historical runs. Không overwrite hoặc xóa vì tên cache.
- Deletion/deployment cần authorization riêng; old-project deletion chỉ sau verified independent backup và concrete manifest được duyệt.
- Không đưa secrets/rawmedia/checkpoints/venv vào Git. Runtime paths qua VIDENTRA_* config; no legacy source imports.
- Chỉ thêm dependencies có lý do; model/processor/weights phải pin revision/hash/license.
- Tests theo milestone; ghi đúng commands chạy, kết quả, hardware và limitations. Không gọi planning QA là engine benchmark.
- Sau task cập nhật progress/decision records mới; không ép sửa workbook/journal cũ khi không có experiment mới.
- Không tự spawn sub-agents nếu user hoặc instructions áp dụng chưa yêu cầu.
- code gọn gàng không spawn các dòng code lặp lại, tổ chức code phải logic
- code dễ dàng bảo trì
- Các files hoặc folders tạm được sinh ra trong lúc chạy thì sau khi chạy xong hãy xóa những file tạm đó để thư mục project luôn gọn gàng
- Mỗi khi gọi subagents để làm việc thì khi làm xong hãy giải phóng subagents dó
- Cho phép sử dụng Internet để phân tích và nghiên cứu
- Những công việc embedding nặng thì mang lên Kaggle notebook để sử dụng GPU của nó
- Sau khi hoàn thành sau một công việc gì đó thì cập nhật vào mục PROGRESS.md ở C:\Users\DELL\Desktop\VIDENTRA\docs
- Cập nhật PROJECT.md về cấu trúc của chương trình
- Cập nhật file requirements.txt cho các gói thư viện mới cần dùng
- Commit github khi đã hoàn thành tự ghi commit message nếu chưa có repo thì tạo repo tên project. Thêm nữa nếu như có chỉnh sửa gì pipeline, công thức, hoặc thêm các chức năng thì cập nhật README.md luôn
- Mỗi khi chạy benhmark xong thì  luôn cập nhật các file benchmark cũ bằng benchmark vừa chạy rồi cập nhật README.md
- Tổ chức thư mục và đặt tên file/folder phải gọn gàng, logic và tự mô tả rõ mục đích/nội dung bên trong (`self-documenting`: ví dụ tách `data/datasets/<id>/`, `data/processed/{catalog,visual_keyframes,visual_clips,text_evidence}/`, `data/cache/{query_embeddings,query_plans,calibration}/`) để người mới nhìn vào hiểu ngay file hoặc thư mục đó chứa gì và làm gì.
- Phân tách rạch ròi 3 phân vùng dữ liệu MSR-VTT (Zero Leakage Rule): (1) `validation/msrvtt/valid_tuning/` và `data/datasets/msrvtt_val_tune/` (`2.000` videos) chỉ dùng để sweep/tuning siêu tham số; (2) `validation/msrvtt/valid_confirm/` và `data/datasets/msrvtt_val_confirm/` (`2.000` videos) chỉ dùng để kiểm chứng độc lập và khóa Champion trước khi test; (3) `test_queries/msrvtt/test_1ka_benchmark/` và `data/datasets/msrvtt/` (`1.000` videos MSR-VTT 1K-A) là tập niêm phong CHỈ dùng để chạy benchmark đo độ chính xác cuối cùng — tuyệt đối KHÔNG được huấn luyện, validate hay tune siêu tham số trên tập test. Tập `train_9k_clean` (`4.585` videos) chỉ dùng cho huấn luyện ranker/adapter/query-bank priors.
- Quy tắc đặt tên chuẩn sạch (Clean Canonical Naming): Các phương pháp, chiến lược, trọng số và tệp mã nguồn đã chốt dùng tên chuẩn trực tiếp (`hybrid_adaptive`, `hybrid_legacy`, `videntra_ltr.json`, `rebuild_corpus_hybrid_adaptive.py`, `run_msrvtt_matrix.py`, `test_ranking_upgrades.py`), không gắn hậu tố phiên bản `_v1`/`_v2` dư thừa; chỉ giữ alias tương thích ngược trong lớp nạp cấu hình (`config/`) khi cần.
- Quy tắc trình bày bảng benchmark trong tài liệu (`README.md`): Cột đầu tiên phải đặt tên là **"Phương pháp"** và ghi rõ tên mô hình hoặc tổ hợp thành phần dễ hiểu cho người đọc (ví dụ: `SigLIP2-Base-384`, `DFN5B-CLIP-ViT-H-14-378`, `6 Backbones + Query-Bank + Multi-Prompt`, `VIDENTRA + Qwen2.5-VL-3B (NF4, 6GB Champion)`), tuyệt đối không dùng mã config nội bộ (`B0`, `B1`, `R8`, `R9a`–`R9d`) thay cho tên phương pháp; các mã config ID chỉ lưu trong script và JSON receipts kiểm toán (`benchmark/`).


