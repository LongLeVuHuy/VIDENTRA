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


