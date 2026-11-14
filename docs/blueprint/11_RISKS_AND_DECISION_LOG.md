# 11 — Rủi ro, giả định và nhật ký quyết định

Ngày 04/11/2026; trạng thái **PROPOSED**, chờ review. Chưa có quyết định nào được chứng minh bằng benchmark VIDENTRA. Ngưỡng thiết kế nằm tại [pipeline](05_VIDENTRA_DETAILED_PIPELINE.md); thí nghiệm kiểm chứng nằm tại [08](08_BENCHMARK_AND_LATENCY_PLAN.md).

## Nhật ký quyết định

| ID | Quyết định hoặc giả thuyết | Lý do | Kiểm chứng và phương án dự phòng |
|---|---|---|---|
| D01 | Repository greenfield; archive dự án cũ | Giữ tài sản, không kế thừa B4 mặc định | E14; không import source cũ |
| D02 | B là đích thiết kế, A baseline, C tùy chọn | Giới hạn 6 GB và khả năng bảo trì | E08/E10; giữ A nếu B không có lợi ích đo được |
| D03 | Frame ledger, PTS rational và piecewise maps | Xử lý VFR, nonzero start và A/V sync | E01/E09; quarantine timeline không hợp lệ |
| D04 | Max-gap 2 s, burst 4 fps, text probe 1 fps | Độ phủ tối thiểu và thêm mẫu ở vùng thay đổi | E02/E04; tăng mật độ hoặc dùng uniform nếu mất recall |
| D05 | SigLIP 2 Base224 ban đầu, lưu FP16 | Checkpoint sẵn có; điểm bắt đầu tiết kiệm tài nguyên | E03; Base384 và FP32 đối chứng |
| D06 | Shared OCR detector, recognizer tiếng Việt có điều kiện | Tránh lặp detector; bảo vệ dấu tiếng Việt | E04/E13; PP-OCRv6 chưa là default VI trước charset/inference gate |
| D07 | Whisper-small và PhoWhisper-small có điều kiện | ASR độc lập video; kiểm tra code-switching | E04; primary-only nếu specialist không có lợi |
| D08 | FAISS Flat → IVF-PQ theo shard; text/metadata trên CPU | RAM 32 GB không chứa toàn bộ Flat ở quy mô đích | E05/E10; điều chỉnh nprobe/shard, audit exact search |
| D09 | Modality tìm độc lập; RRF và quota từng nhánh | OCR/ASR vẫn tìm lại được video ngoài visual Top-K | E06; parallel union baseline |
| D10 | Qwen parser có điều kiện; giữ query gốc | Biểu diễn vai trò/thứ tự và có fallback | E07; rules/original query |
| D11 | Typed graph và bounded binding/temporal search | Kiểm tra nhất quán, không giả định perception đúng | E08; trả unknown khi thiếu bằng chứng |
| D12 | Exact-frame endpoint và generic export | Browser seek không xác định chính xác frame ordinal | E11/E09; source-frame inspector |
| D13 | Tách GT theo task, ghi development exposure | QVH/AVC cũ không phải đánh giá mới | E12/E13; diagnostic riêng, labels ngoài runtime |
| D14 | Warm p50 ≤1 s, p95 ≤3 s; Fast allocation 2 s, Accurate 8 s; RAM ≤24 GiB, VRAM ≤5 GiB | Giữ mục tiêu người dùng, giới hạn thực thi trên 3060/32 GB | E10; tắt stage tùy chọn hoặc xem lại tài nguyên |

Các ngưỡng không suy trực tiếp từ bytes/config đều là giả thuyết ban đầu: OCR confidence 0,80; fallback 20%; audit ngẫu nhiên 5%; OCR IoU 0,5, edit similarity 0,8 và gap 1 s; ngưỡng change của sampler; proposal 4/8/16/32 s; beam 32, tối đa 3 track/entity; search horizon 30/120 s; NMS 0,7; trọng số ranking, candidate quotas và ngưỡng go/no-go. E02/E04/E06/E07/E08/E09/E10 phải kiểm chứng. Chưa gọi các ngưỡng này là “đã tune”.

## Danh mục rủi ro

| Rủi ro | Mức độ và tình trạng | Biện pháp, điều kiện đóng |
|---|---|---|
| Nhầm co-occurrence thành action | Nghiêm trọng; hạn chế phương pháp | Action evidence contract; hard negatives đảo vai trò, E08 |
| Gán người nói chỉ từ ASR | Nghiêm trọng; capability chưa kiểm chứng | Cần audio-visual binding; thiếu thì unknown, E08 |
| Sampler bỏ text/action ngắn | Cao; giới hạn sparse sampling | Burst, text probes, refinement; visibility recall, E02/E04 |
| Mọi retrieval branch đều bỏ sót | Cao | Đo candidate coverage; refine không cứu video chưa tìm thấy, E06 |
| Sai PTS/source/frame mapping | Nghiêm trọng | Exact ledger/maps; không suy frame bằng FPS danh nghĩa, E01 |
| Gọi pseudo-label/pending review là GT | Nghiêm trọng; có pending packets trong audit | Provenance và evaluator isolation, E12 |
| AVC gallery chọn bằng gold được dùng như heldout | Nghiêm trọng; đã quan sát | Gắn diagnostic; corpus mới tách nguồn, E13 |
| Charset PP-OCRv6 thiếu một số chữ có dấu VI | Cao; đã probe dictionary | Kiểm tra charset và inference; không thay dictionary đơn thuần, E04 |
| VieSVTR hạn chế NC, nguy cơ trùng benchmark English | Cao; model card có thông tin liên quan | Kiểm tra quyền và tập VI/EN độc lập, E13/E04 |
| Cache cũ khác model/processor revision | Cao | So hashes, tạo feature namespace mới |
| Nạp mọi model đồng thời vào 6 GB | Cao; nguy cơ không khả thi | Một heavy GPU worker, adaptive batch, admission control, E10 |
| Peak RAM khi train/load index | Cao | Shard khoảng 1M; bounded training sample; tránh full FP32 replicas, E05 |
| Raw media cần 9–36 TB theo bitrate giả định | Cao; storage chưa biết | Xác định media storage trước ingest 10.000 giờ; proxy theo nhu cầu |
| Cold VLM vượt latency allocation | Cao | Bỏ verifier chưa warm/không đủ budget; partial results, E10 |
| Research model thiếu code/weights chạy được | Trung bình; một số chưa xác minh | Nhánh nghiên cứu, không dependency bắt buộc |
| Quyền dữ liệu/media chưa rõ | Cao | Quarantine; kiểm tra riêng code, weights, annotation, media, E13 |
| Backup cùng ổ vật lý | Cao; đã quan sát | Copy sang destination độc lập và thử restore trước xóa, E14 |
| Git history khác chưa tìm thấy | Trung bình | Ghi nhận ancestor repo chưa có commit; kiểm tra remote/backup khác |
| Overfit benchmark nhỏ | Cao | Holdout tách nguồn, khóa ablation plan, CI theo cụm video |
| Exact OCR bị lẫn với tìm không dấu | Trung bình | Recheck NFC gốc; folded index chỉ mở rộng recall |
| Query phức tạp làm bùng nổ trạng thái | Trung bình | Bounded beam, warning, original fallback; timeout không nghĩa unsatisfiable |
| Browser seek thiếu chính xác | Trung bình | Exact-frame endpoint, proxy/source maps, đo sai số pause |
| Mơ hồ như “xanh”, “phía sau” | Cao | Giữ cách hiểu thay thế/residuals; không hard-filter giả thuyết chưa chắc |

## Giả định và thông tin còn thiếu

Người dùng xác nhận **RTX 3060 6 GB VRAM và RAM 32 GB** cho máy đích. CPU, TGP, SSD và OS chưa biết. Competition, quyền corpus, network rules, submission format, bitrate/phân bố video, storage budget và concurrency thực tế cũng chưa chốt.

Khởi đầu một query active, tiếng Việt/English, không cần paid API cho baseline. Không mặc định có RTX 5060 Ti, Kaggle T4 hoặc cloud GPU; chỉ dùng nếu sau này thực sự có tài nguyên phù hợp.

Công thức dung lượng là phép tính xác định. Mật độ sampling thực tế, text volume và throughput chưa đo. Chọn B là phán đoán kỹ thuật tạm thời; không tuyên bố model đã giải quyết compositional retrieval.

## Quy trình thay đổi quyết định

Mỗi thay đổi ghi ngày, D-ID, bằng chứng/run ID, phương án đã so sánh, config được chọn, ảnh hưởng tương thích, migration và rollback. Model/index mới cần snapshot revision; thay đổi ngữ nghĩa QueryGraph/API cần schema version.

Không sửa metrics, nhãn hoặc source receipts cũ để hợp quyết định mới. Khi quyền dataset bị thu hồi, tombstone serving và review retention; không tự xóa nguồn ngoài phạm vi cho phép.

Xử lý câu hỏi mở tại milestone: nguồn/backup M0; timeline M1; model precision M2; charset/routing M3; union M4; parser M5; relations M6; official export M7 khi có format; storage/SLO M8; fine-tuning M9 có điều kiện.

