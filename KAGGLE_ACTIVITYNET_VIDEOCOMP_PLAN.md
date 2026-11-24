# Kế hoạch đánh giá VIDENTRA trên Kaggle với ActivityNet Captions và VideoComp

**Ngày lập 24/11/2026 · Trạng thái đề xuất**

Mục tiêu là đo xem VIDENTRA có tìm đúng video, định vị đủ các sự kiện trong một truy vấn dài và phân biệt được hai mô tả chỉ khác thứ tự hành động hay không. Tất cả các bước từ tạo embedding đến chấm test đều chạy được theo kế hoạch trên Kaggle. Máy cá nhân dùng để quản lý mã nguồn và đọc kết quả. Ngân sách **6 GiB (6.144 MiB) trên một GPU** áp dụng cho đường xử lý truy vấn online. Số đo sẽ gắn với đúng loại GPU Kaggle cấp cho phiên chạy.

## 1. Bộ dữ liệu và cách chia tune, valid, test

**ActivityNet Captions** cung cấp câu mô tả kèm khoảng thời gian. Mỗi đoạn văn có ít nhất hai câu được định vị là một truy vấn đa sự kiện cho bài tìm kiếm toàn kho. **VideoComp** cung cấp cặp mô tả đúng/sai cho cùng video hoặc clip, gồm hai miền ActivityNet-Comp và YouCook2-Comp. ActivityNet-Comp dùng lại video của ActivityNet Captions. [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) · [VideoComp](https://github.com/google-deepmind/video_comp)

| Tập nội bộ | Tỷ lệ nhóm video nguồn | Pool công khai | Sử dụng |
|---|---:|---|---|
| Tune | 40% | Train | Chọn parser, căn chỉnh, tham số rerank và ngân sách hình ảnh. |
| Valid-confirm | 40% | Train, tách nguồn với tune | Chạy một lần để so cấu hình đã chọn với baseline và khóa Champion. |
| Test | 20% | Validation chưa dùng khi phát triển | Chấm một lần sau khi khóa. Ghi rõ đây là test nội bộ khi so với bài báo. |

1. Lập danh mục media trước khi chạy retrieval. Tạo một `source_group_id` xuyên cả ba nguồn annotation. Gộp video cùng ID gốc, clip cắt từ cùng nguồn, hai bộ mô tả `val_1` và `val_2`, caption viết lại và mọi cặp positive/negative. Kiểm tra bản sao bằng hash file, rà bản đăng lại bằng dấu vân tay hình/âm thanh. Nhóm có lịch sử train/tune chồng với pool test hoặc chưa xác định được nguồn được ghi riêng để xử lý trước khi khóa.
2. Chia theo nhóm trong từng miền ActivityNet và YouCook2. Cố định seed `20261124`, sắp nhóm theo hash của seed và ID nguồn. Ví dụ, với 2.000 nhóm đủ điều kiện và đủ quota ở hai pool, lấy 800/800/400. Nếu thiếu, giảm quota theo tỷ lệ và ghi số video, số query thực tế. Mỗi nhóm ActivityNet dùng chung một split cho cả bản Captions và bản Comp.
3. Khóa checksum cho danh sách nguồn, gallery, query và phiên bản annotation. Kiểm tra giao từng cặp tune–valid, tune–test, valid–test đều rỗng. Mỗi query ActivityNet tìm trên toàn gallery của split. Ground truth chỉ evaluator đọc. Video bị thiếu khỏi index sau khi khóa hoặc query không có dự đoán được tính miss, đồng thời báo coverage.

## 2. Chấm đúng cả chuỗi

**Chỉ số chính là Full-chain Hit@1 tại tIoU ≥ 0,5 trên ActivityNet Captions.** Với mỗi truy vấn, hệ thống trả video cùng một khoảng dự đoán `[start, end)` cho từng mệnh đề, giữ ID mệnh đề từ đầu đến cuối. Khoảng thời gian lấy từ PTS và time base của video. Các câu và timestamp tham chiếu của cùng một bản annotation luôn đi chung.

Một truy vấn được **1 điểm** khi video hạng 1 đúng, **tất cả** các sự kiện có khoảng dự đoán khớp với sự kiện tương ứng và mọi quan hệ trước/sau được gắn nhãn hoặc xác định rõ đều đúng. Với mỗi sự kiện, tIoU bằng thời lượng phần giao chia thời lượng phần hợp giữa khoảng dự đoán và khoảng tham chiếu. Cho phép chồng lấn ở các sự kiện diễn ra đồng thời. Ví dụ, chuỗi *mở hộp → lấy cốc → rót nước* có ba tIoU 0,82 / 0,74 / 0,31 sẽ nhận **0 điểm trọn chuỗi**. Tỷ lệ cuối bằng số truy vấn đạt chia cho toàn bộ truy vấn đủ nhãn đã khóa.

| Phép đo bổ sung | Câu hỏi cần trả lời |
|---|---|
| Video Recall@20 của Stage-1 | Video đúng có vào danh sách ứng viên không? |
| Full-chain Hit@1 khi cho sẵn video | Parser và localizer định vị được bao nhiêu chuỗi khi đã biết video? |
| Full-chain Hit@5 trên toàn gallery | Trong năm kết quả đầu có ít nhất một video với một chuỗi hoàn chỉnh không? Chấm từng kết quả độc lập. |
| Tỷ lệ sự kiện đạt tIoU ≥ 0,5, tách nhóm 2, 3 và ≥4 sự kiện | Chất lượng giảm ở độ dài nào và sự kiện nào thường bị bỏ sót? |

**Với VideoComp**, chấm riêng binary accuracy của ba loại *temporal reordering*, *action replacement* và *segment mismatch* trên từng miền. Với mỗi cặp, dùng cùng hàm tính điểm video–văn bản cho hai mô tả, yêu cầu điểm mô tả đúng cao hơn và báo số lần hòa. Riêng *segment mismatch* dùng đúng đoạn `query_video/start_time` đến `query_video/end_time` mà benchmark cung cấp làm đầu vào. Cửa sổ này chỉ thuộc bài so cặp VideoComp. Khi báo metric `all` của tác giả, lấy tích ba accuracy ở thang 0–1. [Giao thức chấm VideoComp](https://github.com/google-deepmind/video_comp/blob/main/README.md)

## 3. Trình tự chạy trên Kaggle

| Bước | Việc cần làm | Đầu ra để kiểm tra hoặc tiếp tục ở phiên sau |
|---|---|---|
| 1. Chuẩn bị | Kiểm tra media, quyền xử lý trên Kaggle, split và schema. Kiểm tra bằng ví dụ tổng hợp rằng thứ hạng sau VLM đến đúng evaluator và mỗi mệnh đề có một khoảng dự đoán. | Manifest có checksum, thống kê coverage, kết quả kiểm tra evaluator. |
| 2. Index offline | Decode theo presentation order, lưu ordinal zero-based và PTS. Tính embedding, OCR/ASR theo từng lô video trên Kaggle. Xử lý tune trước, mở valid và test theo thứ tự khóa cấu hình. | Index từng split, ánh xạ PTS, model/processor revision và checksum từng lô. |
| 3. Tune | So bốn cấu hình gồm truy vấn nguyên câu, căn chỉnh mệnh đề độc lập, DP/Viterbi và DP/Viterbi + VLM. Hai cấu hình cuối dùng cùng danh sách ứng viên. Giữ nguyên truy vấn gốc cho mọi cấu hình. | Thứ hạng, khoảng dự đoán từng sự kiện, điểm theo loại lỗi, chi phí của mỗi cấu hình. |
| 4. Valid-confirm | So đúng cấu hình VLM đã chọn với baseline DP/Viterbi trên cùng query. Đo chênh lệch Full-chain Hit@1 và khoảng tin cậy 95% bằng paired bootstrap theo video nguồn. | Cấu hình được khóa, hash model/index, ngân sách VRAM, quy tắc fallback. |
| 5. Test nội bộ | Chạy cấu hình đã khóa và baseline đã đăng ký. Ghi dự đoán trước khi evaluator mở nhãn. Báo riêng kết quả ActivityNet, ActivityNet-Comp và YouCook2-Comp. | Số đúng/tổng, metric theo độ dài hoặc loại lỗi, khoảng tin cậy, số OOM và fallback. |

Mỗi phiên ghi commit, seed, GPU thực được cấp, phiên bản CUDA/PyTorch, model và revision của input Kaggle. Ghi output theo run ID và lưu phiên bản Kaggle Notebook Output hoặc Kaggle Dataset trước khi kết thúc phiên. Khi tiếp tục, kiểm checksum rồi chạy từ lô còn thiếu. Giữ snapshot lịch sử và tránh đưa media hoặc weights vào Git. [Kaggle Notebook outputs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

## 4. Kiểm tra ngân sách 6 GiB trên GPU Kaggle

- **Cấu hình đầu tiên.** Dùng một GPU cho phép đo online, kể cả khi phiên được cấp hai GPU. Tìm kiếm FAISS/SQLite và căn chỉnh trên CPU. Chạy text encoder trên CPU, giữ text embedding trong RAM. Thử Qwen2.5-VL-3B NF4 với `batch_size=1`, top-3 ứng viên chấm tuần tự. Khóa số visual token và lấy frame từ vùng ứng viên do hệ thống tìm. Decode frame theo PTS với bộ đệm giới hạn để tránh phình RAM trên video dài.
- **Nếu mã hóa văn bản trên CPU quá chậm.** Thử phương án encode trên GPU trong một worker riêng, chuyển embedding về CPU rồi kết thúc worker trước khi nạp VLM. Nếu bật LLM tách mệnh đề, cũng tách nó sang pha riêng. Đo p95 từ lúc nhận query tới kết quả, bao gồm cả thời gian đổi model. Benchmark offline có thể cache embedding theo query và revision, còn phép đo online phải ghi rõ cache lạnh hay ấm.
- **Cách đo.** Đặt lại bộ đếm trước mỗi lượt, ghi peak allocated/reserved của PyTorch và lấy mẫu bộ nhớ toàn thiết bị qua NVML hoặc `nvidia-smi` suốt lúc nạp model, encode và rerank. Ghi cả mức dùng ban đầu, chu kỳ lấy mẫu, RAM, p95 lạnh/ấm và số lượt fallback. Có thể đặt trần allocator PyTorch thấp hơn 6.144 MiB để chừa chỗ cho CUDA context và thư viện khác. Trần allocator chỉ kiểm soát các cấp phát qua PyTorch. Đánh giá ngân sách bằng cả số đo thiết bị, ghi rõ nếu thiếu phép đo này. [Quản lý CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html) · [Giới hạn allocator theo process](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
- **Khi OOM.** Retry một lần với ngân sách pixel/visual token nhỏ hơn đã chọn trên tune, ưu tiên giữ bằng chứng cho từng sự kiện. Nếu vẫn lỗi, trả nguyên kết quả DP/Viterbi và tính query đó vào mẫu số. Giảm top-3 xuống top-2 chủ yếu giảm số lần gọi khi đã dùng batch 1. Biện pháp giảm peak cho mỗi lần gọi là giảm visual token hoặc giải phóng model còn giữ GPU.
- **Quyết định trước test.** Đề xuất chỉ chọn VLM làm mặc định khi trên valid-confirm nó cải thiện Full-chain Hit@1 so với DP/Viterbi với cận dưới CI 95% lớn hơn 0, mức dùng toàn GPU quan sát được nằm trong 6.144 MiB và bài chạy xác nhận hoàn tất không OOM. Nếu chưa đạt, dùng DP/Viterbi làm mặc định và ghi kết quả VLM như một cấu hình thử nghiệm. Báo tên GPU Kaggle cạnh số đo; khả năng chạy thực tế trên RTX 3060 6 GB sẽ được xác minh khi có thiết bị.
