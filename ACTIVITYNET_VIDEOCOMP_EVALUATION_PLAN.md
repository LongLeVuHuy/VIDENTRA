# Kế hoạch đánh giá VIDENTRA bằng ActivityNet Captions và VideoComp

**Ngày lập 24/11/2026 · Trạng thái đề xuất**

Mục tiêu là đo xem VIDENTRA có tìm đúng video, định vị đủ từng sự kiện và nhận ra mô tả sai thứ tự hay không. Máy cá nhân chạy các bước CPU và giữ ground truth. Modal L4 xử lý suy luận nhẹ, L40S xử lý embedding nặng. Chọn GPU bằng số đo trên cùng một lô tune. Giữ một phép đo riêng cho cấu hình online 6 GiB phục vụ mục tiêu RTX 3060 dài hạn.

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

## 3. Phân việc giữa máy cá nhân và Modal

| Nơi chạy | Tác vụ | Khi nào chuyển tải |
|---|---|---|
| **CPU máy cá nhân · chuẩn bị dữ liệu** | Lập manifest, hash và chia nhóm nguồn. Decode bằng PyAV theo presentation order, ghi ordinal zero-based, PTS/time base và mốc gốc. Chọn keyframe, đóng gói frame, clip hoặc audio theo shard. OCR ít frame có thể chạy local. | Gửi lên Modal đầu vào mà model cần, kèm ID và ánh xạ thời gian. Với VideoPrism cần giữ chuỗi frame theo đúng thứ tự và cách lấy mẫu của encoder. |
| **CPU máy cá nhân · tìm kiếm và đánh giá** | Dựng FAISS CPU, SQLite/FTS5, chạy BM25, fusion, DP/Viterbi, scoring VideoComp từ điểm đã cache, evaluator và bootstrap. Dùng rules và text encoder nhẹ cho query lẻ. Ghi index theo shard hoặc dùng memory map để phù hợp RAM 32 GB. | Khi cần text embedding từ backbone lớn, gửi batch văn bản sang đúng text tower trên Modal rồi dùng lại vector local cho mọi lần sweep. |
| **Modal L4 · suy luận nhẹ** | SigLIP2-Base, E5 theo batch, Whisper ASR, OCR có backend GPU khi số frame lớn. Qwen2.5-VL-3B NF4 chấm tuần tự top-3 trên storyboard ứng viên. Nếu dùng Qwen tách câu, chạy ở pha riêng. | Bắt đầu một container, VLM batch 1. Dùng L4 cho cả pilot và online nếu đạt chất lượng, thời gian và chi phí đã đo. |
| **Modal L40S · suy luận nặng** | Embedding nhiều frame/clip cho PE-Core-bigG, DFN5B-CLIP-H, VideoPrism và các backbone lớn còn lại. Nạp từng backbone, gom các video có kích thước đầu vào gần nhau. | So L4 với L40S trên cùng shard và cùng precision. Chọn theo chi phí/video hoàn tất ở throughput cần thiết. Số lượng video lớn tự nó chưa đủ để nâng hạng GPU. |
| **Modal A100-80GB · trường hợp vượt bộ nhớ** | Thử nghiệm có đầu vào lớn thực sự, ví dụ clip dài hoặc VLM đối chứng cần nhiều visual token. | Chỉ thử khi L40S vẫn OOM ở cấu hình đầu vào cần giữ, hoặc pilot cho thấy lợi thế chi phí rõ ràng. Ghi đây là cấu hình cloud riêng nếu thay đổi model, precision hoặc số frame. |

Các mã GPU trong bảng là lựa chọn cho Modal Function khi triển khai. Worker cũ của dự án cung cấp logic trích xuất và quy ước PTS có thể tái sử dụng. Bước tích hợp mới là chuyển input/output của worker sang manifest và Modal Volume. [GPU trên Modal](https://modal.com/docs/guide/gpu) · [Worker đa phương thức hiện có](scripts/kaggle_multimodal_offline_worker.py) · [Worker VideoPrism hiện có](scripts/kaggle_videoprism_worker.py)

## 4. Trình tự triển khai và kiểm soát chi phí

1. **Chuẩn bị local.** Xác minh quyền xử lý media trên Modal. Khóa manifest split, gallery và query. Hoàn thành các mục P0 ở mục 5 trước lần đánh giá chất lượng. Ground truth và evaluator ở máy cá nhân. Với VideoComp, khâu chuẩn bị cấp đúng clip đầu vào theo giao thức và hai mô tả dưới ID ẩn; worker chỉ trả điểm từng mô tả.
2. **Pilot trên tune.** Chọn một shard cố định có video ngắn/dài. Với mỗi tác vụ cần so GPU, dùng cùng hash input, model/processor revision, precision và ngân sách frame/token. Kiểm tra số vector, ánh xạ PTS và độ lệch điểm. Ghi GPU thực cấp, peak VRAM/RAM, thời gian nạp, video/giờ hoặc cặp/giây, thời gian truyền và chi phí/100 video hoặc /1.000 cặp. Dùng đơn giá tại thời điểm chạy và giây tính phí thực tế. [Bảng giá Modal](https://modal.com/pricing)
3. **Tạo và nhận artifact.** Upload mỗi shard một lần. Tách Function L4 và L40S để nạp model một lần cho nhiều input. Dùng `min_containers=0`, pilot bắt đầu với `max_containers=1`, chỉ tăng số container sau khi đo throughput. Lưu vector, evidence hoặc điểm VLM theo run/model/shard trên Modal Volume, mỗi worker ghi vùng riêng. Chỉ công nhận shard hoàn tất khi số dòng và checksum khớp manifest. Khi retry, tái dùng shard đã xác minh. Máy cá nhân nhận artifact rồi dựng index. Ước tính toàn bộ chi phí từ pilot trước khi mở rộng. [Modal Volumes](https://modal.com/docs/guide/volumes) · [Autoscaling](https://modal.com/docs/guide/scale)
4. **Tune rồi valid-confirm.** Đo lần lượt nguyên câu, căn chỉnh mệnh đề độc lập, DP/Viterbi và DP/Viterbi + VLM. Hai phương án cuối dùng cùng shortlist, query gốc và profile. Chọn cấu hình trên tune rồi chạy confirm một lần. Đề xuất giữ VLM mặc định khi cận dưới CI 95% của chênh lệch Full-chain Hit@1 so với DP/Viterbi lớn hơn 0 và chi phí/p95 đáp ứng mức đã chốt sau pilot. Bootstrap theo cặp trên `source_group_id`. Nếu không đạt, khóa baseline.
5. **Test nội bộ.** Sau khi khóa mới xử lý split test. Lưu dự đoán bất biến trước khi evaluator local mở nhãn. Báo số đúng/tổng, nhóm độ dài, từng miền/loại lỗi VideoComp, coverage, OOM và fallback. Đo p95 end-to-end từ máy cá nhân, tách lượt lạnh/ấm và thời gian truyền, chờ worker, nạp model, inference. Ghi rõ trường hợp replay từ cache khi đo.

**Hai cấu hình online.** Với cấu hình cloud, cấp VRAM theo GPU thắng pilot. Với cấu hình mục tiêu **6 GiB**, giữ FAISS và text encoder trên CPU, chỉ chạy VLM 3B NF4 trên một GPU Modal, top-3 tuần tự, batch 1. Nếu text encoder cần GPU, encode xong trong worker riêng, trả vector về local rồi mới rerank. Ghi peak allocated/reserved của PyTorch cùng mức dùng toàn GPU lấy mẫu bằng NVML. Khi OOM, thử lại một lần với ít visual token hơn; nếu vẫn lỗi, trả nguyên kết quả DP/Viterbi và ghi fallback. Đánh dấu đạt ngân sách khi mức dùng toàn thiết bị quan sát được ≤ 6.144 MiB và phiên confirm hoàn tất với 0 OOM. Phép đo trên Modal phải mang tên GPU thực tế; latency và khả năng chạy trên RTX 3060 cần đo khi có máy. [CUDA memory trong PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html)

## 5. Những điểm nên nâng cấp trong VIDENTRA

| Ưu tiên | Phát hiện trong mã hiện tại | Thay đổi đề xuất và cách nghiệm thu |
|---|---|---|
| **P0 · Thứ hạng sau VLM** | VLM sắp lại danh sách moment theo video, nhưng `_build_ranked_videos` xếp lại theo điểm trước rerank. | Truyền thứ tự hoặc điểm cuối cùng tới evaluator. Ca A/B trong đó VLM đưa B lên đầu phải trả B ở đầu cả `ranked_moments` và `ranked_videos`. [Runtime retrieval](src/videntra/retrieval/unified_pipeline.py) |
| **P0 · Chuỗi ≥4 sự kiện** | Nhánh rule lấy ba mệnh đề đầu. Viterbi đã có `selected_indices`, nhưng caller chỉ lấy hai đầu để tạo một khoảng chung. | Giữ ID cho toàn bộ mệnh đề và xuất một khoảng riêng có bằng chứng PTS cho từng sự kiện. Dùng điểm neo của Viterbi để tìm biên; thiếu bằng chứng thì trả trạng thái thiếu. Test chuỗi bốn bước, đảo thứ tự và hai sự kiện đồng thời. [Parser](src/videntra/query/rules.py) · [Viterbi](src/videntra/retrieval/alignment.py) · [Caller](src/videntra/retrieval/unified_pipeline.py) |
| **P0 · Evaluator toàn chuỗi** | `Moment Hit@1` trong runner hiện nhận hit khi khớp một cửa sổ bất kỳ. Loader QVHighlights cũ còn lọc theo target có trong index. | Cài metric ở mục 2 trên manifest đã khóa; giữ cách chấm cũ cho benchmark cũ. Thử ba tình huống: đúng 2/3 bước, sai video dù đúng timestamp, target mất khỏi index. Cả ba phải là miss với mẫu số giữ nguyên. [Runner](scripts/run_validation_benchmark.py) |
| **P1 · Cache điểm khi chạy theo shard** | Khi đã có cache cho query, `score_query_candidates` trả ngay phần giao với shortlist. Ca CPU có cache A, yêu cầu A/B chỉ trả A. | Tra theo từng cặp query–ứng viên, kèm revision model/processor, prompt và hash storyboard. Chấm bù B trên Modal, ghép kết quả theo ID, giữ A. Kiểm tra rerun cùng shard không tính phí inference thêm cho cặp đã hoàn tất. [Cache VLM](src/videntra/ranking/vlm_reranker.py) |
| **P1 · Storyboard và OOM** | Hàm storyboard giữ mọi frame đã decode trong RAM để chọn tám ảnh; nhánh OOM thử lại từng ảnh dù batch có thể đã là 1. | Chọn ordinal qua ledger, decode có giới hạn bộ nhớ quanh vùng ứng viên cho từng sự kiện. Kiểm tra ordinal/PTS, peak RAM theo độ dài video và fallback khi một ảnh đã vượt VRAM. [Storyboard và reranker](src/videntra/ranking/vlm_reranker.py) |

Ba mục P0 quyết định liệu số đo có phản ánh khả năng tìm **đủ và đúng thứ tự sự kiện**. Hai mục P1 nên hoàn thành trước khi mở rộng pilot lên toàn corpus, vì chúng tác động trực tiếp tới RAM và số lượt inference phải trả phí.
