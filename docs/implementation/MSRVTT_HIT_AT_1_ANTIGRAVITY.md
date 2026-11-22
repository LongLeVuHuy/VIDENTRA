# Kế hoạch cho Antigravity nâng Hit@1 trên MSR‑VTT 1K‑A

Ngày 22/11/2026. Mục tiêu là **Video Hit@1 > 50%**, tức ít nhất **501/1.000 query** Text → Video đúng hạng nhất trên test chuẩn. Chạy toàn bộ ma trận trên validation, chọn một cấu hình bằng kết quả validation rồi khóa trước khi chấm test. Ưu tiên **VideoPrism‑LvT‑B độc lập** và **rerank Top‑20**. Phạm vi benchmark là MSR‑VTT. Tiếp tục chạy ma trận validation kể cả khi baseline VideoPrism đã qua 50%.

> **Lệnh giao việc:** Antigravity, triển khai kế hoạch này trong repo VIDENTRA. Đọc AGENTS.md, audit MSR‑VTT ngày 20/11 trong PROGRESS.md và bảng công việc bên dưới. Tái sử dụng worker, cache và evaluator có provenance hợp lệ. Đưa các job nặng lên Kaggle. Chạy từng biến thể validation, ghi kết quả hoặc blocker cụ thể cho từng dòng và khóa cấu hình tốt nhất trước khi chấm test 1K‑A. Sau khi xác minh snapshot MSR‑VTT, lập manifest dọn dữ liệu ngoài phạm vi, sao lưu và trình duyệt các đường dẫn xóa.

## 1. Khóa protocol so sánh

| Phần | Video | Query và gallery |
| --- | ---: | --- |
| Validation tune | 2.000 từ phần MSR‑VTT ngoài 1K‑A | Hai gallery cố định, mỗi gallery 1.000 video và 1.000 query |
| Validation confirm | 2.000 từ phần MSR‑VTT ngoài 1K‑A | Hai gallery cố định khác, mỗi gallery 1.000 video và 1.000 query |
| Test công bố | 1.000 video chuẩn 1K‑A | Một gallery 1.000 video và đúng 1.000 câu gốc của JSFusion 1K‑A |
| Train khi cần | Các video đủ điều kiện còn lại trong phần 9K, tối đa 5.000 | Dùng caption của chính train cho LTR hoặc joint reranker |

Tỷ lệ **80/20** áp dụng cho 5.000 video validation và test. Train là phần dữ liệu riêng để fit biến thể có học. Mỗi query có một video đích nên R@1 và Video Hit@1 bằng nhau. Dùng cùng gallery, query manifest, mapping đích và evaluator cho cả hai dự án. Nếu dự án đối chiếu từng công bố số liệu theo protocol khác, ghi rõ protocol của số liệu đó và ưu tiên chạy lại trên bộ chung này. [MSR‑VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) · [Frozen in Time](https://github.com/m-bain/frozen-in-time) · [Danh sách video 1K‑A](https://raw.githubusercontent.com/albanie/collaborative-experts/master/misc/datasets/msrvtt/val_list_jsfusion.txt) · [CSV query JSFusion](https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_JSFUSION_test.csv)

1. Pin revision/hash của annotation và danh sách video. Khóa 1K‑A trước. Với phần 9K, nhóm theo video nguồn, exact duplicate và near-duplicate đã xác minh. Loại các nhóm đụng test khỏi phần phát triển, giữ nguyên test chuẩn. Dùng seed `20261122` để chọn và phân nhóm vào bốn gallery validation theo nguồn, cân bằng category khi có metadata. Chọn một caption gốc cho mỗi clip theo quy tắc xác định từ seed và annotation ID. Freeze cách chọn, thứ tự và hashes trước khi chạy model. Kiểm tra không giao nhóm nguồn giữa train, tune, confirm và test.
2. Kiểm kê cả media, annotation và ASR/OCR cache theo video ID. Ngày lập kế hoạch, `data/raw/msrvtt/` có 1.000 file video và audit trước đã xác nhận 1.000 ID MSR‑VTT trong DB là gallery 1K‑A. Cần tìm đủ media của tập ngoài 1K‑A từ nguồn có quyền xử lý. Nếu thiếu, tiếp tục smoke kỹ thuật và xây cache 1K‑A từ pixel/audio khi được phép, ghi số coverage thiếu và giữ test score đóng cho đến khi có cấu hình đã khóa. Không lấy 809 query dev lịch sử hoặc 191 query held-out lịch sử để bù validation.
3. Tạo manifest/index riêng cho từng gallery trước bước sinh candidates. Dùng thứ hạng từ gallery đã khóa ngay từ đầu, với thứ tự tie-break cố định theo video ID. Tách payload inference gồm query ID/text gốc, gallery manifest và artifacts khỏi qrels. Chỉ evaluator chấm qrels, còn biến thể có học chỉ dùng nhãn train để fit. Ẩn mapping test khỏi worker Kaggle và runner retrieval. Query test chỉ được encode/chấm sau khi khóa cấu hình. `videntra_ltr_v1.json`, query cache cũ và threshold đã tune bằng nhãn thuộc 1K‑A chỉ có giá trị chẩn đoán lịch sử. Nếu dùng lại cấu trúc A5, ghi lịch sử tiếp xúc 1K‑A và fit lại trọng số/ngưỡng từ train/tune mới.

Mốc nội bộ `1d90b8f9` có 84/191 (43,98%) với gallery hỗn hợp và 44 lỗi có đáp án ở hạng 2–5. Đây là dữ kiện để đặt thứ tự thử, còn baseline 1K‑A cần đo theo protocol mới. [Audit nội bộ](../PROGRESS.md)

## 2. Ma trận thử nghiệm bắt buộc trên validation

Chạy mọi dòng đủ đầu vào trên hai gallery tune. Khi có nhiều tham số trong một dòng, khóa cấu hình tốt nhất của dòng đó trên tune và chấm trên hai gallery confirm. Giữ câu truy vấn gốc cho mọi biến thể. Khi thử reranker, cache đúng 20 ứng viên unique từ mỗi bộ sinh và giữ nguyên membership của từng pool. Khi thử mở rộng truy hồi, tạo pool mới, giữ K=20 và đo cả mức tăng coverage lẫn Hit@1.

| ID | Cấu hình | Phép đối chứng |
| --- | --- | --- |
| B0 | SigLIP2 độc lập với cosine và mean pooling theo video | Baseline frame. Dùng cosine sau chuẩn hóa L2 và sampling đã khóa. |
| B1 | **VideoPrism‑LvT‑B public độc lập**, một embedding toàn video và một embedding cho câu gốc | Xếp mọi video trong gallery bằng cosine. Đo trước, dùng làm baseline chính. |
| B2 | VideoPrism nhiều cửa sổ thời gian, so whole-only với whole + window pooling | Dùng cùng encoder và kiểm độ phủ cửa sổ trên clip ngắn lẫn dài. Tune trọng số cửa sổ và báo Recall@20. |
| B3 | Fusion VideoPrism + SigLIP2 | So weighted RRF với score fusion sau chuẩn hóa. Tạo shortlist bằng điểm trước rerank. |
| B4 | A5 trên gallery MSR‑VTT mới | Ma trận consensus bật/tắt × LTR bật/tắt. LTR fit mới từ train, parser rules giữ cố định. |
| R0 | Replay `alpha=0` trên Top‑20 từ B1 và một pool đối chứng đã chọn trên tune | Thứ tự phải trùng first-stage, ghi hash danh sách theo query. Đây là mốc cho R1–R7. |
| R1 | Rerank cosine toàn video/cửa sổ + score ưu tiên của first-stage | So RRF và normalized-score blend trên cùng Top‑20, có `alpha=0` làm đối chứng. Với pool B1, whole-only là kiểm tra đối chứng vì đã dùng cùng điểm; window bổ sung tín hiệu mới. |
| R2 | Rerank theo frame có liên quan đến query | So max/mean Top‑2 cách nhau theo PTS và query-conditioned pooling kiểu X‑Pool. Chạy với frame features đã cache. |
| R3 | Joint reranker gọn theo VEDJE | Dùng cache feature theo frame, giữ score first-stage làm prior. Fit trên train với hard negatives đến từ retrieval; rà trường hợp nhiều video đều hợp caption. Ghi rõ phần nào tái lập được từ paper và phần nào là biến thể nội bộ. |
| R4 | Caption sinh offline từ hình ảnh/video | Chấm query–caption bằng sparse, dense và fusion trên cùng Top‑20. Chạy thêm một lượt đưa caption vào truy hồi độc lập để đo Recall@20. |
| R5 | OCR / ASR | Chạy vision-only, +OCR, +ASR, +OCR+ASR. So cả rerank pool cố định và union candidates với cùng K. Ghi rõ tín hiệu có dữ liệu, tín hiệu thiếu và intent của query. |
| R6 | Detector crops với/không Kalman tracking | Giữ cùng detector, frame, box thresholds và Top‑20. So global, global + crop, global + crop + track. Báo toàn tập và lát cắt chuyển động/quan hệ định nghĩa từ query trước khi chấm. |
| R7 | Kết hợp các tín hiệu có ích | Dùng scorer thắng ở R1–R3 làm gốc. Thử thêm caption, OCR, ASR, tracking độc lập rồi chạy tổ hợp tích lũy của các tín hiệu đi qua tune. So lại với B1 và từng ablation. |

Với B1, dùng checkpoint public và chế độ tiền xử lý tương ứng. Google công bố **50,1%** zero-shot cho VideoPrism‑LvT‑B public trên MSRVTT‑1K. VEDJE công bố **54,6%** khi rerank 20 ứng viên từ VideoPrism. Đây là mốc tham chiếu và cần đo lại trên artifacts của VIDENTRA. [VideoPrism](https://github.com/google-deepmind/videoprism#zero-shot-video-text-retrieval) · [VEDJE, preprint](https://arxiv.org/html/2610.11850v1) · [X‑Pool](https://openaccess.thecvf.com/content/CVPR2022/html/Gorti_X-Pool_Cross-Modal_Language-Video_Attention_for_Text-Video_Retrieval_CVPR_2022_paper.html) · [Cap4Video](https://arxiv.org/abs/2301.00184)

R6 dùng detector để quan sát, Kalman để dự đoán box và SORT hoặc ByteTrack để nối track. Tách box quan sát khỏi box dự đoán. Lấy frame đủ dày trong từng shot bằng presentation ordinal và PTS/time_base, ngắt track tại cut và reset khi gap vượt ngưỡng đã khóa trên tune. Track chỉ tác động score qua tín hiệu đo được như độ nhất quán object, hướng di chuyển, giao cắt hoặc liên tục qua occlusion ngắn. MSR‑VTT chấm retrieval theo caption nên kết quả của R6 báo tác động lên Hit@1. [SORT](https://arxiv.org/abs/1602.00763) · [ByteTrack](https://arxiv.org/abs/2110.06864)

## 3. Phân việc Kaggle và local

| Thứ tự | Notebook Kaggle riêng tư | Kết quả cần lấy về |
| --- | --- | --- |
| 1 | VideoPrism whole-video, sau đó window embeddings | Vector banks kèm video ID, frame ledger, config, model và tokenizer hash. Pilot 16 clip + 8 query tune trước khi chia shard toàn gallery. |
| 2 | Decode/sampling, SigLIP2 keyframes và OCR | Keyframe manifest theo PTS, frame embeddings và OCR spans với score, bounding box và trạng thái có/thiếu. Dùng chung decode cho các nhánh khi cùng sampling. |
| 3 | ASR từ audio của gallery | Transcript segments với timestamp, ngôn ngữ, số video có audio, không có audio hoặc lỗi. Giữ worker/query artifacts tách khỏi qrels. |
| 4 | Caption video offline, detector/crops/tracking và joint reranker | Cache caption, feature theo frame, detection/track theo shot và manifest. Chạy dense sampling/tracking và cross-modal inference theo ứng viên Top‑20. Với biến thể truy hồi theo caption, sinh caption cho toàn gallery tương ứng. Fit R3 trên train và checkpoint riêng. |

Refit LightGBM, fusion, dựng index, replay, paired evaluation và báo cáo có thể chạy local khi artifacts đã sẵn. Giữ job Kaggle riêng theo model/split để resume theo shard và tận dụng quota. Dùng media đã có quyền xử lý trên Kaggle hoặc nguồn có receipt cho phép; nếu mount chưa có quyền/coverage rõ, ghi blocker và tiếp tục các biến thể dùng dữ liệu hợp lệ. Chốt revision/hash/license của code, weights, tokenizer, media và outputs. Kiểm tra đủ expected IDs, không trùng, đúng dimensions, finite vectors, timeline hợp lệ và errors trước khi ingest. Dùng snapshot/output mới, giữ cache cũ làm đối chứng. Ghi thiết bị và thời gian thực đo cho mỗi job. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)

Tái sử dụng [VideoPrism worker](../../scripts/kaggle_videoprism_worker.py), [worker đa phương thức](../../scripts/kaggle_multimodal_offline_worker.py), [builder VideoPrism](../../scripts/build_videoprism_corpus_index.py), [capture](../../scripts/capture_msr_candidate_pools.py), [replay](../../scripts/run_clip_vp_replay.py), [paired comparator](../../scripts/compare_msr_runs.py) và [LTR CV](../../scripts/train_ltr_ranker_cv.py). Runner lịch sử có preset/đường dẫn mặc định cho corpus hỗn hợp. Khi triển khai, kiểm tra input contracts và làm adapter tối thiểu cho gallery/query manifest mới nếu thiếu. Mọi thay đổi code phục vụ benchmark nằm trong lượt thực thi của Antigravity.

## 4. Chọn cấu hình và điều kiện nghiệm thu

1. Với từng dòng, ghi `Hit@1 = hits/N`, R@5/10, MRR@10, Stage‑1 Recall@20, số query được cứu/bị mất so với B1, coverage và CI 95% ghép cặp theo nhóm nguồn. Theo dõi first-stage Recall@20 vì reranker chỉ đổi thứ tự trong shortlist. Ghi cả latency p50/p95 theo thiết bị thực, cache mode và peak VRAM/RAM cho các ứng viên cuối.
2. Chọn bằng **Hit@1 cao nhất trên confirm** sau khi khóa hyperparameters trên tune. Khi bằng số hit, ưu tiên MRR@10 cao hơn rồi p95 thấp hơn trong cùng môi trường. Fit biến thể có học chỉ trên train. Khóa một champion, B1 và baseline đối chiếu của dự án bên kia nếu có, cùng config hash, code commit, gallery/query hash và model receipt trước khi đọc điểm test.
3. Chạy một lượt test đã khóa cho champion và các baseline đã định trước. So với dự án khác bằng cùng evaluator. Công bố số hit, tỉ lệ và CI 95%. Mốc đạt là **≥501/1.000**. Nếu CI còn chứa 50%, ghi rõ điểm ước lượng vượt 50% và độ bất định. Nếu champion hụt mục tiêu, báo kết quả, phân tích coverage và lỗi đã định trước. Dùng validation cho vòng nghiên cứu tiếp theo và ghi rõ test đã được xem nếu chấm lại về sau.
4. Trả về bảng kết quả toàn ma trận validation, lệch hạng và các dòng bị chặn, manifest splits, hashes, links notebook, prediction files, biên bản chọn champion và bảng test cuối. Lưu run riêng có ngày/commit. Sau khi test hợp lệ, cập nhật benchmark hiện hành và README, giữ archive run cũ. Ghi PROGRESS, cập nhật PROJECT nếu đổi cấu trúc, cập nhật requirements nếu thêm gói và commit phần được phép đưa lên Git.

## 5. Dọn các dataset còn lại

1. Trước hết, chạy inventory ở chế độ dry-run cho các tài sản ngoài MSR‑VTT, gồm AVC, QVHighlights, TextVR, TVR và pilot ngoài phạm vi nếu có. Kiểm tra các thư mục `data/raw`, `data/quarantine`, `validation`, `test_queries` và các outputs dẫn xuất. Lần theo phụ thuộc để nhận diện snapshot/index hoặc cache hỗn hợp có chứa MSR‑VTT. Tạo snapshot MSR‑VTT riêng, kiểm tra coverage và chạy smoke trước khi xem xét dọn snapshot đang dùng.
2. Tạo manifest gồm path tuyệt đối đã resolve, dataset, loại tài sản, kích thước, SHA-256, nơi backup, phụ thuộc và cách khôi phục. Xác minh backup ở vị trí độc lập và thử khôi phục. Ưu tiên đề xuất xóa bản local đã sao lưu hoặc artifacts ngoài MSR‑VTT tái tạo được. Archive raw media, annotation, source receipt, human review và run lịch sử theo AGENTS.md. Backup giữ lại toàn bộ đường dẫn có trong manifest.
3. Trình người dùng danh sách đường dẫn cần xóa cùng số byte thu hồi và bằng chứng backup trước khi xóa. Yêu cầu dọn trong cuộc trò chuyện này xác định phạm vi đề xuất; quyền xóa gắn với manifest đã duyệt. Sau khi được duyệt, xử lý đúng các đường dẫn đã khóa trong PowerShell, kiểm tra junction/reparse point và đối chiếu biên bản dọn. Chạy lại smoke MSR‑VTT sau khi dọn. Để riêng tài sản remote, notebook Kaggle và dữ liệu dùng chung khỏi phạm vi dọn local. [Quy định dự án](../../AGENTS.md) · [Quy trình manifest và backup](../blueprint/01_OLD_PROJECT_RETIREMENT.md)
