# VIDENTRA — Sửa lỗi còn lại và kế hoạch tăng Hit@1 MSR‑VTT

Ngày 15/11/2026. Mốc tham chiếu là bản vá `3224676`, lần rà soát `6e8e03b` và benchmark `08626d79ef744e3080c2fb24c4761cad`. Các cấu hình, ngưỡng và tiêu chí ở phần tune là đề xuất để chạy thí nghiệm. [Benchmark](benchmark/benchmark_summary.json) · [Nhật ký kiểm chứng](docs/PROGRESS.md)

**Thứ tự đề xuất** — Sửa cách đo shortlist và xác minh mapping của patches trước khi chạy thử nghiệm. Đo một visual reranker nhẹ bằng SigLIP 2 hiện có trên Top‑10 của Stage 5. Chỉ mở rộng retrieval hoặc train head sau khi đo được bottleneck và phần tăng riêng của từng tín hiệu. Sửa temporal verifier song song vì tính đúng đắn của bằng chứng.

## 1. Baseline và phạm vi cải thiện

| MSR‑VTT, tìm trên toàn corpus | Hit@1 | Hit@5 | Hit@10 | Visual Recall@120 |
| --- | ---: | ---: | ---: | ---: |
| `dev_150`, 100 query MSR‑VTT | 47/100 = 47,00% | 59,00% | 64,00% | 79,00% |
| `dev_full_859`, 809 query MSR‑VTT | 312/809 = 38,57% | 60,07% | 67,24% | 84,18% |
| `heldout_test_201`, 191 query MSR‑VTT | 71/191 = 37,17% | 56,54% | 67,54% | 83,25% |

- Trên 191 query held-out, **37** target ở hạng 2–5, **21** ở hạng 6–10 và **62** ngoài Top‑10. Bước rerank Top‑10 có 58 ca đứng sai vị trí nhưng còn target trong danh sách. Mốc 50% tương ứng cần thêm ròng 25 Top‑1 đúng, tính cả ca đang đúng bị mất. Đây là kích thước khoảng cần cải thiện. [Benchmark](benchmark/benchmark_summary.json)
- Trên 809 query dev, **174** target ở hạng 2–5, **58** ở hạng 6–10 và **265** ngoài Top‑10. Trong 100 query dev nhỏ, A0/A1/A2/A3/A4/A5 lần lượt đúng **40/43/43/47/47/47** Top‑1. Taxonomy gán 69/120 ca sai held-out vào F4 (`FINE_GRAINED_VISUAL_CONFUSION`). Dùng taxonomy để lấy mẫu review. Mức đạt Top‑1 phải tính bằng nhãn và đánh giá ghép cặp. [Nhật ký đếm lại JSONL](docs/PROGRESS.md)
- Protocol hiện dùng một caption đầu tiên mỗi video và một target có nhãn mỗi query. Corpus gồm 1.324 video từ cả ba dataset, trong đó có 1.000 MSR‑VTT. Chỉ so trực tiếp với paper khi khớp query set, gallery, relevance và exposure của checkpoint với dữ liệu đánh giá. Tác giả MSR‑VTT cũng ghi nhận một số câu chú thích trùng hoặc rất giống nhau. [Chọn caption trong runner](scripts/run_validation_benchmark.py#L313-L345) · [MSR‑VTT supplementary material](https://www.microsoft.com/en-us/research/publication/msr-vtt-large-video-description-dataset-bridging-video-language-supplementary-material/)

## 2. Sửa lỗi và điều kiện để benchmark dùng được

### 2.1. Temporal verifier lấy thời gian từ evidence

**Tái hiện.** Hai OCR observations của cùng video cho `A` tại 6s và `B` tại 2s. Matcher trả `CONTRADICTED` cho `A before B` khi dùng refs, và trả `SUPPORTED` khi caller truyền khoảng A `(1, 2)`, B `(4, 5)`. Hai OCR refs khác ID cùng timestamp 5s cũng trả `SUPPORTED` cho `before`. Cả hai ca đã được tái hiện bằng fixture nhỏ. Đường benchmark mặc định không truyền `event_intervals`. [Chọn khoảng sự kiện](src/videntra/relations/matcher.py#L126-L183) · [Kiểm tra before/after](src/videntra/relations/temporal.py#L32-L86)

**Cách sửa.**

1. Tạo khoảng sự kiện trực tiếp từ evidence refs đã scoped cho đúng `video_id`, moment, predicate và time basis. Với OCR một điểm, dùng `[t, t]`. Với ASR dùng khoảng segment/word đã map về clip. Sự kiện visual cần có observation định thời. Nếu `timeline_revision` hoặc mapping cần cho việc so sánh mà chưa rõ, trả `UNKNOWN` cho quan hệ thời gian, đồng thời giữ đánh giá OCR/ASR riêng nếu đủ evidence.
2. Giữ tham số `event_intervals` để tương thích caller hiện có và dùng nó như cửa sổ lọc occurrence. Chọn refs có điểm/khoảng nằm trong cửa sổ đề xuất, sau đó tính start/end **từ các refs được chọn**. Nếu không có ref thời gian phù hợp thì trả `UNKNOWN` kèm lý do. Nếu một cụm xuất hiện nhiều lần, giữ các occurrence tách biệt theo track/segment và dùng anchor có căn cứ để chọn. Trường hợp nhiều cặp hợp lệ dẫn đến thứ tự khác nhau thì báo `PARTIAL` hoặc `UNKNOWN` theo mức grounding.
3. Khi hai point anchors trùng giờ hoặc sai số độ phân giải che mất thứ tự, trả `PARTIAL` kèm refs. Chuẩn hóa `after` đúng một lần. `max_gap_ms` và `min_gap_ms` dùng hiệu giữa hai biên evidence thật. Giữ semantics của hai interval chạm biên có thứ tự rõ trong fixture hiện có và test riêng cho point trùng giờ.
4. SE‑Viterbi chỉ đề xuất cửa sổ và anchors cho ranking. Giữ OCR/ASR curves gắn với clause và evidence tương ứng. Khi phrase không map được tới clause, ghi trạng thái chưa map. So việc giữ fallback rải phrase hiện tại như một biến thể retrieval trên dev trước khi thay đổi mặc định. Nguồn tham khảo dùng điểm theo sự kiện và dynamic programming trên timestamp. Trong VIDENTRA, phần xác minh vẫn gọi `GraphMatcher` trên refs. [SE‑Viterbi](src/videntra/retrieval/alignment.py#L250-L350) · [TEMPO‑P trong EVISTA](https://arxiv.org/abs/2610.14510)

**Regression.** A=6s/B=2s với khoảng caller sai phải trả `UNKNOWN` khi dùng khoảng làm bộ lọc và `CONTRADICTED` khi chỉ xét refs. Hai refs khác nhau cùng 5s phải `PARTIAL`. Thêm ca `after`, đúng/sai min/max gap, câu có nhiều occurrence, window thứ hai và timeline revision khác nhau. Chạy cả từ `search_with_metadata(verify_relations=True)` để kiểm scope cửa sổ thật. [Test hiện có](tests/relations/test_temporal_relation_grounding.py)

### 2.2. Đo đúng Recall@K ở đầu vào reranker

**Hiện trạng.** `shortlist_before_rerank_vids` được ghi từ toàn bộ `cand_moments` sau khi cộng residual LTR. `union_recall` và `shortlist_recall` trùng nhau trong cả ba split của run hiện tại. Trong khi đó `search_depth` giới hạn số frame hits trước khi gom video. Một giá trị K có thể đang mô tả frame, video hoặc vị trí export tùy stage. [Stage 5–7](src/videntra/retrieval/unified_pipeline.py#L2884-L3110) · [Tổng hợp recall](scripts/run_validation_benchmark.py#L668-L692)

**Cách sửa.** Chụp danh sách ID theo thứ tự tại bốn ranh giới. Mỗi ID là **video duy nhất**. Giữ frame depth và kích thước thực của từng pool ở trường riêng.

| Danh sách | Vị trí ghi | Dùng để chẩn đoán |
| --- | --- | --- |
| `visual_candidate_ids` | Sau gom video từ visual hits. Ghi kèm số frame đã truy xuất và số video phân biệt. | Target mất ngay từ visual hay lọt qua kênh khác. |
| `union_candidate_ids` | Sau hợp nhất các nhánh visual, OCR, ASR và caption khi bật. | Nhánh nào cứu được target. |
| `pre_rerank_video_ids` | Sau base fusion, trước LTR hoặc reranker thử nghiệm. Sort bằng điểm full precision và tie-break ổn định. | Recall@10/20/50 của pool thực sự có thể rerank. |
| `reranker_input_ids`, `ranked_video_ids` | Ranh giới vào/ra của reranker và lần xuất Video@K sau đó. | Target có thực sự được chấm lại và thay đổi hạng ra sao. |

Từ các danh sách trên, evaluator mới join target IDs để tính recall ghép cặp theo query, dataset và toàn run. Export `shortlist_hit`, `target_rank_before`, `target_rank_after` cùng kích thước pool. Ghi `NOT_MEASURED` khi thiếu stage trace và dùng `0` khi stage đã chạy với danh sách rỗng. Dùng lược đồ per-query version mới. Ground truth chỉ vào bước chấm, còn inference ghi được các danh sách trên từ original query và snapshot. Video@K dùng danh sách video, Moment@K giữ danh sách moment. Trong thử nghiệm relation rerank, tất cả video đem so phải dùng score cùng stage. [Tách video/moment hiện tại](src/videntra/retrieval/unified_pipeline.py#L1120-L1164) · [Chấm Video@K](scripts/run_validation_benchmark.py#L598-L636)

**Regression.** Tạo 60 video ứng viên và đặt target ở hạng 21 trong danh sách base. `union_hit=1`, shortlist Hit@10/20 bằng 0 và Hit@50 bằng 1. Thêm fixture nhiều frame hoặc nhiều window cho một video, kênh caption tắt, thiếu stage trace và tie quanh cutoff. Kiểm tra tổng hợp theo dataset khớp với từng dòng JSONL.

### 2.3. Khớp row spatial patches với frame thật trước khi chạy P8‑5

**Phát hiện.** `run_p85_keyframe_backfill` ghi `.npy` theo thứ tự đọc manifest. `_score_late_interaction_window` đọc theo `faiss_id`. Dry-run ingest với 2 rows `float16` đúng shape và file `.npy` thiếu row map vẫn được chấp nhận. Vì vậy, một export đảo thứ tự hoặc export subset có thể trả patch của frame khác dù file hợp lệ về shape. Snapshot benchmark hiện thiếu artifact patches, nên đây là rủi ro cho phép thử sắp tới. [Worker P8‑5](scripts/kaggle_multimodal_offline_worker.py#L626-L713) · [Preflight hiện tại](scripts/ingest_kaggle_offline_results.py#L135-L184) · [Runtime đọc patches](src/videntra/retrieval/unified_pipeline.py#L1454-L1482)

**Cách sửa.**

1. Với dense snapshot, lấy `faiss_id` từ `KeyframeMap`, xác minh unique và coverage liên tục rồi ghi row tại đúng `faiss_id`. Tạo sidecar `row_index`, `faiss_id`, `keyframe_id`, `video_id`, `timeline_revision` và hash ảnh đầu vào. Ghi SHA‑256 của file patch, mapping và visual index trong receipt. Batch thiếu patch phải báo lỗi kèm frame IDs. Với subset, giữ mapping tường minh hoặc một snapshot nhỏ khớp DB/index. Chỉ dùng `--allow-partial-keyframes` khi đã có cơ chế lookup thật sự hiểu subset.
2. Ingest đối chiếu sidecar với `KeyframeMap` của snapshot. Kiểm tra đủ các chiều `N×9×768`, dtype, tính hữu hạn, IDs, mapping và hash. Xác thực caption index/manifest trước khi cập nhật DB. Thử trên snapshot bản sao và cấu hình mở cùng `db_path`, index, patch sidecar và caption indexes. Giữ các artifact để chuyển config về snapshot baseline khi cần.
3. Khi runtime mở array, xác minh receipt thuộc snapshot đang phục vụ. Artifact không khớp phải báo status với lý do rõ. Với benchmark yêu cầu patches, ca mismatch cần dừng sớm để tránh ghi nhận một run thành công với dữ liệu thiếu. Chạy cùng preflight cho dry-run và ghi thật.

**Regression.** Hai keyframe có vector patch phân biệt, manifest đảo thứ tự, một ID lặp, một hàng thiếu, một batch không có features và một caption manifest sai mapping. Dry-run và ingest phải cùng verdict. Kiểm lookup bằng `faiss_id` trên output cuối. [Tests worker/ingest hiện có](tests/indexing/test_kaggle_offline_pipeline.py)

### 2.4. Buộc ablation và đặc trưng model phản ánh cấu hình chạy

- `_build_degraded_diagnostics` đánh dấu patches `disabled_by_config` ở A0–A3. Call site tính `late_int_sc` đang chỉ kiểm tra artifact đã nạp. Khi có dữ liệu mới, cần dùng cùng một điều kiện enable cho compute và diagnostics. Tách caption lexical, caption dense, patches, pooling và LTR thành các flag riêng cho thí nghiệm. Đăng ký flag trong cùng config runner/pipeline và ghi vào provenance. Giữ A0–A5 để tái lập run cũ. [Diagnostics](src/videntra/retrieval/unified_pipeline.py#L1166-L1205) · [Tính patch feature](src/videntra/retrieval/unified_pipeline.py#L2794-L2800)
- Artifact LTR hiện hành có trọng số cuối bằng **0** cho `s_late_interaction` và `rrf_vis_caption`. Khi có file patch/caption, cần đo nhánh mới bằng score residual riêng hoặc huấn luyện lại trên feature có dữ liệu. Một số đường caption còn có thể thay đổi candidate pool trực tiếp. Vì vậy tách thí nghiệm **rerank-only** với pool cố định khỏi thí nghiệm **caption candidate branch**. [Artifact đang dùng](models/ranking/videntra_ltr_v1.json) · [Caption candidate path](src/videntra/retrieval/unified_pipeline.py#L2253-L2326)
- Test khi nạp cả hai artifact. Spy xác nhận flag tắt không gọi hàm score, flag bật ghi có số query được tính, và kênh được yêu cầu nhưng thiếu artifact cho verdict rõ. Dùng fixture hai video có base score bằng nhau để chứng minh khi một feature được bật và có trọng số, ranking thay đổi như cấu hình.

## 3. Thử nghiệm nâng độ chính xác theo thứ tự

### 3.1. Rerank bằng feature đã có, với shortlist cố định

Stage 5 đã có `query_guided_temporal_attention_score`, thường dùng tối đa 5 frame gần peak để tạo feature LTR `temporal_attention_coherence`. Thử nghiệm đầu dùng lại các ma trận SigLIP 2 và hàm này như một **tín hiệu rerank tường minh**, có đối chứng mean/top‑m. So sánh cùng `video_id`, cùng anchor và cùng số frame. CLIP4Clip cung cấp đối chứng mean pooling, X‑Pool cung cấp ý tưởng chọn frame theo query. Cài challenger vào SSOT bằng NumPy trước, dùng encoder và artifact đang có của VIDENTRA. [Feature hiện tại](src/videntra/retrieval/alignment.py#L442-L492) · [CLIP4Clip paper](https://arxiv.org/abs/2104.08860) · [Repo CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) · [X‑Pool paper](https://arxiv.org/abs/2203.15086) · [Repo X‑Pool](https://github.com/layer6ai-labs/xpool)

1. Freeze `ranked_video_ids`, base score, `faiss_id`, anchor và keyframe timestamps của Top‑10 A5 trên dev. Chọn tối đa 5 keyframe theo khoảng cách PTS tới anchor, ưu tiên cửa sổ `±3s`. Nếu cửa sổ thiếu frame, cho phép fallback nearest có cờ coverage và đo riêng. Mỗi video dùng cùng chính sách lấy mẫu. VFR dùng timestamp/ordinal đã decode.
2. Trên cùng frame set, so `mean`, `mean_top_m` với `m=2,3`, và attention theo query. Với attention, bắt đầu từ `tau=12` hiện hành. Chỉ mở `{6,12,24}` nếu attention có tín hiệu cao hơn mean hoặc top‑m ở pilot. Log thêm `max - mean_top_m` để nhận diện ca một frame spike. Thử độ rộng cửa sổ `{±2, ±4}s` hoặc tối đa `{3,5,8}` frames theo từng lượt khi có lý do từ lỗi dev.
3. Dùng residual có cùng scale với base. Điểm xuất phát là `score_new(v) = base(v) + λ × scale_base(q) × clip(z_pool(v), -2, 2)`. `z_pool` là điểm pool đã chuẩn hóa trên shortlist của query, còn `scale_base` lấy độ phân tán của base scores trong cùng shortlist. Nếu phân tán quá nhỏ hoặc thiếu evidence thì giữ base cho ca đó. Bắt đầu `λ ∈ {0, 0.10, 0.25}`. `λ=0` phải cho danh sách baseline y hệt. Giữ tie-break ổn định và full precision cho đến serialization.
4. Nếu challenger giữ được Hit@5 và có net gain Top‑1, kiểm độ nhạy theo K `{5,10,20}`. Báo số target hạng 2–5 lên Top‑1, số Top‑1 cũ bị đẩy xuống, Hit@1 có điều kiện target nằm trong K và Hit@1 toàn tập. Khóa chính sách trước khi chạy test. Tính điều kiện áp dụng từ tín hiệu query lúc inference. MSR‑VTT là lát cắt để đánh giá.

### 3.2. Patches và caption trên Kaggle, rồi chạy ablation riêng

- **Patches.** Sau mục 2.3, tạo `3×3` SigLIP 2 spatial patches cho cùng keyframes. So đối chứng global frame pooling với MaxSim trên patches trong cửa sổ. Patch hiện lấy từ `vision_model.last_hidden_state`. Cùng 768 chiều chưa đủ để khẳng định dot product với text vector nằm trên scale đã học. Thêm đối chứng crop embeddings lấy qua image projection của cùng encoder và so retrieval bằng crop ở tối đa 1–3 vùng có ý nghĩa trên mỗi frame. Nếu raw patch score tạo nhiều false positive kiểu trùng vật thể nhưng sai hành động, thử top‑m patch pooling hoặc clause coverage nhẹ trước khi train projection. [SigLIP 2](https://arxiv.org/abs/2502.14786) · [Kaggle patch extraction](scripts/kaggle_multimodal_offline_worker.py#L684-L713) · [MaxSim hiện tại](src/videntra/indexing/augmented_vector.py#L567)
- **Caption.** Dùng Florence‑2 và dense E5 như nhánh bổ sung. Lượt đầu giữ candidate pool để đo caption làm feature. Lượt sau bật caption candidate recall để xem bao nhiêu target mới lọt Top‑K. Tách `base`, `base + patches`, `base + caption` và tổ hợp sau khi có kết quả từng nhánh. Mỗi caption giữ frame ID, source model/prompt hash và timestamp. Với caption dễ nhầm tên, số hoặc vai trò, giữ evidence OCR/ASR hoặc visual tương ứng khi xác minh. [Florence‑2](https://arxiv.org/abs/2311.06242) · [Multilingual E5](https://arxiv.org/abs/2402.05672)
- **Chi phí và artifact.** Với 11.581 keyframe, riêng `11.581×9×768×2` xấp xỉ **153 MiB** dữ liệu patch chưa tính overhead. Đây là ước lượng kích thước file. Chạy batch nhỏ trên Kaggle T4, tách SigLIP, Florence và E5 theo giai đoạn. Local dùng memory map, chỉ materialize frames của shortlist, đo p95 end‑to‑end, peak RAM/VRAM và index bytes trên máy mục tiêu RTX 3060 6 GB / RAM 32 GB. Dùng revision, weights hash, processor revision và license trong manifest.

### 3.3. Mở rộng recall khi target hụt ở Top‑K

- Sau khi có stage trace, tách **mất trước union** và **có trong union nhưng rơi trước shortlist**. Nếu mất từ frame candidate, thử một nhánh clip embedding nhẹ. Tạo vài cửa sổ theo shot hoặc timeline `2–4s` và lấy mean embeddings đã chuẩn hóa, lưu cùng video, `[start,end]` và mapping PTS. Cho FAISS trả clip candidates rồi union theo video để phục hồi các hành động có nhiều frame cùng hợp ngữ cảnh. Mean pooling là đối chứng có thể chạy lại từ feature hiện tại trước khi tái encode. [FAISS và frame budget hiện tại](src/videntra/retrieval/unified_pipeline.py#L1837-L1866) · [CLIP4Clip](https://arxiv.org/abs/2104.08860)
- Nếu target đã trong union, thử ranking hoặc quota video đơn giản trước khi thêm embeddings. So hai cấu hình có cùng K video cuối. Quét frame depth `{120,240}` để tách ảnh hưởng của lấy thêm frame khỏi ảnh hưởng clip pooling. Nếu một video chiếm nhiều hits, thử tối đa 2–3 frame đại diện cho mỗi video ở bước hợp nhất. Cấp nguồn recall riêng cho OCR/ASR theo intent và giữ original query trong union. Đo gain recall, RAM và latency ở từng cutoff.
- Chỉ thử video encoder hoặc temporal head mới sau khi đã đọc được lỗi còn lại. Nếu nhiều false positives cùng vật thể nhưng sai động từ hay đổi vai, tạo hard negatives human-reviewed có cùng chủ thể/bối cảnh nhưng khác hành động, thứ tự, màu, số lượng hoặc hướng. Thử head query–clip nhẹ trên frozen SigLIP features và budget nhỏ. Paper ICSVR hữu ích để thiết kế lát cắt compositional. [ICSVR](https://arxiv.org/abs/2306.16533)

### 3.4. Tune lại LTR và lexical gate khi feature đã ổn định

1. Sau khi chọn patch/attention/gate có ích, tạo candidate feature JSONL từ inference với feature schema, stage score và snapshot hashes cố định. Evaluator dùng ground truth để gắn nhãn. Trainer đọc rows đã gắn nhãn. Giữ query không có positive để báo coverage, xử lý riêng khi tạo cặp train. Mẫu negatives gồm video rank cao và distractor có cùng visual hoặc lexical cue đã review. Dựng caption index và model features từ ảnh/video hoặc model observations của snapshot.
2. Dùng **5 outer folds theo canonical `source_group`**. Chọn tham số bằng các inner folds của phần train, hoặc bằng tập tune group-disjoint bên trong dev nếu cần pilot nhanh. So `no LTR`, residual pairwise hiện hành và residual train lại trên cùng candidate/feature set. Khởi điểm tune `blend_weight` là `{0.005, 0.015, 0.030}` bên cạnh baseline tắt LTR. Sử dụng cùng hệ số khi train và infer, ghi rõ hệ số hiệu lực trong run. Chỉ thử LightGBM LambdaMART làm challenger nếu residual nhẹ đã đo ổn định hoặc có dấu hiệu interaction phi tuyến. `group` truyền cho LightGBM là số row của mỗi query, còn source group dùng để chia folds. [Trainer hiện tại](scripts/train_ltr_ranker_cv.py#L483-L547) · [LightGBM LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
3. Với F2, so gate baseline với gate giảm boost khi chỉ khớp từ chung và chênh visual lớn. Tách exact phrase (OCR/ASR intent), số/đơn vị và tín hiệu context, để truy vấn chữ hoặc lời thoại vẫn có thể tìm ứng viên từ cả corpus. Theo dõi Win/Loss của MSR‑VTT visual‑only cùng các pilot OCR/ASR tiếng Việt. Nếu refresh OCR/ASR trên Kaggle, đổi từng nhánh và dùng recipe/receipt mới để đo phần đóng góp.

## 4. Lịch tune, tiêu chí và cách chạy

| Đợt | Thí nghiệm tối thiểu | Điều kiện để chuyển bước |
| --- | --- | --- |
| **1. Sửa và chốt baseline** | Temporal regression, stage recall, P8‑5 mapping và cờ ablation. Chạy baseline trên dev khóa theo nguồn. | Hai danh sách video/moment, scores, query IDs và provenance khớp contract. Các thay đổi chỉ do cách chấm được ghi riêng. |
| **2. Tune visual rerank offline** | Trên dev nhỏ để smoke, so mean, top‑m và attention ở K=10. Chọn tối đa 2 cấu hình để chấm toàn bộ dev. | So paired Win/Loss, Hit@1 và Hit@5/MSR‑VTT. Đọc nhóm target nằm trong shortlist và nhóm vắng target. |
| **3. Thử artifact và mở coverage** | Backfill patches/caption từ Kaggle. Đổi từng feature. Thử clip pooling nếu có nhiều miss trước union. | Artifacts có hash/mapping đúng. Cải thiện phải xuất hiện ở slice dự kiến và có chi phí chấp nhận được theo đo thật. |
| **4. Train và đánh giá đã khóa** | Fit lại LTR khi feature chốt. Giữ một cấu hình để đánh giá trên split chưa dùng chọn tham số. | Báo paired delta, CI theo source group, số query có nhãn theo slice và p95/peak RAM/VRAM. |

- **Metric chính đề xuất** là MSR‑VTT global Hit@1 trên dev đầy đủ. Mốc pilot hợp lý là tăng khoảng **3 điểm phần trăm** trên 809 query, tức ít nhất **25 hit ròng** nếu dùng làm ngưỡng số đếm. Báo CI 95% cho *delta* bằng paired bootstrap theo source group. Một cấu hình có delta dương và CI còn cắt 0 vẫn là kết quả pilot, cần xác nhận thêm. Mốc 50% là mục tiêu của một run đánh giá theo protocol đã khóa.
- **Guardrails đề xuất** gồm Hit@5/10, MRR@10, Recall@K trước rerank, số Win/Loss OCR‑only/ASR‑only và nhãn moment của QVH. Với slice nhỏ báo số thay đổi cụ thể, kèm denominator. Theo dõi latency end‑to‑end p50/p95 và peak RAM/VRAM. Chốt mức hồi quy và ngân sách thiết bị ngay từ baseline trên cùng máy. Đo `cold/warm` riêng, ghi môi trường và cache regime.
- **Kiểm nhãn.** Review một mẫu ca top‑2 sai theo rubric `đúng clip / phù hợp ngữ nghĩa / thiếu chi tiết / sai`, hai người độc lập với ca khó. Archive dev có 10 nhóm caption trùng sau chuẩn hóa. Dùng raw video để xác minh positives tương đương rồi lưu sidecar mới, báo metric strict single-label và metric human-verified multi-label song song. Đưa các sample/queries từ cùng nguồn vào cùng outer fold. Bảng chính dùng corpus đầy đủ và filter giống lúc chạy baseline.
- **Quy tắc dùng test.** Dùng `dev_150` để bắt lỗi nhanh, `dev_full_859` và outer OOF để đọc kết quả chính. `heldout_test_201` là mốc lịch sử đã được quan sát khi phát triển. Freeze test mới theo nguồn/near-duplicate trước khi chọn cấu hình cuối nếu có corpus đủ quyền và coverage. Nếu chưa có test mới, ghi rõ kết quả `heldout_test_201` là kiểm tra hồi quy lịch sử.

### Lệnh tham chiếu hiện có

Chạy từ repo root. Dùng thư mục output mới cho từng run. Hai lệnh smoke benchmark sau đây chỉ chạy dev nhỏ. Khi triển khai, thêm lựa chọn dev đầy đủ / test độc lập vào runner để tune mà không chấm held-out trong từng lần quét. Các cờ thí nghiệm mới ở trên là thiết kế cần triển khai, tham số bên dưới đã có trong code.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/relations/test_temporal_relation_grounding.py tests/evaluation/test_validation_benchmark_synthetic.py tests/evaluation/test_provenance_leakage.py tests/indexing/test_kaggle_offline_pipeline.py tests/ranking/test_learned_ranker.py
.\.venv\Scripts\python.exe -B scripts/run_validation_benchmark.py --single-split-only --out-dir work/experiments/msrvtt_smoke_a5
.\.venv\Scripts\python.exe -B scripts/run_validation_benchmark.py --single-split-only --no-ltr --out-dir work/experiments/msrvtt_smoke_no_ltr
```

Sau khi sửa mapping P8‑5, dùng manifest keyframe đã đối chiếu với index. Các path dưới đây là ví dụ triển khai trên Kaggle và snapshot local thử nghiệm.

```bash
python scripts/kaggle_multimodal_offline_worker.py --mode p85 --keyframes-manifest /kaggle/input/videntra-keyframes/keyframes_manifest.jsonl --images-dir /kaggle/input/videntra-keyframes/images --out-dir /kaggle/working/p85-patches --no-enable-florence2 --siglip-batch-size 16
```

```powershell
.\.venv\Scripts\python.exe -B scripts/ingest_kaggle_offline_results.py --mode p85 --outputs-dir work/kaggle/p85-patches --db-path work/snapshots/msrvtt-pilot/metadata.db --processed-dir work/snapshots/msrvtt-pilot --dry-run
```
