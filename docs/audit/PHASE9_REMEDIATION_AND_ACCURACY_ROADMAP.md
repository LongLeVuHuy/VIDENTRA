# VIDENTRA — Kế hoạch sửa lỗi Phase 9 và nâng độ chính xác

Ngày 15/11/2026. Kế hoạch dựa trên lần rà soát tại `99254fc`, ghi ở commit `a808714`, và run `a2b3d80668124f38bfc2b0361066d7ea`. Các thay đổi, cấu hình và tiêu chí đề xuất dưới đây cần được kiểm chứng khi triển khai. [Nhật ký rà soát](../PROGRESS.md#15112026--rà-soát-độc-lập-phase-9-tại-99254fc) · [Benchmark gốc](../../benchmark/benchmark_summary.json)

**Thứ tự triển khai** — Sửa ingest và hợp đồng benchmark trước khi nạp thêm dữ liệu, đồng thời sửa verifier để dừng xác nhận quan hệ thiếu evidence. Tiếp theo đo một baseline trên split đã khóa, dùng dữ liệu đó để quyết định mở rộng candidate pool hay rerank. Hai thử nghiệm chất lượng đầu tiên nên là **thêm tín hiệu visual cục bộ trên shortlist** và **giảm nhiễu chữ/lời thoại theo intent và timestamp**.

## 1. Chọn đúng bài toán từ baseline

| Quan sát từ archive | Hệ quả đối với kế hoạch |
| --- | --- |
| Video Hit@1 là 92/150 (`dev_150`), 357/859 (`dev_full_859`), 78/201 (`heldout_test_201`). Tập test có 191/201 query MSR‑VTT. | Báo cả micro-average, macro-average giữa dataset và kết quả MSR‑VTT riêng. `dev_150` tiện để smoke, còn vòng chọn cấu hình nên dùng dev đầy đủ và các lát cắt theo loại truy vấn. |
| Trên `dev_150`, taxonomy tự động gán 28 ca nhầm hình tương tự (F4), 22 ca mất ứng viên (F1), 8 ca nhiễu lexical (F2), 3 ca trôi biên (F5). | Kiểm tra thủ công một mẫu theo từng nhóm. F1 định hướng tăng recall, F4 định hướng rerank, F2 định hướng fusion và F5 định hướng refine. Các nhãn này mô tả chẩn đoán heuristic của run. |
| A2→A3 thêm ròng 18 Top‑1 đúng trên 150 query. A3→A4 tăng Moment Hit@1 ở tIoU 0,5 từ 1 lên 16 trên 22 query có nhãn. A4→A5 giữ nguyên trạng thái đúng/sai Top‑1 của 150 query, Moment nDCG@5 tăng 0,4986→0,5121. | Dùng A3/A4 làm đối chứng phù hợp cho fusion và localization. Đo riêng phần tăng thêm của caption, attention, T‑SOT và LTR. Test có 0/2 moment đúng tại ngưỡng 0,5, cần thêm nguồn có nhãn thời gian để đọc kết quả. |
| Snapshot dùng 6.912 OCR observations từ RapidOCR và 3.860 ASR segments từ Faster‑Whisper Tiny, có dense E5 text. Caption và spatial patches chưa có trong snapshot. | Chạy đối chứng giữ nguyên visual index và chỉ đổi một nhánh dữ liệu. Các thay đổi trong code và các model đã tạo dữ liệu cần có trường riêng trong provenance. |

[Bảng ablation và taxonomy](../../benchmark/benchmark_report.md) · [Run và thông tin snapshot](../PROGRESS.md)

## 2. Sửa các lỗi đã xác nhận

### 2.1. Bảo toàn OCR/ASR khi ingest và retry

**Hiện trạng.** Worker đặt ID ASR dạng `<video_id>__asr_<ordinal>` cho cả PhoWhisper lẫn Faster‑Whisper. `OfflineMetadataStore` ghi `INSERT OR REPLACE` theo khóa đó. Một dòng PhoWhisper đã thay dòng Faster‑Whisper Tiny trong bản sao DB. Các ID ổn định của OCR observations và evidence members cũng cần được kiểm tra trong cùng bản sửa vì chúng tham chiếu nhau. SQLite `REPLACE` có thể xóa dòng xung đột để chèn dòng mới, tạo thêm hệ quả với foreign key và trigger. [Worker](../../scripts/kaggle_multimodal_offline_worker.py#L919-L965) · [SQLite store](../../src/videntra/storage/database.py#L509-L666) · [SQLite ON CONFLICT](https://www.sqlite.org/lang_conflict.html)

**Cách sửa.**

1. Dùng ID xác định từ loại record, clip hoặc keyframe, `recipe_hash` và ordinal trong recipe. Tạo `recipe_hash` từ model ID, commit/revision, weights hash và cấu hình ảnh hưởng đầu ra như VAD, chunking, decoder. Đưa asset hash hoặc timeline revision vào định danh qua clip/keyframe hoặc recipe để tránh hai timeline dùng chung ID. Áp dụng quy tắc tại nơi trích xuất, rồi truyền ID đó xuyên qua segment/word, observation/member và FTS. ID legacy tiếp tục tra được trong snapshot gốc.
2. Ghi phiên bản receipt, hash và số dòng của mỗi JSONL, đầu vào media/keyframe và model provenance. Preflight đối chiếu receipt với file, xác thực video/keyframe tồn tại, OCR box hợp lệ, khoảng ASR có `start < end` và cùng time basis với clip. `--dry-run` dùng cùng preflight. Artifact legacy thiếu receipt hoặc provenance nên đi qua đường import có xác nhận định dạng riêng, giữ trạng thái nguồn `unverified`.
3. Trước khi commit, phân biệt **ID và payload trùng** (retry, giữ nguyên) với **ID trùng nhưng payload khác** (báo lỗi kèm ID). Gom insert record, members, groups và cập nhật FTS vào cùng transaction trên một connection, bật foreign keys. Đọc lại dữ liệu của tất cả recipe thuộc video/keyframe vừa cập nhật để dựng groups. Nếu dùng group ID cũ dựa trên ordinal, tạo groups theo phiên bản snapshot để giữ liên kết lịch sử. Có thể rebuild FTS toàn bộ trong snapshot thử nghiệm đầu tiên, sau đó tối ưu theo video nếu đo thấy cần. [SQLite UPSERT](https://www.sqlite.org/lang_upsert.html) · [Transaction](https://www.sqlite.org/lang_transaction.html)
4. Thử migration trên SQLite backup mới có hash, kiểm `integrity_check`, `foreign_key_check`, số dòng theo `model_id`, và truy vấn FTS cho cùng một cụm từ trước/sau. Nếu receipt cũ có transcript đã bị thay, dựng lại từ artifact nguồn còn giữ. Báo rõ mức phục hồi nếu thiếu artifact. Promotion của snapshot dùng đường dẫn cấu hình, giữ snapshot cũ để đối chiếu.

**Regression cần có.** Fixture từ hai recipe cùng video và ordinal giữ đủ cả hai transcript, có word/member liên kết đúng, FTS trả bằng chứng có model ID. Retry cùng bundle giữ nguyên hash dữ liệu logic và số record. Gây lỗi sau bước insert nhưng trước khi rebuild FTS phải rollback toàn bộ. Kiểm thêm một receipt bị sửa hash và một keyframe tham chiếu sai video. [Điểm tích hợp ingest](../../scripts/ingest_kaggle_offline_results.py#L331-L413)

### 2.2. Khóa split theo nguồn và tái tạo OOF

**Hiện trạng.** `extract_dev859_ltr_dataset` gán `source_video_id` từ target clip đầu tiên. Hai clip AVC của `xvAvN5xn5oY` đã rơi vào fold 0 và 4. Held-out đã chấm hiện có giao bằng 0 theo clip ID và theo tiền tố nguồn AVC/QVH đã đối chiếu. Cổng hiện tại tính chồng lấn theo query ID. Báo rõ phạm vi kiểm định này trong provenance. [Tạo training rows](../../scripts/train_ltr_ranker_cv.py#L584-L602) · [Provenance](../../src/videntra/evaluation/provenance.py#L104-L181)

**Cách sửa.**

1. Dùng manifest `clip_id → source_group` từ provenance dataset. AVC dùng nguồn gốc của các cửa sổ `__s…__e…`, QVH dùng YouTube source ID đã xác thực, MSR‑VTT dùng ID nguồn trong metadata. Prefix là bước kiểm tra phụ khi format đã biết. Gộp exact duplicates qua asset hash. Khi kiểm near-duplicates, dùng dấu vân tay hình/âm thanh để đề xuất cặp rồi xác nhận trước khi gộp nhóm. Ghi rõ mức kiểm tra trong manifest.
2. Tái dùng [split lock](../../src/videntra/evaluation/split_lock.py) để freeze dev và test, báo coverage theo dataset và số group. Mỗi query có tất cả candidate rows nằm cùng fold. Với nhiều target thuộc nhiều nhóm, buộc các nhóm đó cùng fold bằng thành phần liên thông hoặc giữ query ngoài tập train cho đến khi xác định nhãn. Trình đánh giá chỉ nhận artifact có manifest hash và rule nhóm nguồn rõ ràng. [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
3. Tách hai lần chạy. Inference nhận `query_id`, original query và snapshot, xuất candidate rows cùng base score và feature vector. Evaluator/trainer offline mới join ground truth, gán labels và mapping nhóm. Khi trích feature cho LTR, dùng pipeline với LTR blend bằng 0 như hiện tại và xác minh parity trên cùng candidate. Giữ các query thiếu positive trong thống kê recall/candidate coverage, kể cả khi chúng không tạo được cặp train.
4. Lưu `fold_id`, `train_group_ids`, `eval_group_ids`, `eval_query_ids` và manifest hash vào artifact mới. Mỗi query OOF phải khớp duy nhất một fold, chuẩn hóa feature học riêng từ train của fold. Khi chọn hyperparameters hoặc ngưỡng, dùng dev tools hay inner folds theo group. `heldout_test_201` đã được quan sát qua nhiều vòng phát triển, nên ghi là held-out lịch sử. Dành split khóa mới theo nguồn cho đánh giá cuối sau khi chốt cấu hình.

**Regression cần có.** Pair AVC cùng nguồn được gán cùng fold. Fold artifact có query trùng, thiếu hoặc group xuất hiện cả train lẫn eval phải bị chặn. Ca multi-target và ca `source_group` chưa xác định phải báo trạng thái rõ. Xuất thống kê theo query ID, clip ID và source group theo từng split, kèm số lượng query đủ coverage.

### 2.3. Xác minh temporal relation bằng bằng chứng của từng sự kiện

**Hiện trạng.** Pipeline chia `anchor_win` ở điểm giữa để gán hai khoảng, rồi dùng chúng trong verifier. Fixture có hai event `unknown` và evidence rỗng vẫn ra `before=supported` khi đầu vào có hai khoảng giả định. SE‑Viterbi hỗ trợ OCR/ASR curves nhưng call site chỉ truyền visual curves. [Call site và khoảng sự kiện](../../src/videntra/retrieval/unified_pipeline.py#L1775-L1813) · [GraphMatcher](../../src/videntra/relations/matcher.py#L63-L115)

**Cách sửa.**

1. Bỏ suy ra event intervals bằng điểm giữa. Giữ `anchor_win` làm đề xuất tìm kiếm. Trong `GraphMatcher`, chỉ cho temporal predicate dùng event đã gắn `predicate_id`, `video_id`, timeline revision, khoảng thời gian, trạng thái grounding và `evidence_refs` hợp lệ nằm trong moment đang xét. Cùng video có thể có nhiều windows, nên ràng buộc evidence theo window, bao gồm cả đề xuất `_win2` và `_win3`. `UNKNOWN` khi thiếu anchor hoặc time basis chưa đối chiếu. Gắn refs của hai event khi trả kết quả temporal.
2. Sau bản sửa an toàn, tạo đường điểm cho **từng mệnh đề**. Visual lấy điểm tương đồng clause theo frame. OCR lấy phrase match gần timestamp frame theo track hoặc vùng chữ. ASR lấy phrase match theo interval segment hoặc word với mapping về clip. Tách mask khả dụng khỏi điểm nhỏ. Lưu trạng thái `missing`, `failed` hoặc `available` để quyết định có đủ căn cứ xác minh hay chỉ dùng đường điểm để đề xuất.
3. Đưa cả ba curves vào `viterbi_subevent_localization`, chuyển `after` về thứ tự chuẩn ở parser có test, áp dụng khoảng cách bằng giây từ ledger. Dùng đường đi làm giả thuyết cho retrieval. Mỗi event cần được đối chiếu về đúng predicate và refs trước khi dùng để xác minh. Nếu có ràng buộc min/max gap, tính khoảng cách giữa **hai biên sự kiện thực sự grounded**. Trường hợp overlap hoặc cùng frame diễn giải theo operator và tolerance đã định nghĩa, kèm lý do. Ý tưởng evidence theo event có tiền lệ ở TEMPO‑P. [Hàm SE‑Viterbi](../../src/videntra/retrieval/alignment.py#L250-L352) · [EVISTA/TEMPO‑P, preprint](https://arxiv.org/abs/2610.14510)

**Regression cần có.** `A before B` có hai OCR/ASR anchor → `SUPPORTED`. Đảo anchor của A và B → `CONTRADICTED` nếu grounding hai event đủ rõ. Chỉ có một anchor → `UNKNOWN`. Thêm ca window thứ hai không chứa evidence, ca hai event chung anchor, timestamp VFR và `before/after` có min/max gap. Đo riêng số `SUPPORTED` sai trên hard negatives trước khi bật relation rerank mặc định.

### 2.4. Gắn benchmark với model thực thi và trạng thái từng kênh

**Hiện trạng.** Pipeline trả `degraded` dạng mapping nhưng runner chỉ đọc list, tuple, set hoặc string. `run_benchmark(ltr_model_path=...)` hash tham số rồi khởi tạo runner không truyền đường dẫn, để pipeline dùng mặc định. [Đọc diagnostics](../../scripts/run_validation_benchmark.py#L441-L453) · [Khởi tạo benchmark](../../scripts/run_validation_benchmark.py#L801-L834)

**Cách sửa.**

1. Chốt mapping `channel → reason` cho diagnostics và giữ adapter đọc kiểu cũ trong runner nếu vẫn còn caller. Gom kênh lỗi cùng số query ảnh hưởng và ít nhất một lý do mẫu. Tách `disabled_by_config`, `missing_artifact`, `failed` và `ready`. Ablation cố ý tắt kênh có trạng thái khác với run full bị mất artifact. Run chỉ mang tên cấu hình mà manifest và trạng thái thực thi xác nhận.
2. Truyền `ltr_model_path` xuyên CLI (tham số đề xuất `--ltr-model-path`), runner và pipeline. Đối chiếu đường dẫn đã resolve, SHA-256 của artifact được nạp, feature schema và trạng thái load trước khi chạy. Nếu muốn chọn baseline tắt LTR, ghi lựa chọn đó rõ trong config. Với OOF, ghi số query thực sự áp dụng fold theo từng split và số fallback. Benchmark phải phản ánh fold weights đã dùng, kể cả khi model chính được nạp đúng.
3. Ghi trạng thái worktree và hash file đang chạy bên cạnh commit, cùng snapshot hash, hash các optional artifacts, model/tokenizer revisions, cache config, hardware, cutoffs. Khóa danh sách file trước inference và kiểm lại cuối run. Dùng fixture hai artifact có trọng số khác nhau để xác minh điểm ranking và provenance đi theo cùng artifact, kể cả khi đổi cwd. [Mã provenance run](../../scripts/run_validation_benchmark.py#L978-L1075)

**Regression cần có.** Lỗi E5 giả lập và missing caption khi cấu hình yêu cầu phải xuất hiện trong summary. Config tắt caption có trạng thái đúng. Custom LTR nạp đúng hash, file lỗi báo sớm, fold weights được áp dụng và khôi phục đúng giữa các query. Kiểm tra JSON/report hiển thị cùng trạng thái.

### 2.5. Chấm Video@K theo video và Moment@K theo khoảnh khắc

**Hiện trạng.** Pipeline cắt top 10 moments rồi evaluator deduplicate. Archive ghi nhận 17/150 query dev và 16/201 query held-out có một video chiếm nhiều vị trí trong top 10. [Output Stage 7](../../src/videntra/retrieval/unified_pipeline.py#L2938-L2951) · [Evaluator hiện tại](../../scripts/run_validation_benchmark.py#L471-L495)

**Cách sửa.** Từ cùng scored candidates, tạo hai danh sách nội bộ. `ranked_videos` gom theo `video_id` trước `[:K]`, dùng score của đại diện tốt nhất và tie-break ổn định. `ranked_moments` dùng score từng moment và NMS hoặc quy tắc đa cửa sổ hiện hành. Gọi chung một lần inference rồi lấy mỗi danh sách cho đúng bộ metric. `Hit@K`, MRR@K và Video nDCG dùng danh sách video. Moment mAP, nDCG và tIoU dùng danh sách moments. Giữ độ chính xác timestamp gốc đến lúc chấm. Làm tròn ở lớp trình bày. Đưa đủ hai danh sách vào kết quả per-query hoặc dùng schema version để truy vết. Endpoint/UI tiếp tục nhận danh sách moments. Trường bổ sung phục vụ Video@K.

**Regression cần có.** Mười một moments với hai đề xuất cho cùng video ở đầu và target là video phân biệt thứ mười. Video Hit@10 phải bằng 1, trong khi danh sách top 10 moments vẫn giữ cả hai đề xuất nếu qua NMS. Thêm ca corpus dưới K và tie score. Tạo run mới để so số trước/sau, dùng raw predictions nếu archive đủ độ sâu cho một phân tích hồi tố.

### 2.6. Đồng bộ báo cáo, mẫu số và acceptance

- Sinh bảng benchmark từ cùng summary generator. README dẫn tới bảng mới hoặc lấy số qua một bước render có kiểm tra nhất quán, tránh sao từng ô. Sửa `dev_full_859` thành 65,76/211,21 ms, `heldout_test_201` thành 57,07/100,50 ms nếu vẫn hiển thị run lịch sử đó. Ghi rõ latency online end-to-end và non-neural là hai số đo khác nhau. Giữ `NOT_MEASURED` cho cold latency và RAM chưa có đo. [README](../../README.md#L279-L314) · [Benchmark JSON](../../benchmark/benchmark_summary.json)
- Ghi `pure_asr_top1_count`, `pure_asr_fp_top1_count` và tỷ lệ có mẫu số. Tập dev hiện có 0/2 sai và held-out có 0/1. Báo N cho từng chỉ số moment, coverage và sự khác biệt giữa từng candidate pool. Render p-value từ kết quả paired test trong summary và dùng `NOT_MEASURED` khi thiếu phép đo. Giữ tên run, split version và metric version khi cập nhật docs, để checkpoint lịch sử vẫn đối chiếu được. [Report generator](../../scripts/run_validation_benchmark.py#L1109-L1190)

## 3. Các hướng tăng độ chính xác

### 3.1. F1 và F4 — tăng recall trước khi tăng độ phức tạp rerank

**Giả thuyết.** Một số query mất target vì keyframe chưa phủ hành động hoặc embedding toàn ảnh bỏ sót vật thể nhỏ. Khi target đã có trong shortlist, context gần peak và vùng ảnh có thể phân biệt các cảnh nhìn gần giống nhau. Chẩn đoán theo từng tầng trước khi chọn giải pháp.

1. Export video IDs ở visual top 30/120 frames đã gom video, mỗi nhánh text, union sau dedup và shortlist trước rerank. Báo **Recall@K video** với K là số video phân biệt. Giữ thêm thống kê độ sâu frame thực tế. Đo tỷ lệ target có sẵn nhưng về sai Top‑1, tách khỏi tỷ lệ mất target. Chạy oracle khả quan chỉ ở evaluator để ước lượng trần của shortlist.
2. Thử hai thay đổi độc lập trên cùng corpus. **Bổ sung ứng viên** bằng cách cấp quota ứng viên riêng cho original query, từng clause và OCR/ASR có intent. Dùng RRF cơ bản làm đối chứng và giữ cùng ngân sách số video khi so. Nếu recall đã cao và F4 còn nhiều, **rerank shortlist** bằng patch feature hiện có (`P8-5`), tính MaxSim theo clause, lấy thêm độ phủ hoặc điểm yếu nhất giữa các clause. So với đối chứng `768d`, `768d+864d` và context pooling trên các keyframe gần peak. Một điểm MaxSim cao ở một patch cần được đọc cùng độ phủ để tránh thưởng nhầm vật thể đơn lẻ. [RRF](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) · [SigLIP 2](https://arxiv.org/abs/2502.14786)
3. Nếu F1 vẫn tập trung ở hành động ngắn, thử index nhiều tỷ lệ thời gian offline. Giữ cùng số vector khi so sampling đều với bổ sung quanh chuyển cảnh/chuyển động, sau đó vẽ chất lượng theo ngân sách index ở bước riêng. Nếu F4 còn tập trung ở động từ hoặc đổi vai sau khi thêm patch, thử head tương tác query–frame/clip nhỏ trên shortlist, dùng feature đóng băng và hard negatives đã review. Chỉ cân nhắc LoRA hoặc encoder video mới khi head nhẹ chạm trần trên split mới. [Multi-granularity và temporal reranking](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/html/Tran_Towards_Efficient_and_Robust_Moment_Retrieval_System_A_Unified_Framework_CVPRW_2025_paper.html) · [CONQUER](https://arxiv.org/abs/2109.10016)

**Thước đo.** F1 dùng union Recall@K và số target được cứu ở cùng ngân sách. F4 dùng Video Hit@1 có điều kiện target ở shortlist, đồng thời báo Hit@1 toàn tập, MRR@10 và nDCG@10. So các lát cắt có vật thể nhỏ, hành động, đổi vai, scene dài và ngắn. Báo giá phải trả bằng index size, RAM/VRAM và p95 quanh stage thay đổi.

### 3.2. F2 — tăng chất lượng OCR/ASR tiếng Việt và giảm nhiễu lexical

**Giả thuyết.** Kênh chữ và lời thoại đã tăng chất lượng rõ trong A2→A3. Cần giữ khả năng chỉ dựa vào chữ hoặc lời thoại khi truy vấn yêu cầu chúng, đồng thời giảm điểm của tín hiệu trùng từ nhưng sai nội dung hoặc sai thời điểm.

1. Dựng pilot có nhãn thủ công trên cùng clip/keyframe cho RapidOCR so với OCRRouter dùng VietOCR hoặc PaddleOCR, và Faster‑Whisper Tiny so với Small/PhoWhisper Small. Thử từng cặp với dữ liệu tiếng Việt có dấu, mất dấu, chữ nhỏ, tên riêng, đọc số, giọng vùng và code-switch. Dùng cùng preprocessing/ROI khi so recognizer. Báo CER/WER theo quy tắc chuẩn hóa đã chốt, thêm exact match cụm/số/tên và phân bố sai lệch timestamp. PhoWhisper Small là ứng viên có bằng chứng từ benchmark ASR tiếng Việt của tác giả. Hiệu quả trên corpus VIDENTRA cần đo trực tiếp. [PhoWhisper model card](https://huggingface.co/vinai/PhoWhisper-small) · [VietOCR](https://github.com/pbcquoc/vietocr)
2. Với ASR, dùng VAD, đối chiếu giờ audio với timeline clip và ưu tiên word timestamps cho truy vấn trích dẫn. Nếu chỉ có khoảng chunk, lưu độ phân giải theo chunk và giới hạn cách dùng cho định vị. Với OCR, giữ cả văn bản có dấu và bản gập dấu, khử trùng kết quả cùng track/thời gian. Khớp exact phrase, số và đơn vị được tính riêng với dense E5. Dense text có thể tìm paraphrase và dùng kênh độc lập để đối chứng. [Faster‑Whisper VAD và word timestamps](https://github.com/SYSTRAN/faster-whisper#word-level-timestamps) · [Multilingual E5](https://arxiv.org/abs/2402.05672)
3. So baseline gate hiện tại với gate theo intent và độ đặc hiệu. Cụm hiếm trùng nguyên văn, có timestamp và đúng nhánh được bảo toàn ngay cả khi visual yếu. Từ phổ biến hoặc chỉ dense hit được giảm trọng số khi lạc khỏi cửa sổ visual hay không nhất quán số. Thử các biến thể cùng candidate pool để tách hiệu quả **ranking** khỏi hiệu quả **recall**. Cho phép evidence OCR/ASR mới đóng vai trò candidate độc lập trước khi áp gate.
4. Sau khi sửa ingest, chạy `--mode ocr_asr` trên tập pilot rồi mở rộng trên Kaggle. Đo B0 (snapshot gốc), B1 (chỉ đổi OCR), B2 (chỉ đổi ASR), B3 (cả hai). Nếu B1/B2 cùng cải thiện, chạy B3 có/không có gate mới để đo tương tác. Ghi revision, hash weights và phiên bản tokenizer trong receipt.

Một điểm khởi đầu cho thí nghiệm gate là `final_score = base_score + λ × gate × lexical_score`. Dùng `lexical_score` đã chuẩn hóa trên dev. Đặt `gate` theo các trạng thái cụ thể như exact phrase đúng intent và cùng cửa sổ, từ chung trùng ngoài cửa sổ, hoặc sai số/đơn vị ở predicate yêu cầu số. Ca exact phrase có thể có anchor thời gian từ OCR/ASR. Ca từ chung ngoài cửa sổ được giảm trọng số. Ca sai số/đơn vị loại phần boost của predicate tương ứng. Ghi thành phần điểm và lý do gate theo từng kết quả để xem ca tăng hạng nào đúng. Chọn `λ` từ dev khóa theo nguồn, cùng candidate pool và base score trong tất cả biến thể.

**Thước đo.** Theo dõi Video Hit@1/Recall@K ở nhóm OCR‑only, ASR‑only, multilingual và visual‑only, kèm số FP và mẫu số. Thêm hard negatives có từ phổ biến trùng, số khác, từ đồng âm, lời thoại đúng ở đoạn ngoài khoảng và biển chữ gần giống. Cần đánh giá cả clip đúng nhưng sai thời điểm.

### 3.3. Caption, Small‑LLM và LTR — thêm từng tín hiệu rồi đo phần tăng thêm

- **Florence‑2 + E5 captions.** Dùng `--mode p85` trên Kaggle sau khi đã khóa snapshot. Ghi caption theo frame ID, timeline, model và prompt hash. Cho caption tạo ứng viên như một nhánh riêng, đồng thời thử chỉ dùng nó làm feature trên shortlist. Mỗi run giữ cùng visual/text corpus. So `base`, `base + patches`, `base + captions`, `base + patches + captions`. Nếu log ghi kênh caption thiếu, chạy lại cấu hình caption trước khi kết luận. Caption là model observation. Hạn chế dùng caption đứng một mình để xác minh quan hệ hoặc tên/số chưa có evidence trực tiếp. [Florence‑2](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.html)
- **Small‑LLM query decomposition.** Bảo toàn original query và so `rules` với `LLM compact + rules fallback` trên group multi-clause/temporal. Đánh giá parser trước bằng cách kiểm mất phủ định, đổi vai, sai `before/after`, thay số hoặc trích dẫn. Ghi fallback rate, cache hit theo namespace model/prompt/schema, chạy cùng config khi so. Chỉ dùng cache được tạo từ câu truy vấn inference hợp lệ. Để trình đánh giá xử lý truy vấn test khóa mới sau khi chốt cấu hình.
- **LTR.** Tái huấn luyện residual pairwise baseline sau khi sửa source group. So `base`, `linear residual 18d`, `24d`, rồi mới dùng nhánh LightGBM sẵn có trong trainer làm challenger. Cố định candidate rows và base score. Đưa positive và hard negatives vào cùng query group, còn source group dùng cho CV. Chỉ bổ sung graded labels nếu có relevance đã review, và tách head/metric video với moment để video đúng nhưng biên sai không trở thành tín hiệu nhiễu. Với LambdaMART, sort data theo query, truyền `group` là số rows mỗi query, tune độ sâu, early stopping và cutoff theo inner dev folds. Báo rõ score dùng cho ranking. Nếu API thực sự cần xác suất, fit calibrator bằng dữ liệu riêng. [Trainer hiện tại](../../scripts/train_ltr_ranker_cv.py#L314-L485) · [LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)

### 3.4. F5 và multi-event — cải thiện biên bằng tín hiệu cục bộ có nhãn

1. Giữ `expand_boundary_contrast` làm đối chứng. Trên cùng danh sách video và anchor, thử tăng mật độ lấy mẫu theo PTS quanh peak ở shortlist nhỏ. Đi từ ordinal/timestamp trong ledger và mapping audio. Thử smoothing theo thời gian có mask ranh giới shot, hoặc refine start/end theo câu nếu kết quả cho thấy trôi biên có hệ thống. Mỗi run thay một yếu tố để biết lợi ích đến từ thêm frame, smoothing, shot prior hay boundary rule. Segment ngắn và dài được báo riêng. [Mã chọn biên hiện tại](../../src/videntra/localization/refine.py) · [QVHighlights và Moment‑DETR](https://proceedings.neurips.cc/paper_files/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html)
2. Với truy vấn nhiều sự kiện, cố định shortlist rồi so `sequential DP visual-only`, `SE‑Viterbi visual-only`, `SE‑Viterbi + OCR/ASR` và `+ T‑SOT` thành các run riêng. Bổ sung ablation attention và multi-window độc lập. Kiểm các ca cùng frame, sai thứ tự, audio đến trước hình, gap dài, hard cut và dấu timestamp thiếu. Dùng T‑SOT như tín hiệu tương đồng cho retrieval, để relation verifier dựa vào đường bằng chứng đã kiểm tra. [Alignment](../../src/videntra/retrieval/alignment.py) · [TEMPO‑P](https://arxiv.org/abs/2610.14510)
3. Nếu biên vẫn trôi sau khi tăng độ phủ frame và quy mô nhãn đủ để train, thử localizer nhẹ trên feature đóng băng. QD‑DETR là tham chiếu về biểu diễn video phụ thuộc truy vấn, phù hợp để lấy ý tưởng cho head cục bộ. Việc so điểm với QVHighlights công bố cần dùng evaluator và protocol tương ứng. Benchmark VIDENTRA tìm trong cả corpus, nên báo riêng phép đo nội bộ với phép đo trên video đã cho. [QD‑DETR, CVPR 2023](https://arxiv.org/abs/2303.13874) · [Evaluator QVHighlights gốc](https://github.com/jayleicn/moment_detr)

**Thước đo.** Báo Moment Hit@1 ở tIoU 0,5 và 0,7, mAP theo ngưỡng, sai lệch start/end theo giây và recall proposal. Đo toàn corpus và đo riêng khi target đã nằm trong shortlist. Tập có 22 query dev và 2 query test có nhãn biên dùng để debug/so ghép cặp. Muốn kết luận tổng quát, thêm corpus đã xác minh source group, quyền sử dụng, coverage, nhãn moment và hard negatives.

## 4. Thứ tự chạy và cách chốt một nâng cấp

| Đợt | Đầu ra kiểm tra được | Quyết định tiếp theo |
| --- | --- | --- |
| **1. Sửa dữ liệu và cách chấm** | Fixture ingest collision và rollback, manifest nguồn và OOF mới, model/provenance parity, verifier case thiếu evidence, Video@K và report đồng nhất. | Chốt snapshot baseline mới theo cùng schema kết quả. Báo các thay đổi số do sửa cách chấm riêng với thay đổi chất lượng. |
| **2. Phân tích lỗi dev** | Per-query stage recall, oracle shortlist, F1/F2/F4/F5 đã spot-check, nhãn OCR/ASR intent và multi-event, buckets theo dataset/language. | F1 cao thì thử **candidate coverage**. F4 cao khi target có sẵn thì thử **patch/context rerank**. F2 cao ở lexical thì ưu tiên **gate** và **chất lượng evidence**. |
| **3. Thử nghiệm từng thành phần** | Các cặp baseline/challenger chạy trên cùng split và cùng query IDs, mỗi lần đổi một yếu tố. Model nặng và backfill qua Kaggle, lưu receipt và snapshot version. | Ghép hai thành phần đã cho tín hiệu tốt để đo tương tác. Thử LTR hoặc neural localizer sau khi coverage và nhãn cho thấy phần tăng tiềm năng. |
| **4. Đánh giá cấu hình đã chốt** | Run trên split nguồn mới đã freeze, có per-query differences, confidence intervals và latency/memory trên thiết bị đã ghi đầy đủ. | Chọn cấu hình dựa trên metric chính, mức hồi quy theo slice và chi phí quan sát được. Giữ các run đối chứng để truy vết. |

- Với mỗi thí nghiệm, ghi giả thuyết, metric chính, lát cắt, ngưỡng hồi quy chấp nhận và ngân sách trước khi xem kết quả. Dùng **paired bootstrap theo source group** cho chênh lệch metric khi nhiều query dùng chung nguồn. Kèm số query/cặp chuyển đúng↔sai và số nhóm. Khi khoảng tin cậy còn rộng hoặc số dương ít, giữ kết quả ở mức pilot.
- Báo pre/post candidate Recall@K, Video Hit@1, MRR@10, nDCG@10, Moment Hit@1 và mAP nếu có nhãn. Tách lỗi không có trong index khỏi lỗi rank. Export inference queries và predictions riêng, để evaluator đính ground truth sau. Hậu kiểm bằng dữ liệu human-reviewed, mỗi hard negative ghi lý do như đổi vai, sai thứ tự, sai số hoặc sai biên.
- Bench CPU và RTX 3060 6 GB/RAM 32 GB theo môi trường thực tế. Ghi cấu hình, p50/p95, cold/warm, peak RAM/VRAM và dung lượng index. Dùng Kaggle cho embedding/caption hàng loạt, batch nhỏ ở pilot, shard theo video và resume từ receipt. Đặt budget shortlist `K` và số frames như tham số thử nghiệm. Có thể bắt đầu với shortlist 5, 10 và 20 video, giữ ngân sách frame cố định trong lượt quét đầu, rồi chọn theo chất lượng và chi phí đã đo.

### Lệnh tham chiếu khi triển khai

Các lệnh bên dưới dùng entry point hiện có. Lệnh full benchmark lưu run mới dưới `work/experiments`. Khi thực hiện ablation riêng hoặc đổi snapshot, lấy tham số từ manifest của experiment đã triển khai ở đợt tương ứng.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/evaluation/test_kaggle_ingest_safety.py tests/evaluation/test_split_lock.py tests/evaluation/test_provenance_leakage.py tests/evaluation/test_validation_benchmark_synthetic.py tests/evaluation/test_benchmark_verdicts.py tests/relations tests/retrieval/test_tsot_and_attention.py tests/localization
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B scripts/run_validation_benchmark.py --single-split-only --max-msrvtt 100
.\.venv\Scripts\python.exe -B scripts/run_validation_benchmark.py
```
