# VIDENTRA — Rà soát Phase 9 và kế hoạch sửa lỗi, nâng cấp

**Ngày 16/11/2026.** Đối chiếu các thay đổi Antigravity bàn giao đến commit `4ba4ebc` với mã nguồn tại `338efb7`. Phạm vi gồm SSOT retrieval, validator và cache Qwen, huấn luyện LTR, capture/replay, provenance benchmark và các artifact liên quan. Kết quả bên dưới phân biệt số đọc từ run lưu sẵn, phép tính lại trên predictions đã lưu và kiểm thử mới. Các thay đổi mã nguồn và thí nghiệm GPU được mô tả dưới dạng công việc đề xuất.

## A. Những phần đã triển khai và bằng chứng

| Phần | Trạng thái đã xác minh |
| --- | --- |
| P0 hardening và validator | `a710e37` chặn tự bật Qwen trên CUDA, đồng bộ version cache/receipt, kiểm SHA-256 labels và split manifest. `02803a2` bổ sung kiểm phạm vi phủ định/số lượng theo clause hoặc noun phrase, xử lý trích dẫn OCR/ASR và giữ câu gốc khi plan bị từ chối. |
| COCO-80 và Stage 2 | `4ba4ebc` lọc detector requirements ở bước mở rộng QueryGraph và khi đọc routing plan, giữ thực thể ngoài COCO trong graph. Stage 2 lấy điểm clause từ ma trận trộn 768d/864d (0,70/0,30) khi embedding và mapping phù hợp, dùng 864d làm fallback. Bộ lọc clause có hai điểm cần sửa ở B1. |
| Consensus và clip reserve | Consensus chạy khi `ablation_config == "A5"`, kể cả khi `enable_clip_branch=False`. Nó có thể nâng một video **đã có** trong `cand_moments`. Clip reserve là tùy chọn riêng, giữ K−1 video đầu và thay slot cuối. Replay clip reserve với `alpha=0` cô lập việc đổi pool ở cuối, trong khi A5 tham chiếu ngày 16/11 đã có Consensus. |
| Qwen và LoRA | SHA-256 của hai cache local khớp receipt. Capture base nhận `205` cache hits, `617` fallback và `37` rule-tier. LoRA nhận `821` cache hits, `1` fallback và `37` rule-tier. Cả hai có `0` live calls. Manifest SFT có 597 query thực từ `tune` và 139 mẫu bổ sung theo nhóm template, chia thành 626 train và 110 validation. |
| Kaggle VideoPrism và P8-5 | Báo cáo bàn giao ghi VideoPrism `CANCEL_ACKNOWLEDGED`. Lượt rà soát chưa thấy shard inference thật trong workspace. Run `0e7ec648` ghi `e5_caption_index`, `vlm_caption_lexical` và `spatial_patches` là `missing_artifact` trên tổng 1.210 lượt đánh giá. `temporal_rerank` ở trạng thái `disabled_by_config`. |
| UI và OCR/ASR | Lịch sử ghi các bản sửa Video Inspector và backend OCR/ASR, với các đường kiểm thử liên quan nằm trong suite. Audit này chưa kiểm UI tương tác trên trình duyệt hoặc đo lại chất lượng OCR/ASR theo mô hình. |

[Báo cáo replay và Kaggle](benchmark/msrvtt_clip_qwen_videoprism_audit.md) · [Nhật ký triển khai](docs/PROGRESS.md) · [Ma trận MSR-VTT](VIDENTRA_MSRVTT_IMPLEMENTATION_PLAN.md)

### Kết quả tham chiếu A5

| Run `0e7ec648` | Hit@1 | MRR | Video nDCG@10 | Phạm vi |
| --- | ---: | ---: | ---: | --- |
| `dev_150` | 93/150 (62,00%) | 0,6548 | 0,6752 | 129 query dùng fold OOF, 21 dùng fit-all |
| `dev_full_859` | 376/859 (43,77%) | 0,5142 | 0,5558 | 742 query dùng fold OOF, 117 dùng fit-all. MSR-VTT đạt 331/809 (40,91%). |
| `heldout_test_201` | 80/201 (39,80%) | 0,4805 | 0,5291 | `clean_disjoint` theo kiểm tra query ID của LTR. MSR-VTT đạt 72/191 (37,70%). |

- **MSR-VTT dev.** Rules A5 tăng `312 → 331/809` (+2,35 điểm phần trăm, 21 thắng, 2 thua). Tách `tune` `226 → 238/566` và `confirm` `86 → 93/243`. So sánh này bao gồm các thay đổi pipeline giữa hai lần chạy. Cần ablation để quy phần tăng cho từng thay đổi.
- **MSR-VTT clip reserve.** Tại K=10, Recall@10 tăng `544 → 555/809`. Tại K=20, Recall@20 tăng `607 → 611/809` (+0,49 điểm phần trăm, 9 thắng, 5 thua). Hit@1 giữ `331/809` ở `alpha=0`.
- **Qwen.** Tại K=20, cache base hòa rules về Hit@1 (0 thắng, 0 thua), mất một hit Recall@20. LoRA giảm `331 → 330/809` trên dev, hòa `93/243` trên `confirm` và giảm chỉ số Moment@0.5 dựa trên top-1 anchor từ `16/22 → 15/22`. `821/822` qua validator là tỷ lệ theo lượt query đủ điều kiện. `812` là số record cache duy nhất được export. Giữ rules cho MSR-VTT.
- **Held-out lịch sử.** Ghép đúng 201 query giữa `8c40e943` và `0e7ec648` theo query ID, câu gốc và labels đã lưu. Hit@1 toàn tập tăng `78 → 80/201` (2 thắng, 0 thua, +1,00 điểm phần trăm, CI 95% [0,00; 2,56], 143 nhóm). Riêng MSR-VTT tăng `71 → 72/191` (1 thắng, 0 thua, +0,52 điểm phần trăm, CI [0,00; 1,67], 133 nhóm). Tính lại bằng paired bootstrap theo `source_group`, 10.000 mẫu, seed `20261115`. Mốc 72/191 từng xuất hiện ở Phase 6. Đây là kiểm tra hồi quy trên split đã được xem; tác động riêng của Consensus vẫn chưa được đo.
- **Nghiệm thu `dev_150`.** Moment Hit@1 ở tIoU ≥ 0,5 là `16/22`, CI 95% [54,55%; 90,91%]. Moment nDCG@5 `0,5121` dưới ngưỡng `0,5300`. Pure-ASR FP `0/2`. Non-neural p50/p95 `53,71/124,35 ms` vượt ngưỡng `20/45 ms`. End-to-end p50/p95 `319,90/652,41 ms` trên Windows CPU, 4 logical processors theo provenance. Cold latency, peak RAM và hệ thống chạy trên RTX 3060 6 GB còn thiếu số đo.

[Báo cáo chuẩn](benchmark/benchmark_report.md) · [Summary JSON](benchmark/benchmark_summary.json) · [Replay ghép cặp dev](benchmark/msrvtt_clip_qwen_videoprism_audit.md)

## B. Lỗi và rủi ro cần xử lý

### B1. Bộ lọc clause có thể loại mất thông tin — ưu tiên P0

`unified_pipeline.py` chuyển về rules cho mọi routing plan có đúng một `visual_clause` và không có OCR/ASR, kể cả clause giữ được thông tin hữu ích. Ở nhánh dùng LLM, phép so `_meaningful_grounding_tokens(c) == _meaningful_grounding_tokens(query)` dùng tập token không thứ tự. Probe đã xác nhận “a dog chases a cat” và “a cat chases a dog” cho cùng tập token. Đây là lỗi trong tiêu chí nhận diện trùng; probe chỉ đo hàm so, mức ảnh hưởng với cache thật cần đo riêng.

1. Tách hai quyết định trong Stage 2. Validator kiểm clause có bám span nguồn và giữ đúng vai trò, phủ định, số lượng, thứ tự. Bộ khử trùng chỉ bỏ embedding phụ nếu clause trùng câu gốc sau chuẩn hóa bảo thủ, ví dụ NFC, hoa/thường, whitespace, dấu kết câu và mạo từ đầu câu khi an toàn. Giữ thứ tự, số lần xuất hiện, số, từ phủ định và phần trích dẫn. Hạn chế dùng tập token cho lexical overlap.
2. Giữ embedding câu gốc. Với một clause phụ hợp lệ và có ích, dùng nhánh ensemble 768d/864d khi mapping đầy đủ. Với clause bỏ mất ràng buộc, bỏ plan và trace lý do fallback. Khi truy vấn có OCR/ASR, giữ nội dung literal trong nhánh riêng.
3. Tập trung chuẩn hóa detector class ở một helper dùng chung của query layer. Đối chiếu class với taxonomy YOLO/COCO được pin. So alias nguyên cụm trước khi dùng từ cuối cụm, kiểm ca `screen` trong `phone screen` và `flat screen`. Giữ span, màu và cardinality khi chúng hợp lệ. Khi detector chưa chạy hoặc thiếu coverage, đánh dấu evidence `unknown` và tránh áp `no_object_penalty` chỉ vì thiếu dữ liệu. Tách trường hợp này khỏi ứng viên đã được detector kiểm tra.
4. Thêm regression ở `tests/query/test_llm_decomposer.py` và `tests/api/test_unified_parity.py` cho đổi mạo từ, đảo vai trò, số lượng, phủ định, câu tiếng Việt có dấu, visual clause hữu ích đứng một mình, multi-clause có và thiếu mapping 768d. Kiểm output capture với cache cũ và CLI/API khi tắt LLM. Đo parity của rules riêng trước khi chấm Qwen.

[Mã SSOT](src/videntra/retrieval/unified_pipeline.py) · [Validator](src/videntra/query/llm.py)

### B2. Consensus chạy mặc định ở A5 và chưa có ablation cô lập — ưu tiên P0

Code chọn video đầu của frame+clip RRF. Điều kiện 1 là `rank_frame=1`, `rank_clip=1`, `f_diff≥0,008`, `c_diff≥0` và `0<a5_gap≤0,045`. Điều kiện 2 dùng `rank_frame≤2`, `rank_clip=1`, `f_diff≥0,005`, `c_diff≥0,015` và `0<a5_gap≤0,030`. Khi đạt cổng, cộng `a5_gap + 0,002` vào điểm ứng viên. `clip_scores` là cosine với vector trung bình các frame SigLIP2 đã chuẩn hóa. Sliding window là tùy chọn cho điểm/anchor riêng. Frame và clip mean dùng cùng frame bank 768d.

1. Thêm cờ `enable_clip_consensus_boost` riêng trong SSOT và truyền qua benchmark/capture. Giữ cấu hình tái lập A5 ngày 16/11 bằng preset ghi rõ cả hai cờ Consensus và reserve. Ghi effective flags vào diagnostics, config và fingerprint. Thêm test tắt Consensus vẫn chạy được reserve và bật Consensus với reserve tắt.
2. Chụp Top-K trước/sau Consensus cùng `rank_frame`, `rank_clip`, `f_diff`, `c_diff`, `a5_gap` và lý do bỏ qua. So A5 bật/tắt trên cùng snapshot, LTR, rules, candidate set và chính sách timeline. Đếm cả thay đổi ID hạng 1, thắng/thua Hit@1 và ảnh hưởng moment. Kiểm tra boundary của mỗi điều kiện, ứng viên vắng mặt, filter gallery, tie-break và điểm thiếu.
3. Chạy ma trận nhỏ trên dev với hai trục `LTR off/on` và `Consensus off/on`. Dùng rules để đo Consensus và LTR trước, dùng cache Qwen khi cần cô lập B1. Nếu cần thử gate học, chỉ dùng nhóm tune để fit logistic nhỏ, khóa ngưỡng và đánh giá trên nguồn mới. `confirm` hiện tại được giữ để kiểm tra hồi quy cho các câu đã nằm trong dev từng được xem. Quyết định default dựa vào điểm tăng và CI của tập xác nhận mới, cùng guardrails modality và latency.

[Mã Consensus](src/videntra/retrieval/unified_pipeline.py) · [Nhánh SigLIP2](src/videntra/retrieval/clip_branch.py)

### B3. Clip index nhận snapshot đã đổi và làm lại việc mỗi query — ưu tiên P0

`matches()` gọi băm lại mapping video→fid nếu mapping được truyền. Signature chỉ lấy ba hàng vector đầu/giữa/cuối và không băm PTS. Fixture sáu hàng đã xác nhận đổi hàng thứ hai vẫn trả `matches=True`; đổi PTS ở hàng đó giữ nguyên signature. Khi mapping vắng mặt, `matches()` dựa vào shape trừ trường hợp caller truyền và kiểm `expected_signature` qua `validate_snapshot()`. Chi phí chạy hot path chưa được profile.

1. Tạo token snapshot khi load/publish từ checksum vector/FAISS, mapping `video_id → (faiss_id, presentation_ordinal, pts_time)` và phiên bản schema/model. So hash toàn tệp ở biên load hoặc build. Chuẩn hóa mapping vào đúng thứ tự nguồn, cố định arrays/mapping cho vòng đời snapshot. Dùng cùng nguồn mapping để build index và tính token.
2. Trong query chỉ so token giữa active snapshot và cached clip index. Invalidate khi đổi snapshot, index hoặc ledger. Dùng lock cho lần build đầu và publish cache sau khi validate xong. Khi fail, giữ snapshot nhất quán và trả lý do degrade qua metadata. Tái sử dụng điểm giữa Consensus và reserve cho cùng query/filter.
3. Bổ sung regression trong `tests/retrieval/test_clip_branch.py` cho thay đổi vector ngoài sample, đổi PTS, đổi mapping/array cùng shape, mapping fallback và lần gọi lặp không băm lại toàn bộ. Đo riêng cold build và warm search. Profile cả hash, dot products toàn gallery, sort/argmax và cửa sổ tùy chọn rồi mới chọn chỗ vector hóa.

[Chữ ký và cache clip index](src/videntra/retrieval/clip_branch.py)

### B4. Run tham chiếu có source dirty và gate còn hẹp — ưu tiên P0

Provenance của `0e7ec648` ghi `02803a2`, `Dirty=True` và 19 paths, gồm `llm.py` và `unified_pipeline.py`. Kiểm byte cho bốn source files được audit cho thấy hash run trùng checkout hiện tại và nội dung commit `4ba4ebc` sau xử lý khác biệt CRLF của `llm.py`. Đây là đối chiếu có phạm vi. Receipt hiện tại vẫn cần ghi đúng trạng thái lịch sử.

1. Trong `run_validation_benchmark.py`, thêm chế độ xuất bản reference có preflight **trước khi ghi output**. Chạy từ checkout sạch, ghi run vào thư mục mới dưới `work/experiments`. Git state không xác định thì reference preflight fail. Dev dirty vẫn có thể chạy ngoài `benchmark/`, gắn nhãn thí nghiệm kèm source diff/hash. Chỉ đồng bộ report + summary từ một run đã verify input/output, kèm receipt và run ID. Giữ bản lịch sử theo run ID.
2. Bổ sung hash config hiệu lực, code retrieval/evaluator, snapshot components, cache/receipt, labels và split manifest, track CPU/GPU và thread settings. Kiểm file đầu vào không đổi trước/sau. Với capture/replay tách runtime bundle (query text/ID + pool) và evaluator bundle (targets/labels). Thêm test xáo gold rồi kiểm output retrieval cho cùng query/snapshot giữ nguyên. Ghi số query bị lọc do target ngoài gallery ở tầng evaluator và báo coverage bên cạnh các metric.
3. `evaluate_provenance_gate()` hiện dùng giao query ID để gắn nhãn `clean_disjoint`. Bổ sung kiểm canonical source/media hash cho LTR train so với test, kể cả query IDs khác nhau và clip cắt từ cùng nguồn. Kiểm normalized query hoặc paraphrase family khi có metadata đáng tin. Group nhiều target bằng connected components như trainer. `dev_150` là tập con của `dev_full_859`. Với query dev ngoài danh sách train nhưng cùng source với train, chọn fold giữ nguồn đó ngoài train hoặc bỏ LTR cho query và ghi lý do. Thêm fixture trùng nguồn khác ID, query đa target và group chưa xác minh.
4. Khóa benchmark xác nhận mới theo source và asset exposure trước khi xem predictions. `tune=601` và `confirm=258` trong lock 70/30 của dev hiện tại có 0 overlap nhóm theo manifest. `heldout_test_201` lịch sử dùng cho hồi quy, kể cả sau khi gate kiểm tra thêm source.

[Runner](scripts/run_validation_benchmark.py) · [Gate LTR](src/videntra/evaluation/provenance.py) · [Paired evaluator](scripts/compare_msr_runs.py)

### B5. LTR cũ có biên điều chỉnh nhỏ, và trainer tree đang báo OOF của linear — ưu tiên P0 đo, P2 train lại

Artifact fit-all 24 features có `Σ|w|=0,007411`, bias 0. Nếu tất cả `|z_j|≤3`, điểm cộng `0,015 × tanh(w·z+b)` có độ lớn tối đa khoảng `0,000333` mỗi ứng viên. Chênh lệch điểm cộng giữa hai ứng viên có thể tới khoảng `0,000667` theo cùng giả định. Runtime chưa áp cận `|z|≤3`, và các cặp sát nhau vẫn có thể đảo hạng. Các fold OOF có trọng số riêng. `l2_reg=250000` và blend nhỏ là lý do cần đo sensitivity, cùng với thay đổi Stage 2/Stage 6 sau khi LTR được huấn luyện.

1. Dùng `--no-ltr` ở benchmark và capture để làm đối chứng, hoặc đặt `ltr_blend_weight=0.0` trong fixture SSOT. CLI hiện không có `--ltr-blend-weight`. Bật/tắt Consensus riêng theo B2 để thấy LTR có đổi điều kiện của gate hay không. Xuất phân vị điểm cộng, số top-1/top-5 flip, base margins ở các flip, Hit@1/MRR và khoảng tin cậy ghép cặp. Ghi SHA feature schema, policy sinh pool, snapshot và code sau khi capture.
2. `train_ltr_ranker_cv.py` hiện fit linear cho từng fold; nhánh LightGBM tạo tree artifact ở bước fit-all sau folds. Nếu thử tree, tạo **tree cùng backend và feature contract trong từng fold** và dùng scorer runtime để đánh giá. Schema artifact phải khớp `model_type`, features, scaling, score mode và snapshot; test việc chuyển fold/restore state và từ chối schema lệch.
3. Giữ residual làm mốc. Nếu tín hiệu mới có gain trên pool cố định, so linear pairwise đơn giản với LambdaMART. Với chế độ ranker chính, dùng score contract rõ: model score hoặc fusion với A5 sau chuẩn hóa trong từng query. `blend=1.0` hiện vẫn là `base + tanh(ltr_score)`. Fit normalization trên train fold, nhóm candidates theo query, sinh negatives từ pool đã freeze, chỉ join relevance trong evaluator/trainer. Không tái sử dụng OOF predictions để chọn mô hình trên cùng fold rồi gọi đó là xác nhận.

[Trainer](scripts/train_ltr_ranker_cv.py) · [Artifact LTR](models/ranking/videntra_ltr_v1.json)

### B6. P8-5 thiếu artifact, bước ingest có thể cập nhật snapshot theo nhiều phần — ưu tiên P1

`ingest_p85_artifacts()` copy sidecars theo từng file và cập nhật caption/FTS bằng transaction SQLite riêng. Nếu lỗi giữa chừng, thư mục đích có thể chứa tổ hợp cũ/mới. Worker ghi revision Florence/E5 ở summary; các trường model và checksum cần có trong receipt dùng để nghiệm thu.

1. Dùng snapshot staging mới với DB và FAISS tương ứng. Worker và ingest chạy `--mode p85`. Tạo manifest từ `faiss_id` chính tắc và file frame hợp lệ. Chạy Kaggle trên subset nhỏ để xác minh loader và mapping, sau đó mới backfill snapshot đủ điều kiện.
2. Preflight `--dry-run` với `p85_receipt.json`, `spatial_patches_map.jsonl`, caption JSON, E5 caption index/manifest và checksum visual FAISS. Kiểm mapping `faiss_id/keyframe_id/video_id`, PTS, shape/dtype, finite vectors và coverage thực theo branch. Pin revision/hash/license cho model/processor của SigLIP, Florence-2 và E5. Phân biệt run chỉ có patches với run có đủ caption lexical/dense. Kiểm lỗi giữa copy và commit trên fixture. Verify cả snapshot staging trước khi dùng cho benchmark và giữ đường quay về snapshot gốc.
3. Chạy riêng caption lexical, caption dense và spatial patches. Ghi candidate rescue, fallback, độ phủ và chi phí. Chốt một cấu hình trước khi thử tổ hợp. Nhãn caption sinh từ Florence-2 chỉ là evidence có provenance.

[Worker P8-5](scripts/kaggle_multimodal_offline_worker.py) · [Ingest](scripts/ingest_kaggle_offline_results.py) · [Model card Florence-2](https://huggingface.co/microsoft/Florence-2-base)

### B7. Qwen và moment cần điều kiện khác nhau — ưu tiên P1/P2

- **Qwen.** Giữ rules cho truy vấn MSR-VTT trong lúc chưa có gain. Nếu tiếp tục nghiên cứu, phân loại nguyên nhân `617` fallback của base (cache thiếu, validator reject, prompt version/revision mismatch) từ receipt và diagnostics. Thử Qwen trên tập đa sự kiện có nhãn người cho scope, roles, quan hệ và quote. Báo `schema-valid`, semantic accuracy, parser coverage, Hit@1/moment và latency trên cùng câu. Chỉ mở thêm LoRA khi chứng minh parser là bottleneck trên lát cắt đó.
- **Moment.** Mục tiêu thu tối thiểu 100 query có nhãn tách nguồn là mốc thu thập, cỡ mẫu nghiệm thu dựa vào số nguồn và hiệu ứng tối thiểu. Báo riêng localization oracle-video và end-to-end retrieval, kể cả miss khi không có prediction. Dùng tIoU, Moment nDCG@5, coverage và boundary MAE có mẫu số. Bổ sung pure-ASR positives/negatives và nhiễu OCR/ASR để guardrail `0/2` có ý nghĩa hơn.

### B8. Hồ sơ lịch sử trong working tree

Lúc bắt đầu rà soát, working tree có `AGENTS.md` đã sửa và 12 tài liệu Markdown đã bị xóa khỏi checkout. Các bản đã commit có thể truy xuất bằng `git show <commit>:<path>`. Giữ các diff này ngoài commit audit. Khi cần tham chiếu lại runbook/audit cũ, trỏ đến phiên bản có commit xác định.

## C. Hướng nâng cấp theo thứ tự thử nghiệm

### C1. Tách giới hạn coverage và rerank, rồi chạy VideoPrism

Trên MSR-VTT dev, A5 `Recall@20 = 607/809` và Hit@1 `331/809`. Clip reserve đạt `611/809`; vì vậy `198/809` ở ngoài **pool replay K=20** của clip và `280/809` có target trong pool nhưng hạng 1 sai. Đây là trần oracle của pool replay đó. Đối chiếu Stage Recall@K của pipeline để tìm rơi ở visual/proposal union/shortlist/rerank. Trong 478 lỗi MSR-VTT dev, taxonomy ghi 281 `F4_FINE_GRAINED_VISUAL_CONFUSION`, 128 `F1_VISUAL_CANDIDATE_MISS` và 69 `F2_LEXICAL_DISTRACTOR_OVERTHROW`. Taxonomy là chẩn đoán tự động. Đọc mẫu thắng/thua trước khi chọn giải pháp.

1. **Pilot VideoPrism-LvT-B trên Kaggle.** Freeze query bundle và pool đã capture từ SSOT. Worker `kaggle_videoprism_worker.py` dùng checkpoint/tokenizer/runtime đã pin. Chạy GPU thật trên nhóm nhỏ, xác minh 16 frame theo PTS, media SHA, query SHA, unit norms hữu hạn và bốn file shard có hash đúng receipt. Đo throughput, VRAM và resume trước khi mở rộng. Nếu dùng video đã có trên Kaggle, kiểm bytes và quyền sử dụng của nguồn. Bắt đầu từ pool candidate cần chấm. Embedding toàn gallery 1.324 video là bước riêng để thử VideoPrism như nhánh tăng recall.
2. **Rerank trên pool cố định.** Chạy `R01` trên A5, `R11` trên clip reserve và chỉ chạy `Q11` nếu cần kiểm tương tác với cache Qwen. So `alpha=0` và các mức `{0,25; 0,5; 0,6; 0,75; 1,0}` trên tune, chọn K trong `{10,20}` rồi khóa. Replay có `--vp-coverage-policy full` và `preserve-unscored`. Với gallery hỗn hợp phải báo full/partial/fallback, tỷ lệ query có thể đổi top-1 và effect trên toàn slice. Thí nghiệm Top-5 cố định giải quyết nhầm hạng đầu. Để thử cả ứng viên reserve ở hạng K, dùng pool K đầy đủ và `alpha>0,5`, hoặc đưa ứng viên mới vào shortlist bằng policy có kiểm soát và đo lại coverage.
3. **Cổng quyết định đề xuất.** Đăng ký mục tiêu Hit@1 MSR-VTT tăng ít nhất 3 điểm phần trăm và CI ghép cặp theo source có cận dưới >0 trên tập xác nhận mới. Báo Recall@K, MRR@10, nDCG@10, số thắng/thua, coverage và giới hạn latency. Nếu coverage thấp hoặc CI rộng, giữ run như pilot và ghi blocker cụ thể.

[Runbook và lệnh capture/replay](benchmark/msrvtt_clip_qwen_videoprism_audit.md) · [Worker](scripts/kaggle_videoprism_worker.py) · [VideoPrism chính thức](https://github.com/google-deepmind/videoprism) · [Bài báo VideoPrism](https://proceedings.mlr.press/v235/zhao24f.html)

### C2. Mở rộng Recall@20 bằng nhánh bổ sung

1. Ưu tiên Florence-2 caption lexical/dense sau B6. Đánh giá matched caption rows và positive rescue từ pool hiện tại. Giữ ngân sách K như nhau và báo thêm recall trước khi cap để thấy việc đẩy ứng viên cũ ra khỏi pool.
2. Nếu còn coverage miss, thử Rocchio có trọng số trên normalized embeddings với tối đa ba ứng viên đầu, hoặc thêm hai cách diễn đạt có span grounding và hợp nhất RRF cùng câu gốc. Chạy từng nhánh riêng. Chặn query drift ở OCR/ASR, negation, số lượng và entity roles. Dùng đúng dimensionality/normalization của từng branch.
3. Mục tiêu tune đề xuất là Recall@20 từ `75,03%` (A5) lên ≥80% với cùng gallery và K. Khi chọn công thức, xem cả query được cứu lẫn query bị đẩy khỏi pool. Chấm lại Hit@1 và guardrails QVH/AVC trên cấu hình đã khóa. Nếu thêm kênh mới chỉ có coverage một phần, báo toàn tập và lát cắt coverage.

### C3. VLM rerank Top-5 như thí nghiệm có chi phí giới hạn

Chỉ chạy khi C1 cho thấy còn nhiều nhầm lẫn giữa các ứng viên đã vào Top-5. Trong held-out lịch sử `0e7ec648`, 36/191 query MSR-VTT có target ở hạng 2–5, 21/191 ở hạng 6–10 và 62/191 ngoài Top-10. Đây là cơ sở để chọn pool, không dùng các video gold để tạo input cho VLM.

1. Chọn một checkpoint VLM nhỏ có quyền dùng phù hợp và pin revision, hash, processor, prompt, số frame và decoding. Lấy 4–8 frame trên mỗi ứng viên theo PTS từ timeline hoặc A5 anchor, đặt quy tắc fallback khi video/frame thiếu. Inference Kaggle offline theo lô nhỏ với câu gốc và ẩn thứ hạng A5 khỏi prompt. Gặp input lỗi hoặc output mơ hồ thì giữ thứ tự A5 cho phần chưa chấm.
2. Cache khóa theo hash câu gốc, media/snapshot, frame ordinals/PTS, recipe, model và prompt. Lưu raw score và trạng thái parse. Điểm model chỉ làm tín hiệu xếp hạng. Pilot đo tỷ lệ score hợp lệ, thời gian và VRAM trước khi scale; tải việc là số query × số ứng viên thực × số frame.
3. So A5, VideoPrism và VLM trên cùng Top-5; thử fusion kết hợp sau khi biết đóng góp riêng. Báo delta ghép cặp và kiểm thủ công hard negatives về action, role và attribute. Dừng scale nếu phần cải thiện nhỏ hoặc chi phí vượt budget đã khóa.

### C4. Reranker học được hoặc fine-tune biểu diễn khi đã xác định bottleneck

- **Reranker.** Khi có gain từ C1/C2/C3, dùng features A5, SigLIP2, VideoPrism/VLM và lexical để so residual, pairwise linear và LambdaMART theo B5. Missing features mang cờ missingness. Train và đánh giá trên pool thật, log tỷ lệ query có positive trong pool. Lưu bản không LTR làm đối chứng và rollback.
- **Biểu diễn.** Nếu recall vẫn là nút thắt sau khi so các nhánh, thử projection head nhỏ trên SigLIP2 trước khi mở LoRA text tower. Dùng caption/media có quyền trên tune, nhóm toàn bộ caption của cùng video/source trước khi split, chọn negatives từ capture và xử lý khả năng false negative của nhiều caption. Dùng cùng dimension/normalization giữa train và inference. Nếu thay video embeddings thì rebuild snapshot riêng; nếu thay text tower chỉ dùng khi có contract tương thích frame bank. Đo Recall@K, Hit@1, coverage và chi phí encode; lưu baseline frozen và hashes.

### C5. Định vị moment và hiệu năng

- **Moment.** Mở tập nhãn theo B7, ưu tiên corpus đã có quyền và ledger đủ. Theo hồ sơ QVH, 18 clip source-disjoint đang quarantine với `rights_status=unknown`; audit coverage lịch sử ghi dev `22/1.240` query có target trong index và test mới `0/18`. Xác minh quyền và coverage trước khi dùng các clip mới. So baseline contrast boundary, multi-peak NMS và DP đơn điệu cho đa sự kiện. Freeze số proposal và temporal policy theo tune, đo Moment nDCG@5 cùng tIoU 0,3/0,5/0,7 và boundary MAE. Dùng PTS/time_base cùng mapping clip/source; giữ thứ tự decode presentation-order. Mục tiêu dự án là Moment nDCG@5 ≥0,53 trên protocol đã khóa. [Bài báo QVHighlights](https://proceedings.neurips.cc/paper_files/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html) · [Hồ sơ coverage](docs/PROGRESS.md)
- **Latency.** Profile ngay trong B3/B4. Dùng cùng snapshot/query list và config, tách cold load, warm cache bật/tắt, text encoder, visual scoring, FTS, Stage 4–6, Consensus và serialize. Báo p50/p95/peak RSS, VRAM, threads và thiết bị, chạy CPU và RTX 3060 riêng. Sửa stage chi phí lớn nhất với test parity thứ hạng và anchor. Giữ ngưỡng đang dùng `20/45 ms` cho non-neural; mốc trung gian `30 ms` ở p50 cần gắn nhãn, cùng p95. Kiểm tải concurrency riêng sau khi warm single-query đã ổn định.

## D. Trình tự bàn giao và nghiệm thu

| Bước | Việc cần bàn giao | Qua bước khi |
| --- | --- | --- |
| 0 | B1–B5. Regression các ca đã tái hiện, flags/phân biệt A5+Consensus và reserve, provenance gate, capture A5 rules có và không LTR/Consensus. | Cùng snapshot và labels lock, chỉ đổi biến đã định. Có source hashes, top-k parity baseline, gain/loss và lỗi khi gate từ chối. |
| 1 | Pilot shard VideoPrism (C1) và P8-5 staging (B6), lên lịch Kaggle theo năng lực thực. | Artifacts thật, receipts khớp, coverage/fallback rõ. GPU smoke ghi thời gian và bộ nhớ đã đo. |
| 2 | Cô lập rerank VideoPrism và recall từ caption. Thử VLM nếu còn đủ lỗi trong Top-5. | Báo ghép cặp trên tune, khóa cấu hình trước khi chấm confirm. Có metric, CI, theo dataset, kênh OCR/ASR và moment. |
| 3 | Chọn tối đa một challenger chính cho vòng xác nhận, cùng kiểm hồi quy trên historical held-out. Train ranker/projection khi C4 đủ điều kiện. | Công bố run ID, số query và số nhóm, cấu hình khóa, phạm vi coverage, guardrails và quyết định bật/giữ experimental. |

Các lệnh hiện có để tạo đối chứng LTR trên dev, chạy sau khi đã chọn cùng snapshot và ghi cờ Consensus thực tế. Đổi thư mục output cho mỗi lượt.

```powershell
$py = '.\.venv\Scripts\python.exe'
$phase9Run = 'work/experiments/phase9_ltr_' + (Get-Date -Format yyyyMMdd_HHmmss)
& $py -B scripts/run_validation_benchmark.py --splits dev_full_859 --out-dir "$phase9Run/with_ltr"
& $py -B scripts/run_validation_benchmark.py --splits dev_full_859 --no-ltr --out-dir "$phase9Run/without_ltr"
```

Dùng `capture_msr_candidate_pools.py --rules-only` để tạo baseline replay có `llm_live_calls=0`. Khi B2 có cờ mới, chạy ma trận B2 trên cùng recipe, dùng `run_clip_vp_replay.py --alpha 0` kiểm parity rồi chấm bằng `compare_msr_runs.py` với labels và split lock đã hash. Nếu bất kỳ nhánh nào rớt artifact, ghi status và tách run degraded khỏi kết quả nghiệm thu challenger.

## E. Kiểm tra trong lượt rà soát

- `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` trên Windows CPU, PyTorch `2.15.0+cpu`: **588 passed, 3 warnings trong 75,38 giây**. Các warning liên quan API CPU quantization. Kiểm thử gồm fixtures/regression theo suite hiện hành.
- Probe read-only đã xác nhận signature bỏ sót một hàng embedding và PTS, hàm token-set đồng nhất hai câu đảo vai trò, và cache SHA-256 trùng receipt ở cả base lẫn LoRA. Phép ghép held-out dùng hai run trong `work/experiments` và predictions/target đã lưu, không sinh thêm predictions.
- Lượt rà soát ban đầu chưa chạy engine benchmark mới, Kaggle GPU hoặc đo trên RTX 3060.

---

## F. Kết quả triển khai B1–B6 và Thực nghiệm Bước 0 (16–17/11/2026)

### F.1. Các bản sửa B1–B6 đã hoàn tất & kiểm thử hồi quy
- **B1 (Query Decomposer, chuẩn hóa COCO & `detector_ran` guard):** Bổ sung `"flat screen"/"tv screen"/"television screen": "tv"`, `VALID_COCO_CLASSES`, `resolve_coco_class()`, `normalize_object_requirements()` ([rules.py](src/videntra/query/rules.py)); `canonical_clause_key()` & `is_redundant_visual_clause()` bảo toàn thứ tự từ và ngoặc kép ([llm.py](src/videntra/query/llm.py)); truyền cờ `detector_ran` vào `compute_object_verification_boost` ([entities.py](src/videntra/relations/entities.py)) và `_fast_object_boost_from_summary` ([unified_pipeline.py](src/videntra/retrieval/unified_pipeline.py)) để video chưa chạy YOLO không bị phạt `-0.10`; bảo toàn single-clause `use_llm_clauses` trước khi lọc mệnh đề thừa (fallback `decompose_query_clauses` khi rỗng và không có OCR/ASR).
- **B2 (`--a5-preset` & cờ tách Consensus/LTR):** Thêm `--a5-preset {a5_full,a5_lite,a5_no_consensus,a5_no_ltr}`, `--enable-clip-consensus-boost` / `--disable-clip-consensus-boost`, `--ltr-blend-weight` vào `UnifiedMultimodalSearchPipeline`, [capture_msr_candidate_pools.py](scripts/capture_msr_candidate_pools.py), và [run_validation_benchmark.py](scripts/run_validation_benchmark.py); telemetry ghi đầy đủ `clip_consensus_stats` và `ltr_stats` (`applied_count`, `top1_flipped_count`, `max_abs_delta`, `p50_max_abs_delta`).
- **B3 (Snapshot-safe `ClipBranchIndex` v2 & cache O(1)):** Nâng cấp `CLIP_BRANCH_SCHEMA_VERSION = "clip_branch_v2"` kèm SHA-256 toàn phần embeddings (`compute_embedding_signature`), `fids` và `mapping` bao gồm `(presentation_ordinal, pts_time)` ([clip_branch.py](src/videntra/retrieval/clip_branch.py)); thêm `invalidate_clip_branch_index()`, `publish_snapshot()`, khóa `_clip_index_lock`, `_ensure_clip_branch_index()` kiểm tra O(1) qua `_snapshot_clip_token` và tái sử dụng `_last_clip_score_res`/`_last_primary_q_768`.
- **B4 (Provenance Gate theo connected source group & media SHA-256):** Hợp nhất quy tắc `assign_connected_source_groups()`, `resolve_oof_fold_for_query()`, `compute_source_manifest_hash()`, `compute_effective_config_sha256()`, `compute_code_sha256()`, `get_cpu_thread_settings()` ([provenance.py](src/videntra/evaluation/provenance.py)); phát hiện và chặn rò rỉ liên thông nguồn/media (`"source_leakage_detected"`, `"unverified_source_group"`, `"fold_model_type_mismatch"`).
- **B5 (Per-fold LightGBM CV & state isolation):** Thêm `_apply_payload()`, `snapshot_state()`, `restore_state()`, `apply_fold_payload()` ([learned_ranker.py](src/videntra/ranking/learned_ranker.py)); mặc định `--backend pairwise_logistic` và huấn luyện/đánh giá `tree_ensemble` bên trong từng CV fold cho nhánh `lightgbm` ([train_ltr_ranker_cv.py](scripts/train_ltr_ranker_cv.py)), kiểm thử với `tests/ranking/test_learned_ranker.py` (`11 passed, 0 skipped`).
- **B6 (P8-5 Receipt v1.1 & Atomic Staging Ingest):** Bổ sung `p85_coverage_mode` và pin model/license Florence-2/E5 trong [kaggle_multimodal_offline_worker.py](scripts/kaggle_multimodal_offline_worker.py); bổ sung `--staging-dir`, đóng kết nối SQLite tường minh (`conn.close()`), copy file-first kèm rollback `.bak.<pid>` trên giao dịch `BEGIN IMMEDIATE`/`COMMIT` trong [ingest_kaggle_offline_results.py](scripts/ingest_kaggle_offline_results.py).

### F.2. Kết quả Ma trận 2×2 Consensus × LTR (Bước 0 — Same Snapshot, Same Rules)

#### Trên `dev_full_859` (859 queries / 858 source groups, `eval_mode=oof` 742 OOF + 117 fallback)

| Cell (`--a5-preset`) | Consensus / LTR | Global Hit@1 | MSR-VTT (`n=809`) | MRR@10 | Recall@10 | Consensus (`applied/swap`) | LTR (`applied/flip`) | So với `C0_L0` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0_L0` (`a5_lite --no-ltr`)** | OFF / OFF | `41.56% (357/859)` | `38.57% (312/809)` | `0.5009` | `68.80% (591)` | `0 / 0` | `0 / 0` | Baseline |
| **`C0_L1` (`a5_no_consensus`)** | OFF / `0.015` | `41.56% (357/859)` | `38.57% (312/809)` | `0.5010` | `68.80% (591)` | `0 / 0` | `859 / 0` | `0W/0L (+0.00 pp)` |
| **`C1_L0` (`a5_no_ltr`)** | ON / OFF | **`43.77% (376/859)`** | **`40.91% (331/809)`** | `0.5141` | `68.80% (591)` | `46 / 46` | `0 / 0` | **`21W/2L (+2.21 pp, CI [+1.16, +3.37])`** |
| **`C1_L1` (`a5_full`)** | ON / `0.015` | **`43.77% (376/859)`** | **`40.91% (331/809)`** | **`0.5142`** | `68.80% (591)` | `46 / 46` | `859 / 0` | **`21W/2L (+2.21 pp, CI [+1.16, +3.37])`** |

- Khớp `100%` (`859/859 ties`) giữa `C1_L1` mới và `R00_k10` lịch sử; trên lát cắt `confirm` (`n=258`, gồm `243` MSR-VTT), `C1_L1 vs C0_L0` và `C1_L1 vs C0_L1` đều đạt **`+2.71 pp` (`37.98% -> 40.70%`, `7W/0L`, CI `[+0.78, +4.65]`)**; trên `tune` (`n=601`): **`+2.00 pp` (`14W/2L`, CI `[+0.83, +3.33]`)**; `0W/0L` trên AVC (`26/28`), QVHighlights (`19/22`), và Moment (`0.7273`).

#### Trên `heldout_test_201` (201 queries, `eval_mode=clean_disjoint`, `source_group_overlap=0`, `leakage_free=True`, `gate_passed=True`)

| Cell (`--a5-preset`) | Global Hit@1 | Within-DS Hit@1 | Hit@5 / Hit@10 | Global MRR | nDCG@10 | Moment nDCG@5 / Hit@1@0.5 | Consensus / LTR Swaps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0_L0` (`a5_lite --no-ltr`)** | `38.81% (78/201)` | `40.80%` | `58.21% / 68.66%` | `0.4754` | `0.5254` | `0.3155` / `0.00%` | `0 / 0` |
| **`C0_L1` (`a5_no_consensus`)** | `38.81% (78/201)` | `40.80%` | `58.21% / 68.66%` | `0.4755` | `0.5255` | `0.3155` / `0.00%` | `0 / 0` |
| **`C1_L0` (`a5_no_ltr`)** | **`39.80% (80/201)`** | `41.29%` | `58.21% / 68.66%` | `0.4804` | **`0.5291`** | **`0.5000` / `50.00%`** | `7 / 0` |
| **`C1_L1` (`a5_full`)** | **`39.80% (80/201)`** | **`41.29%`** | `58.21% / 68.66%` | **`0.4805`** | **`0.5291`** | **`0.5000` / `50.00%`** | `7 / 0` |

### F.3. Kiểm chứng Qwen hậu-B1 & Benchmark Đa Tập Tham Chiếu
- **Post-B1 `Q00` (`extractive-v2`, `205` cache hits):** Khôi phục hoàn toàn Recall@10 từ `590/859` (tiền-B1) lên **`591/859`** (`+1` net recall gain so với tiền-B1; hòa `0W/0L` về Hit@1 `43.77%` so với `C1_L1`, MRR `0.5142 -> 0.5132`).
- **Post-B1 `L00` (`lora-sft-v1`, `821` cache hits):** Khớp `859/859` so với tiền-B1 `L00_k10` (`43.66% (375/859)`, MRR `0.5134`, `0W/1L` so với `C1_L1` ở `tune`, `0W/0L` trên `confirm`). Theo tiêu chí B7/P3, giữ **`--rules-only` + `a5_full`** làm mặc định.
- **Benchmark Đa Tập Tham Chiếu (`Run ID 41473d9023c54a2fadf405801367659a`, snapshot bất biến `work/experiments/20261117_002157_all_splits_a9f4367_41473d90`, `leakage_free=True`, `gate_passed=True`):**
  - `dev_150` (OOF `129/150`): Video Hit@1 **`62.00%` (`93/150`, `[54.00%, 69.33%]`)**, MRR **`0.6548`**, nDCG@10 **`0.6752`**, Moment Hit@1@0.5 **`72.73%`**, Moment nDCG@5 **`0.5121`**, Pure-ASR FP **`0/2`**, Non-Neural p50/p95 **`71.37 / 169.05 ms`**.
  - `dev_full_859` (OOF `742/859`): Hit@1 **`43.77%`**, MRR **`0.5142`**, nDCG@10 **`0.5558`**, Non-Neural p50/p95 **`50.32 / 111.63 ms`**.
  - `heldout_test_201` (`clean_disjoint`): Hit@1 **`39.80%`**, MRR **`0.4805`**, nDCG@10 **`0.5291`**, Non-Neural p50/p95 **`47.51 / 89.48 ms`**.
- **Trạng thái các bước GPU Kaggle (`C1`, `C2`, `C3`):** Phần local B1–B6 và Bước 0 đã nghiệm thu; các bước phụ thuộc GPU Kaggle (`VideoPrism-LvT-B` `C1`, `P8-5` Florence-2 captions & spatial patches `B6/C2`, `VLM Top-5` `C3`) giữ ở trạng thái **chờ chạy Kaggle GPU**.
