# VIDENTRA — Kế hoạch nâng Video Hit@1 sau Phase 9

**Ngày 17/11/2026 · Code được rà soát `5e1f91b` · Benchmark tham chiếu `41473d90`.** Phạm vi là text-to-video retrieval trên gallery hỗn hợp của VIDENTRA, ưu tiên MSR-VTT. Kế hoạch gồm việc có thể làm bằng snapshot hiện tại và ba thử nghiệm Kaggle. Mọi ngưỡng cải thiện bên dưới là tiêu chí đề xuất cho vòng thử, kết quả chỉ công bố sau khi đo.

## 1. Hit@1 dưới 50% có đáng lo không?

**Có, nếu sản phẩm cần trả đúng video ở vị trí đầu.** A5 đúng khoảng 4 trên 10 query trên hai tập lớn. Trên MSR‑VTT held-out, 119/191 query trượt hạng 1. Nên đặt 50% làm mốc sản phẩm để theo dõi với cùng gallery và split, đồng thời đo số thắng/thua khi thay từng thành phần.

| Tập của run `41473d90` | Video Hit@1 | Cách đọc |
| --- | ---: | --- |
| `dev_150` | 93/150 = **62,00%** | Tập con gồm 100 MSR‑VTT, 22 QVHighlights, 28 AVC. Tỷ trọng QVH/AVC cao hơn dev đầy đủ. |
| `dev_full_859` | 376/859 = **43,77%** | MSR‑VTT chiếm 809/859 query và đạt **331/809 = 40,91%**. |
| `heldout_test_201` | 80/201 = **39,80%** | MSR‑VTT đạt **72/191 = 37,70%**. Held-out lịch sử đã được xem nhiều lượt, phù hợp làm kiểm tra hồi quy. |

Run này có receipt, hai file predictions kiểm lại khớp checksum. Provenance ghi checkout `a9f4367` với `Dirty=True` và `publish_reference=False`. Kết quả có giá trị làm mốc chẩn đoán. Khi chấm challenger và công bố reference mới, tái tạo baseline từ checkout sạch và freeze snapshot, code, preset và query IDs trước khi so sánh. [Báo cáo](benchmark/benchmark_report.md) · [Summary và metrics theo dataset](benchmark/benchmark_summary.json) · [Provenance](benchmark/provenance.json)

### So với nghiên cứu công bố

| Công trình | Kết quả tác giả báo trên protocol của họ | Điều có thể áp dụng |
| --- | --- | --- |
| CLIP4Clip `seqTransf` | **44,5% T2V R@1** ở 1k‑A, có huấn luyện MSR‑VTT theo Training‑9K. Bản mean pooling đạt 43,1% cùng thiết lập. | Giữ mean pooling làm đối chứng mạnh. Thử pooling cửa sổ trước khi trả chi phí thêm mô hình chuỗi. [Bài báo, Bảng 1](https://arxiv.org/pdf/2104.08860) · [Repo của tác giả](https://github.com/ArrowLuo/CLIP4Clip) |
| VideoPrism‑LvT‑B public | **50,1% T2V R@1** zero-shot trên MSRVTT‑1K theo bảng repo. Bản trong paper là 51,4%. | Dùng đúng checkpoint public mà worker đã pin. Thử như điểm độc lập trên pool cố định, rồi mới thử truy hồi toàn gallery. [Repo và bảng zero-shot](https://github.com/google-deepmind/videoprism#zero-shot-video-text-retrieval) · [Bài báo ICML](https://proceedings.mlr.press/v235/zhao24f.html) |
| InternVideo2 `stage2` 1B | **51,9% T2V R@1** zero-shot ở cấu hình tác giả công bố. Repo có môi trường và script MSR‑VTT. | Challenger dự phòng nếu VideoPrism không bổ sung được tín hiệu hoặc vướng môi trường. Kiểm model config và artifact riêng trước khi quyết định. [Model zoo chính thức](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/MODEL_ZOO.md) · [Bài báo](https://arxiv.org/abs/2403.15377) |

Khi mỗi query có một video đích, Hit@1 và R@1 cùng kiểm tra vị trí đầu. Để so trực tiếp cần cùng gallery, danh sách query, quy ước label và chế độ huấn luyện. VIDENTRA đang tìm trên **1.324 video hỗn hợp** và chấm các split nội bộ. Nếu cần so với paper, tạo bảng MSR‑VTT 1k‑A riêng bằng đúng 1.000 cặp gốc và gallery của protocol, giữ thêm bảng hỗn hợp làm chỉ số sản phẩm. `Within-Dataset` trong report hiện thu hẹp từ candidates đã sinh, nên để đối chiếu paper cần chạy lại truy hồi trên gallery 1k‑A ngay từ đầu. [MSR‑VTT gốc](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) · [Runner benchmark](scripts/run_validation_benchmark.py)

## 2. Điểm mất chính xác trong dự án

Đếm lại từ hai file evaluation của `41473d90`. `Visual` và `union` là điều kiện video đích có trong tập ứng viên tương ứng. `Shortlist@20` là thứ tự **trước LTR/Consensus**, Top-5 là danh sách **cuối** trả về. Báo riêng hai loại pool để không dùng nhầm trần của reranker.

| Kết quả chỉ trên MSR‑VTT | `dev_full_859` (n=809) | `heldout_test_201` (n=191) |
| --- | ---: | ---: |
| Video đích xuất hiện trong visual candidates | 681 | 159 |
| Video đích xuất hiện trong union visual + lexical/caption | 694 | 164 |
| Video đích vào shortlist trước rerank K=20 | 606 | 142 |
| Video đích đứng hạng 1 cuối | 331 | 72 |
| Video đích đứng hạng 2–5 cuối | 157 | 36 |
| Video đích đứng hạng 6–10 cuối | 56 | 21 |
| Video đích ở ngoài Top-10 cuối | 265 | 62 |
| **Sai hạng 1 dù video đích đã vào union** | **363/478 lỗi** | **92/119 lỗi** |
| **Video đích vắng khỏi union** | **115/478 lỗi** | **27/119 lỗi** |

- **Ưu tiên xếp hạng.** Trên held-out MSR‑VTT, trần của một reranker chỉ reorder đúng Top-5 hiện tại là `(72+36)/191 = 56,54%`. Để chạm 50% cần `96/191`, thêm **24 hit ròng**. Nếu chỉ dùng Top-5 và giữ mọi hit cũ, cần cứu 24/36 ca đang ở hạng 2–5. Vì vậy dùng Top-5 để kiểm chứng VLM, đồng thời chuẩn bị K=10/20 với VideoPrism hoặc scorer rẻ hơn để có không gian cải thiện.
- **Tăng coverage song song.** Stage 2 lấy ngân sách **120 frame**, trung bình chỉ **46,46 video khác nhau** trên MSR‑VTT dev. Cần đo việc nhiều frame cùng video tiêu tốn ngân sách và đối chứng retrieval theo video. Nếu chỉ mở rộng frame depth, candidate set có thể tăng chi phí ở Stage 4–6, nên giữ ngân sách video cố định trong phép thử chính.
- **Loại lỗi để kiểm tay.** Trong 478 miss MSR‑VTT dev, taxonomy đánh dấu 281 `F4_FINE_GRAINED_VISUAL_CONFUSION`, 128 `F1_VISUAL_CANDIDATE_MISS`, 69 `F2_LEXICAL_DISTRACTOR_OVERTHROW`. Nhãn tự động hỗ trợ chọn mẫu review. Ví dụ 128 visual miss và 115 union miss là hai điểm đo khác nhau, vì lexical có thể cứu một phần. Xem 40–60 query tune, lấy mẫu theo nhóm nguồn và đủ trường hợp role/action, số lượng, vật thể, cảnh chung, OCR/ASR. Chấm ứng viên ẩn tên hệ thống, giữ qrels gốc và lưu audit relevance bổ sung riêng cho mô tả chung có nhiều video phù hợp.
- **Các bản sửa đã đo.** Step 0 trên cùng snapshot cô lập Consensus, tăng toàn dev từ `357 → 376/859`, held-out từ `78 → 80/201`. LTR 24d với blend 0,015 có 0 lần đổi Top-1 trong ablation. `extractive-v2` hòa rules về Hit@1 dev, LoRA thấp hơn một hit. Lấy `a5_full` + rules làm mốc và tập trung thêm tín hiệu video thay cho tune Qwen parser.

[Dự đoán dev đã lưu](benchmark/dev_full_evaluation.jsonl) · [Dự đoán held-out đã lưu](benchmark/heldout_test_201_evaluation.jsonl) · [Kết quả Step 0](docs/PROGRESS.md)

## 3. P0 — Sửa điểm rơi candidates và thử rerank nhẹ trong lúc chờ Kaggle

### P0.1 Stage 2 chọn exact Top-K theo điểm đã trộn

Trong `UnifiedMultimodalSearchPipeline`, nhánh single-clause với hai ma trận đã align tính cả `sims_768_all` và `sims_864_all`, sau đó lấy union của Top-120 từng vector điểm rồi mới trộn. Một frame tốt ở cả hai thang có thể đứng ngoài hai danh sách này dù điểm trộn cao nhất. Fixture K=1 đã tái hiện. Ba frame A/B/C có `s768=(0,90; 0,74; 0,89)`, `s864=(0,30; 0,95; 0,88)`. Top-1 từng nhánh là A và B, còn `0,70*s768+0,30*s864` cho C điểm **0,887**, cao nhất. Mức ảnh hưởng trên dữ liệu thật cần đo khi triển khai.

1. Dùng `mixed = 0.70 * sims_768_all + 0.30 * sims_864_all` trên tất cả row hợp lệ rồi chọn Top-K từ `mixed`. Nhánh multi-clause đã có ma trận blended đầy đủ trước top-k theo hàng. Dùng chung logic chọn theo điểm, giữ quy tắc clause/SE-Viterbi. Với mapping thiếu hoặc repo phải dùng đường FAISS fallback, giữ behavior hiện hành và trace `retrieval_path`.
2. Áp dụng valid-mask theo snapshot và filter trước khi lấy Top-K. Loại vector/score lỗi, dùng tie-break `(-score, faiss_id)` ổn định tại cutoff. Trả cùng schema của visual candidates và giữ `pts_time`/source frame ID của row gốc.
3. Regression cho fixture trên, tie ngay cutoff, ít hơn K frame hợp lệ, filter loại frame hạng đầu, snapshot thiếu một bank và parity single-/multi-clause. So legacy vs exact trên tune với **cùng blend, K, gallery và stages sau**, log số frame/video mới đưa vào, số video đích được cứu hoặc mất và latency Stage 2. Sau đó mới đo tương tác với Stage 4–6.

[Mã Stage 2](src/videntra/retrieval/unified_pipeline.py#L2631)

### P0.2 Shortlist theo video, giữ anchors cho Stage 4–6

Với điểm raw/đã trộn của mọi frame, group theo `video_id`. Với cùng ngân sách video, thử xếp hạng theo max frame, theo mean top-2 **cách nhau theo PTS** và theo RRF của hai thứ hạng này. Giữ 1–2 anchors mỗi video và giữ phần frame-local hiện tại cho localization. Sau đó thử reserve khoảng 10% slot cho nhánh clip mean hiện có, thêm vào trước khi cắt pool rerank. Mỗi biến thể thay một yếu tố và giữ OCR/ASR độc lập với visual.

- **Cách tích hợp.** Mở rộng kết quả `ClipBranchIndex`/visual proposal thành danh sách unique video có `video_id, branch_rank, anchor_fid, pts, score`. Stage 4 dùng anchors để tạo visual clusters. FTS5/dense OCR, ASR và caption vẫn tìm trên corpus và có quota riêng. Chốt ngân sách Top-20 cho reranker, thử K=10/20/40 để lập đường quality–latency.
- **Tiêu chí.** Báo Visual distinct videos, union coverage, Recall@5/10/20/50, novel-rescue và displacement, QVH/AVC, Moment@0.5 và p95. Mốc thử nghiệm hữu ích là **Recall@20 tăng ≥3 pp** trên tune với K cố định, Hit@1 và guardrails đi qua confirm. Mốc hướng tới sau khi kết hợp tín hiệu là Recall@20 ≥80%. Stage 2 exact có thể được nhận như bản sửa tính đúng của ranking khi parity và các guardrail đạt.
- **Đối chứng.** Cả hai ý tưởng pooling và video rescoring có thể chạy với embeddings 768d hiện tại, so với mean pooling theo CLIP4Clip. `compute_temporal_pool_score()` đã có `mean_top2` và `attention`. Trong pilot rank-only chỉ đổi scorers, giữ nguyên moment anchors để biết gain đến từ video score. Nếu cần thăng hạng theo độ phủ nhiều mệnh đề, dùng chính query clauses đã ground và so điểm clause×frame trong window của ứng viên. X-CLIP là nguồn cho ý tưởng tương tác ở nhiều mức, còn ở đây dùng model và embeddings đang có. [Scorer hiện có](src/videntra/retrieval/alignment.py#L500) · [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) · [X-CLIP paper](https://arxiv.org/abs/2207.07285) · [X-CLIP repo](https://github.com/xuguohai/X-CLIP)

### P0.3 Chẩn đoán lexical và caption trước khi thêm luật

1. Capture Top-K trước/sau Stage 5 với `visual_score, ocr_rank, asr_rank, caption_rank, exact_span_match, modality_intent, evidence_status`. Đối chiếu matched label chỉ trong evaluator. Với query thuần nhìn, thử loại **score bonus** OCR/ASR trong replay, vẫn giữ candidate đã được lexical tìm. Với query quote/tên riêng/speech, giữ retrieval và evidence scoring riêng. So number of flips và F2 trên các slice đã khóa.
2. Khi sắp mở P8-5, tách branch rank caption khỏi `rrf_vis`. Hiện `keyframe caption hit` có thể append vào `visual_candidates`, kể cả một video mới có visual cosine từ lookup, rồi được xếp và cấp `rrf_vis` khi Stage 4 tạo clusters. Thêm `branch_provenance` và điểm `rrf_caption` cho caption-only seeds. Kiểm phủ khi candidate bị filter, tách giới hạn mỗi video của caption lexical/dense, rồi dùng fusion đã normalize. Kiểm trường hợp caption-only không được tính như một hit visual độc lập, caption có ảnh nhưng thiếu dấu hiệu hành động, và OCR/ASR chỉ được credit khi evidence tương ứng.
3. Đo hubness trước khi chạy QB-Norm. Đếm phân phối số lần video xuất hiện trong Top-1/5 trên tune, so phần đuôi và nhóm video lặp thường xuyên. Nếu có nhiều lỗi vì một số video phổ thông luôn lên cao, thử inverted softmax dùng query bank freeze từ **train/tune có source tách nhóm**. Dùng log-sum-exp cho ổn định. Giữ confirm/test tách khỏi bank và báo cả độ lệch chất lượng theo tập. [QB-Norm, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.pdf)

### P0.4 Kết quả cần có trước khi tốn thời gian GPU

- Mỗi query xuất riêng `before_base_fusion`, `before_rerank`, `after_rerank` với danh sách IDs và lý do vào pool, tách evaluator rows có gold. Freeze A5 pool và clip pool từ cùng lần capture. `alpha=0` phải tái tạo đúng top-1 và thứ tự base pool trong replay.
- Đo lại baseline A5 trên dev với rules, lock `tune/confirm` theo source và config/hash, giữ gallery 1.324 video. Khi candidate policy đổi, cần report per-stage recall và chấm engine end-to-end. Khi chỉ đổi scorer, bắt buộc giữ pool giống nhau trong paired comparison.
- Khi muốn xác minh baseline hiện hành, chạy lệnh dưới từ project root. Đường `work/experiments` dành cho run mới. Lệnh này chạy A5 hiện có, P0.1–P0.3 cần thêm cờ challenger khi triển khai.

```powershell
$py = '.\.venv\Scripts\python.exe'
$hitPlanRun = 'work/experiments/hit_at_1_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$env:VIDENTRA_LLM_ENABLED = '0'
& $py -B scripts/run_validation_benchmark.py --splits dev_full_859 --a5-preset a5_full --out-dir "$hitPlanRun/baseline"
& $py -B scripts/capture_msr_candidate_pools.py --split dev_full_859 --a5-preset a5_full --rules-only --max-k 20 --out-dir "$hitPlanRun/pools" --eval-dir "$hitPlanRun/evaluation"
& $py -B scripts/run_clip_vp_replay.py --pools-dir "$hitPlanRun/pools" --pool a5 --k 20 --alpha 0 --out-dir "$hitPlanRun/a5_k20"
```

## 4. C1 — VideoPrism-LvT-B trên Kaggle, ưu tiên GPU đầu tiên

**Trạng thái đã kiểm 17/11.** `kaggle kernels status` trả `CANCEL_ACKNOWLEDGED` cho `lvhuylong/videntra-msr-videoprism-dev-20261115`. `kaggle kernels files` không liệt kê output. Workspace chưa có shard thật. Lịch sử ghi bản v1 dừng vì import `faiss`, launcher local đã thêm pin FAISS. Lần tới chạy smoke mới và thu log JAX/GPU để biết có thể tiếp tục. [Notebook cũ](https://www.kaggle.com/code/lvhuylong/videntra-msr-videoprism-dev-20261115) · [Audit và lệnh replay cũ](benchmark/msrvtt_clip_qwen_videoprism_audit.md)

### Smoke có thể kiểm chứng

1. Đóng gói bằng `prepare_clip_vp_kaggle_bundle.py` từ pools rules A5, chỉ gửi query text, IDs, manifest candidate và media có quyền sử dụng. Query dùng cho smoke chọn bằng query ID/source đã freeze, không chọn theo việc A5 thắng hay thua. Với media MSR‑VTT đang mount sẵn trên tài khoản, kiểm hash từng file so snapshot. Khớp ID, relative path và SHA-256 của từng video trong manifest; nếu trùng basename thì fail và sửa mapping. Worker dùng 16 frame theo time bins và presentation ordinal, giữ chính sách PTS của dự án.
2. Tạo notebook riêng với **pilot 6 video × 4 query**, dừng sau smoke và giữ log. Pin source `4f831e332b80d6e77bdd43882206f23aba2a67a7` của VideoPrism và revision weights `0119cca7d1fe61aa1d5718e1f71605703e1562e7` đang dùng trong worker. Checkpoint và SentencePiece kiểm SHA-256 theo constants. Ghi runtime package versions, thiết bị JAX, tốc độ/VRAM đo được và frame decode errors. Đặt memory preallocation theo launcher; khởi đầu `video_batch_size=1`. Fail nếu JAX rơi về CPU cho lượt GPU. [Worker hiện hành](scripts/kaggle_videoprism_worker.py) · [Model card checkpoint](https://huggingface.co/google/videoprism-lvt-base-f16r288)
3. Nghiệm thu đủ `6` video và `4` query, `dry_run=false`, errors rỗng và đủ 4 file `video_vectors.npy`, `query_vectors.npy`, `video_rows.jsonl`, `query_rows.jsonl`. Đối chiếu row ID, dim 768, finite/nonzero, hash file và hash câu gốc. Worker có thể trả `SUCCESS` cùng `errors`, nên gate cần so expected counts và log từng miss. Nếu scale, chia shard video deterministic, chỉ resume shard hoàn tất đã verify; `--resume` hiện kiểm receipt đã xong. Giữ video vectors tái dùng, encode query text cùng model revision, kiểm toàn bộ receipt trước lúc merge.

### Chấm để tách rerank và recall

| Cặp đo | Giữ cố định | Điều muốn biết |
| --- | --- | --- |
| A5 K=10/20, `alpha=0` → A5 + VP | Cùng IDs và thứ tự base trước rerank | VP có đưa target đã ở trong pool lên hạng đầu? |
| Clip K=20, `alpha=0` → clip + VP | Cùng clip pool, cùng devices/offline artifacts | VP có tận dụng ứng viên ở slot reserve? |
| A5+VP → A5+VP với thêm candidate từ VP | Cùng K cuối và cùng scoring | VP branch có cứu mục tiêu ngoài union? Chỉ làm sau khi có vector phủ đủ gallery đã chọn. |

Weighted RRF hiện dùng công thức `(1−alpha)/(60+r_A5) + alpha/(60+r_VP)`. Sweep nhỏ `alpha ∈ {0,25; 0,50; 0,75; 1,00}` trên tune, chọn một cấu hình. Với slot reserve ở cuối, `alpha≤0,5` và tie ưu tiên A5 khiến slot cuối không vượt được A5 hạng 1 trong full-pool ranking. Nếu shard chỉ phủ MSR‑VTT, `preserve-unscored` giữ vị trí video thiếu vector và rerank các vị trí coverage. Query có top-1 A5 thiếu vector sẽ không đổi top-1; báo `top1_change_eligible` cùng full/partial/fallback và số query cả tập. Báo alpha=0 công khai và chấm K=20 riêng.

```powershell
# Tiếp tục trong phiên PowerShell đã tạo baseline. VIDENTRA_VP_SHARD chỉ tới shard thật đã verify.
& $py -B scripts/run_clip_vp_replay.py --pools-dir "$hitPlanRun/pools" --pool a5 --k 20 --alpha 0.75 --vp-dir $env:VIDENTRA_VP_SHARD --vp-coverage-policy preserve-unscored --out-dir "$hitPlanRun/vp_a5_k20"
& $py -B scripts/compare_msr_runs.py --baseline "$hitPlanRun/a5_k20" --challenger "$hitPlanRun/vp_a5_k20" --labels "$hitPlanRun/evaluation/clip_vp_dev_full_859_labels.jsonl" --split-lock "$hitPlanRun/evaluation/clip_vp_dev_full_859_split_lock.json" --require-same-base-pool --out-dir "$hitPlanRun/compare_vp"
```

Lệnh minh họa một mức alpha. Trước lúc dùng kết quả để chọn tham số, truyền `--filter-manifest` và `--filter-split tune` với lock sẵn có. Với bản chốt đánh giá `confirm` riêng và giữ các output riêng. Trong thí nghiệm tăng coverage, truy vấn toàn bank VP bằng cosine, hợp nhất video IDs trước khi cap, thêm anchors PTS từ visual timeline của từng video và báo số mới vào union/top-K. [Bundle](scripts/prepare_clip_vp_kaggle_bundle.py) · [Replay](scripts/run_clip_vp_replay.py) · [Paired evaluator](scripts/compare_msr_runs.py)

## 5. C2 — P8-5 dùng caption để cứu candidate và patches để phân biệt chi tiết

**Lý do thử.** Cap4Video dùng điểm query–caption để bổ sung query–video. VIDENTRA có worker Florence-2 và E5 để thử riêng nhánh này. Caption sinh từ ảnh có thể thấy vật thể/thuộc tính mà embedding tổng quát bỏ qua; các tuyên bố về hành động qua nhiều frame cần tín hiệu video. [Cap4Video paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.html) · [Repo Cap4Video](https://github.com/whwu95/Cap4Video) · [Model card Florence‑2](https://huggingface.co/microsoft/Florence-2-base)

1. Tạo package pilot **50–100 keyframe** theo video, dataset, thời lượng và khóa `faiss_id`, `keyframe_id`, `video_id`, PTS, hash ảnh. Chọn frame media có quyền, dùng `--skip-gt`. Ưu tiên pilot xác minh GPU và chất lượng caption. Subset pilot có thể không thỏa loader spatial patches dạng dense toàn snapshot, nên kiểm ma trận/mapping và mode riêng trước khi chấm full.
2. Gọi worker `--mode p85 --keyframes-manifest … --images-dir … --out-dir …`. Dùng output dir khác cho pilot và full. Truyền đường dẫn FAISS tương ứng qua wrapper/job và kiểm receipt. Lưu pin SigLIP, Florence, E5, counts, tỷ lệ caption rỗng/lặp và lỗi decode. `p85_receipt.json` v1.1 phải phân biệt coverage subset/full và kiểm khớp checksum artifact.
3. Ingest vào **bản sao** DB/snapshot. `--staging-dir` hiện là alias cho thư mục **input** của ingest; đích xác định bởi `--db-path` và `--processed-dir`. Dùng `--dry-run` trước khi ghi. Bảo toàn file FAISS tương ứng trong snapshot staging để preflight đối chiếu hash. Kiểm kênh `ready` sau khi load qua SSOT. Kênh nào thiếu artifact tiếp tục đánh dấu `missing_artifact` trong báo cáo.
4. So **caption lexical-only**, **caption dense-only**, **spatial-only** rồi tổ hợp đã chọn. Với lexical/dense, đánh giá recall trên cùng K và số unique video mới. Giới hạn 1–2 caption anchors/video cho union và giữ full captions khi xem chi tiết. Với patches, đo riêng trên fixed pool, ưu tiên lát cắt có chi tiết nhỏ, số lượng hoặc vị trí. Đo guardrails pure-ASR/OCR, moment và p95. Thu thập caption sai/hallucinated để hiểu các top‑1 regression.
5. **Sửa đường dẫn runner trước khi chấm staging.** `MultimodalBenchmarkRunner` truyền DB/FAISS mặc định từ `data/processed/` vào SSOT. CLI chưa expose path này, và `input_paths` dùng cùng đường dẫn cố định. Thêm `--db-path` / `--faiss-path` vào `run_benchmark()` và CLI, resolve một lần rồi truyền vào runner và provenance. Hash cả patch tensor/map/receipt, caption FAISS/manifest và DB staging. Loader optional artifacts đọc cùng thư mục với DB. Dùng test hai snapshot có channel status khác nhau để chứng minh chọn đúng snapshot và ghi đúng hash.
6. Benchmark runner đã có `--disable-vlm-caption-lexical`, `--disable-vlm-caption-dense` và `--disable-spatial-patches`. Verify snapshot thực sự đọc từ bản staging theo config/metadata của run trước khi tin kết quả. Mục tiêu C2 là thêm **recall K=20** và giữ candidate tốt cũ. Nếu recall tăng mà Hit@1 giảm, giữ branch như nguồn proposal có quota nhỏ rồi đem pool đó sang C1/C3 để đánh giá.

[Mã worker](scripts/kaggle_multimodal_offline_worker.py) · [Đóng gói keyframes](scripts/package_kaggle_inputs.py) · [Ingest P8‑5](scripts/ingest_kaggle_offline_results.py)

## 6. C3 — Reranker đa phương thức chuyên dụng trên Top-5

**Ưu tiên `Qwen3-VL-Reranker-2B` cho pilot.** Đây là model chấm relevance query–document có hỗ trợ cả document video, khác vai trò của Qwen2.5-1.5B đang dùng để phân rã câu. Repo hỗ trợ đầu vào là chuỗi frame. Khởi chạy qua Transformers trong môi trường Kaggle riêng, pin model, processor, wrapper và dependencies rồi kiểm thiết bị thực tế. [Model card Apache‑2.0](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · [Repo và định dạng input](https://github.com/QwenLM/Qwen3-VL-Embedding#model-input-specification) · [Technical report](https://arxiv.org/abs/2601.04720)

1. Đóng băng A5 Top-5 và lấy **30–50 query tune** theo seed đã ghi, kèm guardrails OCR/ASR/QVH/AVC. Đầu tiên smoke 3–5 query, FP16 và batch 1 với attention backend khả dụng trên GPU thực tế. Mỗi cặp query/video đưa 4 frame xếp theo presentation order, sau đó thử 8 frame và log pixel budget/peak VRAM. Policy frame uniform theo PTS hoặc quanh **A5 anchor do engine dự đoán**. Khi truyền danh sách frame, giới hạn số frame ngay từ sampling. GPU pilot ghi rõ phiên bản runtime, p50/p95, throughput, OOM, số pair có score và fallback.
2. Hợp đồng cache `query_text_sha256, video_id, media_sha256, snapshot_sha256, frame_ordinals, frame_pts_time, sampling_recipe, model/processor_revision, code/prompt_sha256, score, status`. Không đưa rank A5, GT IDs hay GT timestamps vào input model. Lưu output relevance theo mô hình, dùng như tín hiệu xếp hạng. Test đảo thứ tự candidates và đối chiếu score từng pair để chặn ảnh hưởng do thứ tự/batching.
3. Trên **cùng Top-5**, chấm A5, VP, VL rerank thuần và VL + A5 rank fusion. So `Hit@1` và `flip_precision = số hit được cứu / số query đổi Top-1`, kèm số hit mất và số query đổi giữa hai kết quả đều sai. Trật tự các slot thiếu score giữ nguyên; nếu top-1 không được chấm, trả baseline cho top-1 và ghi reason. Để tiết kiệm GPU ở lần scale, thử gate theo A5 margin và intent, chọn ngưỡng tune rồi khóa trước confirm. Tính cả query fallback trong metric chính.
4. Nếu Top-5 cho gain nhất quán nhưng trần thấp, thử **VP trên Top-20 → VL trên Top-5 của VP**, so với **VP trên Top-20** để đo phần tăng do VLM. Nếu 4 frame thiếu tín hiệu chuyển động, dùng 8 frame theo PTS và kiểm query về thứ tự. Nếu chưa đo budget trên RTX 3060 6 GB, giữ VLM là nhánh offline experimental, ghi latency local là chưa đo. Mọi online/Accurate mode cần benchmark phần cứng đích riêng.

## 7. Khi nào huấn luyện và cách chấp nhận kết quả

### Điều kiện để train ranker hoặc thích nghi encoder

- Khi đã có feature mới đo được gain, capture feature pre/post với cờ missingness, baseline scores, ranks VP, caption và VLM. Dùng cùng Top-20 để so RRF, pairwise linear và LightGBM per-fold trong `train_ltr_ranker_cv.py`. Tách fold theo nguồn, dùng nội bộ tune để chọn hyperparameters và đặt confirm bên ngoài fit. Đo cùng chính sách scorer runtime; join relevance trong trainer/evaluator sau khi sinh pool. Theo dõi log pool có target, false-negative và score distribution. LTR 24d hiện tại có thể dùng làm đối chứng, tránh đưa `a5_full` và `a5_no_ltr` vào hai tập khác nhau.
- Nếu **coverage** vẫn thấp, thử mở rộng sampling theo cảnh hoặc projection head nhỏ từ embeddings đã có trên train tách source. So với frozen backbone trên cùng gallery và query. Dùng caption/media có quyền và chống overlap với 1k‑A hoặc test nội bộ. Caption gold của query đánh giá không đưa vào index caption. Chỉ train lại full encoder sau khi có bằng chứng projection và video-native baseline chưa đáp ứng mục tiêu.

### Thứ tự giao việc và cổng nghiệm thu

| Thứ tự | Deliverable | Điều kiện tiến tiếp |
| --- | --- | --- |
| **1 · CPU** | Fix exact blend, bản trace trước/sau Stage 5, fixed-pool capture, review lỗi theo nguồn. | Parity `alpha=0`, cùng snapshot/recipe, source gate đạt, baseline có receipt mới. |
| **2 · GPU C1** | Smoke VideoPrism mới và replay fixed pool K=10/20. | Shard thật đủ hash và counts. Đo Hit@1 trên toàn slice với fallback. Chỉ scale coverage sau khi thấy tín hiệu thêm. |
| **3 · GPU C2** | P8‑5 staging và ablation từng branch. | Recall@20 tăng ở cùng K; log rescue/displacement, kênh `ready` và kiểm guardrails. Công việc đóng gói có thể chuẩn bị trong khi chờ C1. |
| **4 · GPU C3** | Relevance cache và replay Qwen3‑VL trên Top-5, sau đó cascade nếu pilot tốt. | Gain tăng thêm trên fixed pool hoặc trên cùng cascade; net wins dương trên tune/confirm và đo chi phí/coverage đủ. |
| **5 · Xác nhận** | Chọn tối đa một challenger chính, freeze rồi chạy trên nguồn mới. | Mục tiêu vòng đầu **≥+3 pp MSR‑VTT Hit@1** và cận dưới CI 95% ghép cặp >0, với ngân sách latency đã khóa, QVH/AVC và OCR/ASR/moment đạt guardrails. Mốc sau đó **≥50% trên gallery hỗn hợp** với split và cỡ mẫu công bố. |

- Dùng paired bootstrap theo `source_group` cho delta, báo `n_query, n_group, W/L, CI`. `dev_150` thuộc dev đầy đủ, còn `heldout_test_201` lịch sử dùng hồi quy. Khi confirm/held-out từng được xem hoặc dùng chọn ngưỡng, cần thêm split source-disjoint chưa xem để xác nhận. Nếu CI rộng, ghi thiếu cỡ mẫu và thu thêm nguồn theo protocol. Muốn công bố “đã vượt 50%” với độ tin cậy thống kê, yêu cầu cả cận dưới CI của Hit@1 tuyệt đối trên tập xác nhận vượt 50%.
- Freeze qrels và gallery trước thử. Báo coverage riêng cho query có target ngoài snapshot. Ghi `Hit@1` chính trên global gallery và theo dataset, `Hit@5/10`, `MRR@10`, `nDCG@10`, recall qua từng stage, Moment@0.5/nDCG, số sample pure-ASR/OCR, cold/warm p50/p95 và tài nguyên. Giữ artifact guard và split lock. Khi một nhánh chưa có score/artifact, report coverage và fallback; các lượt smoke/synthetic để debug.
- Sau một benchmark nghiệm thu, đồng bộ `benchmark/` và `README.md` từ run đã verify. Lưu mã commit, run ID và hashes. Đưa `--publish-reference` vào lượt chạy từ checkout sạch. CPU và GPU báo riêng, mục tiêu non-neural hiện là `20/45 ms p50/p95`; pilot nghiên cứu cần chỉ rõ nếu chưa đạt.

## 8. Phạm vi xác minh của lượt lập kế hoạch

- Đã duyệt nguồn nghiên cứu công khai và repo upstream, đọc SSOT/capture/worker/ingest, kiểm checksum hai JSONL của `41473d90`, đếm lại các rank/coverage và chạy fixture exact-blend. Đã truy vấn trạng thái và file output của notebook VideoPrism cũ bằng Kaggle CLI. Số chất lượng là từ run đã lưu.
- Chạy `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/retrieval/test_clip_branch.py tests/evaluation/test_clip_vp_replay.py tests/indexing/test_videoprism_kaggle_pipeline.py tests/indexing/test_kaggle_offline_pipeline.py` trên Windows CPU. **34 passed trong 7,08 giây**.
- Lượt này là nghiên cứu và lập kế hoạch. Thay đổi retrieval, benchmark sinh predictions mới và các thí nghiệm Kaggle ở trên chờ lượt triển khai.

## 9. Nhật ký triển khai cục bộ & kết quả thực nghiệm CPU (17/11/2026)

### Đã triển khai & kiểm thử (`604/604` pytest PASS)

- **P0.1 Exact Blended Stage 2 (`src/videntra/retrieval/unified_pipeline.py`):** Bổ sung `_build_stage2_valid_mask`, `_select_topk_valid_frames` (tie-break `(-score, fid)` ổn định ngay tại cutoff) và mặc định `stage2_selection_mode="exact"` tính `0.70*s768 + 0.30*s864` trước khi cắt Top-K; bảo toàn đường `"legacy_union"` (flag) và `"faiss_fallback"`.
- **P0.2 Video-Level Proposal (`src/videntra/retrieval/unified_pipeline.py`):** Thêm `_build_video_level_visual_proposals()` với cờ `video_proposal_mode` (`frame_topk`, `video_max`, `video_mean_top2_pts`, `video_rrf_max_top2`), `video_proposal_top_n`, `video_proposal_anchors_per_video`, `video_proposal_min_pts_gap`, `video_proposal_clip_reserve_ratio`.
- **P0.3 Provenance, Pure-Visual Gate & Hubness (`unified_pipeline.py`, `visual.py`):** Tách `caption_only_candidates` khỏi nhánh `visual` (`rrf_vis == 0.0`, `"visual" not in branches`, `rrf_caption` tách riêng), giới hạn $\le 2$ anchors/video/nguồn, xác minh SHA-256 `faiss_e5_caption_384.bin` & `e5_caption_manifest.json` qua `p85_receipt.json`; thêm `_classify_modality_intent()`, cờ `disable_pure_visual_lexical_bonus`, `compute_hubness_report()` & `apply_querybank_norm()`.
- **P0.4 Traces & Fixed-Pool Capture (`unified_pipeline.py`, `capture_msr_candidate_pools.py`):** Xuất `stage_traces` (`before_base_fusion`, `before_rerank`, `after_rerank`) và kiểm tra parity `alpha=0` (`0` thay đổi Top-1).
- **Sec 5.5 Staging Benchmark Paths (`scripts/run_validation_benchmark.py`):** Thêm `--db-path`, `--faiss-path`, `--model-dir` cùng các cờ P0.1–P0.3, resolve sidecar theo `db_path.parent` và băm đầy đủ `spatial_patches_*` / `p85_receipt.json` / `faiss_e5_*`.
- **C1/C2/C3 Local Tooling:** Thêm `--max-queries`/`--max-videos` & kiểm lỗi trùng `relative_path`/basename (`prepare_clip_vp_kaggle_bundle.py`), `expected_video_count`/`expected_query_count` + `require_zero_errors=True` (`kaggle_videoprism_worker.py`), `--stratified-pilot` + `image_sha256` (`package_kaggle_inputs.py`), `src/videntra/ranking/vl_reranker_cache.py` + cascade `--vl-cache`/`--vl-top-m`/`--vl-mode`/`--vl-beta` (`run_clip_vp_replay.py`) + `flip_precision`, `rescued_hits`, `lost_hits`, `both_wrong_swaps` (`compare_msr_runs.py`).

### Kết quả thực nghiệm CPU P0 trên `dev_full_859` (`rules-only`, `a5_full`, `K=20`)

| Thí nghiệm | Slice | Hit@1 (Base $\to$ Chal) | W / L | MRR@10 | Recall@20 | QVH Hit@1 | AVC Hit@1 | Quyết định |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `legacy -> exact` | `overall` (`859`) | `43.77% -> 43.77%` | `0 / 0` | `0.5142 -> 0.5146` | `76.13%` | `86.36%` | `92.86%` | **Chấp nhận mặc định** (`Hit@5`: `534 -> 535`) |
| `legacy -> exact` | `tune` / `confirm` | `45.09% -> 45.09%` / `40.70% -> 40.70%` | `0/0` / `0/0` | `0.5242` / `0.4921` | `76.21% (+1)` / `75.97% (-1)` | `93.33%` / `71.43%` | `95.00%` / `87.50%` | Giữ nguyên guardrails |
| `exact -> v_max` | `overall` (`859`) | `43.77% -> 41.44%` | `10 / 30` | `0.5002` | `75.79%` | `68.18%` | `92.86%` | Không chọn mặc định (tụt QVH `-18.18 pp`) |
| `exact -> v_top2` | `overall` (`859`) | `43.77% -> 41.33%` | `10 / 31` | `0.4996` | `75.67%` | `68.18%` | `92.86%` | Không chọn mặc định |
| `exact -> v_rrf` | `overall` (`859`) | `43.77% -> 41.44%` | `10 / 30` | `0.5002` | `75.55%` | `68.18%` | `92.86%` | Không chọn mặc định |
| `exact -> v_top2_clip10` | `overall` (`859`) | `43.77% -> 41.33%` | `10 / 31` | `0.4995` | `75.79%` (`tune 76.54%`) | `68.18%` | `92.86%` | Giữ flag nghiên cứu |
| `exact -> no_lex_pv` | `overall` (`859`) | `43.77% -> 42.84%` | `8 / 16` | `0.5058` | `75.90%` | `86.36%` | `75.00%` | Giữ mặc định `False` (tụt AVC `-17.86 pp`) |

- **Hubness trên `exact` (`dev_full_859`):** `unique_top1_videos = 571/859`, `top1_gini = 0.2738` (`tune = 0.2268`, `confirm = 0.1281`), `topk_gini = 0.4576`, `topk_max_freq = 54`.
- **Benchmark tham chiếu mới (`Run ID 0ee1a4317b97474a8dda740f8c7bc5a1`, đồng bộ [`benchmark/`](benchmark/)):** `dev_150` Hit@1 **`62.00%`**, MRR **`0.6571`**, nDCG@10 **`0.6770`**, Moment@0.5 **`72.73%`**, Pure-ASR FP `0/2`; `dev_full_859` **`43.77%`**, MRR **`0.5146`**; `heldout_test_201` **`39.80%`**, MRR **`0.4805`**, `leakage_free=True`, `gate_passed=True`.
- **Chờ chạy Kaggle GPU:** `C1` (VideoPrism-LvT-B 6v×4q smoke & full shard), `C2` (`P8-5` Florence-2/E5/patches staging), `C3` (`Qwen3-VL-Reranker-2B` Top-5 / VP→VL cascade).

