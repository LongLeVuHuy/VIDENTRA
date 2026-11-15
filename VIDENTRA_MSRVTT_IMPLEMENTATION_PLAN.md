# Kế hoạch triển khai và đo Hit@1 MSR-VTT cho Antigravity

Cập nhật ngày 15/11/2026. Điểm xuất phát là commit `56a79c7`. Mục tiêu là đo đóng góp riêng của clip reserve, VideoPrism‑LvT‑B và phân rã query bằng Qwen2.5‑1.5B‑Instruct. Chỉ đề xuất thay cấu hình mặc định khi kết quả ghép cặp đã qua các bước dưới đây.

## 1. Mốc kiểm chứng và thứ tự làm

Snapshot audit có 1.324 video trong gallery. `dev_full_859` có 809 query MSR-VTT. A5 dùng LTR theo fold OOF cho 742/859 query toàn dev, trọng số cuối cho phần còn lại theo provenance gate, và temporal rerank `lambda=0`.

| Đã đo trên MSR-VTT dev | A5 | Clip reserve | Kết luận |
| --- | ---: | ---: | --- |
| Hit@1, K=10 và K=20, `alpha=0` | 312/809 | 312/809 | 0 thắng, 0 thua |
| Recall@10 | 544/809 | 555/809 | 15 ca được, 4 ca mất |
| Recall@20 | 606/809 | 611/809 | 9 ca được, 4 ca mất |

Kaggle CLI kiểm tra lúc 19:47 UTC+7 ngày 15/11/2026. Cả hai notebook chưa có tệp output để tải.

| Notebook | Trạng thái | Kết quả có thể xác minh |
| --- | --- | --- |
| VideoPrism v2 | `QUEUED` | Đang chờ GPU. Lượt v1 dừng do thiếu FAISS và v2 đã bổ sung `faiss-cpu==1.13.2`. Chờ shard thật để chấm R01 và R11. |
| Qwen2.5-1.5B | `RUNNING` | Log mốc 380/809 query ghi 343 plan qua validator, 17 fallback và 20 query thuộc tier ngắn. Chờ cache hoàn chỉnh để chấm Q00 và Q10. |

Mốc Hit@1 đã đo vẫn là **312/809 (38,57%)** của A5 và clip reserve với `alpha=0`. Chênh lệch Hit@1 của VideoPrism và Qwen đang chờ artifact. [Audit đã chạy](benchmark/msrvtt_clip_qwen_videoprism_audit.md) · [Qwen trên Kaggle](https://www.kaggle.com/code/lvhuylong/videntra-msr-qwen15b-dev-20261115) · [VideoPrism trên Kaggle](https://www.kaggle.com/code/lvhuylong/videntra-msr-videoprism-dev-20261115)

1. **P0, làm ngay.** Khóa các inputs benchmark, kiểm tra artifact và bảo đảm evaluator nhận đúng đối chứng. Hoàn tất hai notebook hiện có.
2. **P1, làm khi có artifact thật.** Chạy bốn đối chứng clip/VideoPrism và hai đối chứng Qwen. Báo số theo query trước khi tune hoặc đổi công thức.
3. **P2, làm theo kết quả P1.** Điều chỉnh việc đổi hạng nhất, giữ thông tin query trong phân rã, hoặc tăng coverage của clip. Mỗi thử nghiệm đổi một yếu tố.
4. **P3, trước khi đề xuất bật mặc định.** Khóa cấu hình, chấm confirm, kiểm tra hồi quy và đo chi phí qua entry point thật.

## 2. P0 — Việc cần sửa và điều kiện nhận artifact

### 2.1. Làm chặt đường benchmark

| Vị trí | Việc cần làm | Điều kiện hoàn thành |
| --- | --- | --- |
| [capture_msr_candidate_pools.py](scripts/capture_msr_candidate_pools.py) | Ghi receipt đủ để khóa lần chạy. Tái dùng mã băm ở `videntra.evaluation.provenance`. Ghi commit, SHA-256 của DB, FAISS, weights SigLIP2 và LTR, snapshot gallery, split lock, cấu hình retrieval, seed, device và chế độ query plan. Lưu fingerprint khi capture và truyền sang replay. | Hai capture đối chứng có cùng fingerprint cho inputs bắt buộc, phần khác biệt đã khai báo chỉ nằm ở biến đang thử. Trên snapshot cũ phải tái lập 312/809 và so khớp top 1 theo từng query. Nếu snapshot đổi, lập baseline mới và ghi lý do. |
| [run_clip_vp_replay.py](scripts/run_clip_vp_replay.py) | Với `alpha>0`, code hiện có thể hoàn tất bằng fallback khi chỉ sai đường dẫn hoặc thiếu shard. Thêm chế độ kiểm tra nghiêm cho benchmark, hoặc kiểm tra receipt trước khi tính metrics. Lỗi thiếu cả shard, sai hash, shard mô phỏng hay số query được VP chấm bằng 0 phải dừng bước chấm VP. Giữ fallback hợp lệ cho từng query trong shard thật có coverage một phần. | Thử sai đường dẫn và một shard đúng cấu trúc nhưng không khớp query. Cả hai bị chặn khỏi báo cáo VP thật. Shard hợp lệ được ghi `vp_receipt_sha256`, `vp_dry_run=false` và số query được chấm. |
| [compare_msr_runs.py](scripts/compare_msr_runs.py) | Repo đã so tập query IDs, hash nội dung query và K. Với run mới, bắt buộc hai phía có fingerprint gallery, kiểm hash của dữ liệu replay đối chiếu receipt, và kiểm các inputs cần giống theo cặp. Với so R00–R01 hay R10–R11 bên dưới, kiểm `base_video_ids` bằng nhau theo query. Ghi actual full/partial/fallback coverage và số query đủ điều kiện đổi top 1. | Thay một ID ứng viên, sửa một hash hoặc xáo cặp từ snapshot khác làm benchmark báo lỗi rõ. Đổi từ A5 sang clip hoặc từ rules sang Qwen theo ma trận vẫn chạy. |

Commit `56a79c7` đã sửa đọc `files[name].sha256`, kiểm bốn checksum của shard, thứ tự `row_id`, query hash và phân biệt artifact mô phỏng. Giữ các kiểm tra đó. Với `preserve-unscored`, gắn thêm chỉ báo `top1_change_eligible` khi vị trí đầu có vector và có ít nhất hai video được chấm. Nếu A5 top 1 thiếu vector, nó được giữ tại chỗ. Đếm riêng trường hợp vị trí đầu có thể thay đổi với trường hợp thứ hạng đã thực sự đổi.

Chia dev theo `source_group` và lưu manifest tune/confirm có seed trước khi xem kết quả của các model. Có thể dùng tỷ lệ khoảng 70/30 theo nhóm nguồn. Nếu cần bộ lọc, thêm tùy chọn ở evaluator đọc danh sách IDs từ manifest; cắt cùng một tập cho cả baseline, challenger và labels rồi chạy phép kiểm parity như hiện tại. Chấm toàn dev để đối chiếu audit, chọn tham số từ phần tune, lấy phần confirm làm phép kiểm cho cấu hình đã chọn. Vì dev đã dùng trong các audit trước, gọi confirm là kiểm chứng nội bộ. `heldout_test_201` đã được xem ở các lượt trước và dùng như mốc hồi quy lịch sử. Dùng split theo nguồn mới nếu cần tuyên bố tổng quát hóa.

### 2.2. VideoPrism — xác nhận môi trường rồi encode

1. Dùng phiên Kaggle đã gửi. Kiểm trạng thái và log. Nếu v2 lỗi, sửa lỗi đầu tiên trong log của bản notebook đó rồi tạo version mới. Ghi phiên bản Python, FAISS, JAX/driver, PyAV, TensorFlow, device thực và kết quả import worker. Chạy preflight import, kiểm CUDA backend rồi encode một batch nhỏ cả video lẫn query trước khi chọn batch size. Nếu chạm giới hạn phiên GPU, chờ phiên đang chạy giải phóng slot.
2. Dùng [runbook VideoPrism](VIDENTRA_CLIP_RETRIEVAL_VIDEOPRISM_KAGGLE_RUNBOOK.md) để xác minh pin code, checkpoint và tokenizer. Mỗi video dùng 16 frame RGB `288×288` theo recipe đã hash. Lấy frame ordinal từ decoded presentation-order ledger và timestamp từ PTS/time base. Spot-check ít nhất một video ngắn, một video dọc hoặc có rotation, và một video có frame lặp gần biên.
3. Lấy hợp ứng viên từ **cả K=10 và K=20**, cho A5 và clip. Clip reserve được tính lại theo K nên K=10 có thể chọn một video khác với reserve K=20. Nếu đã có cache cho toàn bộ video MSR-VTT thì đối chiếu cache với hợp ứng viên và chỉ encode những video thiếu trong phạm vi dataset đang thử. Liệt kê asset thiếu, lỗi decode, sai checksum theo ID. Gửi query gốc và manifest inference. Labels, gold IDs và evaluator giữ ở local.
4. Encode, tải shard về thư mục version riêng và xác minh `video_vectors.npy`, `video_rows.jsonl`, `query_vectors.npy`, `query_rows.jsonl`, `receipt.json`. So số dòng, IDs duy nhất, dim hai phía, vectors hữu hạn, norm dương, hashes, recipe, model revision, số lỗi và `dry_run=false`. Ghi receipt hoàn tất sau khi đọc lại shard. Nếu chia shard để resume, khớp pin/recipe/schema và chỉ merge khi IDs duy nhất qua mọi shard.
5. Chạy đủ hai chính sách `full` và `preserve-unscored` ở một cấu hình ban đầu để định lượng coverage. Theo audit, shard chỉ có MSR-VTT cho K=20 có tối đa 22/809 pool A5 đủ toàn bộ vectors. Giữ mẫu số 809 và ghi hit ở các lát cắt có coverage. Với mục tiêu đo VideoPrism trên toàn gallery, chuẩn bị thêm asset AVC/QVHighlights được phép sử dụng để lấp phần thiếu, giữ cùng recipe và kiểm lại baseline.

### 2.3. Qwen — khóa đầu ra phân rã trước khi chấm ranking

1. Tải cache và receipt của phiên GPU hiện có sau khi hoàn thành. Đối chiếu revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, weights SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`, tokenizer hash, `compact-v1`, input hash và artifact hash. Checkpoint này khớp SHA-256 của weights local đã kiểm trong audit. [Model chính thức](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
2. Đối chiếu cả 809 query theo ba nhóm gồm rule theo complexity gate, plan đi qua validator và fallback. Ghi cache hit trên pipeline thực tế. Nếu notebook dùng checkpoint giữa chừng, nối phần chạy lại theo cache key và kiểm key trùng hoặc sai revision. Nếu cần batch để tiết kiệm GPU, thử một mẫu cố định bằng batch 1 và batch 16 với cùng prompt và cách decode. So parse, nội dung plan và fallback từng query trước khi chọn. Ghi GPU/runtime đo được.
3. Chạy capture với `--rules-only` và capture Qwen với `--query-plan-cache` cùng `--query-plan-revision`. [AdaptiveQueryDecomposer](src/videntra/query/llm.py) đọc cache đóng băng, kiểm metadata, cache key và query graph. Cache miss đi qua rules. Giữ embedding của nguyên câu và thêm tín hiệu mệnh đề qua cơ chế hiện có. Đưa các query dataset khác về đường rules nếu bundle thử chỉ chứa MSR-VTT. Ghi fallback và coverage. Mỗi phương án prompt/gate mới phải có version riêng và cache riêng.

## 3. P1 — Chạy ma trận ghép cặp

| Mã | Query plan | Pool | VideoPrism | So cặp để đo |
| --- | --- | --- | --- | --- |
| R00 | Rules | A5 | Tắt | Mốc |
| R10 | Rules | Clip reserve | Tắt | R10 − R00 đo ứng viên clip |
| R01 | Rules | A5 | Bật | R01 − R00 đo VP trên cùng pool |
| R11 | Rules | Clip reserve | Bật | R11 − R10 đo VP trên pool clip. R11 − R01 đo clip khi đã có VP |
| Q00 | Qwen cache | A5 | Tắt | Q00 − R00 đo phân rã trong SSOT |
| Q10 | Qwen cache | Clip reserve | Tắt | Q10 − R10 đo Qwen khi thêm reserve |
| Q11 | Qwen cache | Clip reserve | Bật | Sau khi khóa cấu hình riêng. Q11 − R11 đo Qwen trên cấu hình clip+VP |

Chạy K=10 và K=20, giữ cùng K trong mỗi cặp. Dùng `alpha=0` cho R00, R10, Q00, Q10. Với VP, thử sweep nhỏ `0.25, 0.50, 0.60, 0.75, 1.00` trên phần tune. Dùng cùng `alpha`, coverage policy và shard cho R01/R11. Với RRF toàn pool, ứng viên reserve ở hạng cuối chỉ có khả năng vượt A5 hạng 1 khi `alpha>0.5`. Với rerank coverage một phần, kiểm theo các vị trí có vector và thống kê riêng việc top 1 có thể đổi.

Lệnh mẫu dưới đây dùng CLI đang có. Mỗi lần chạy tạo thư mục mới. Mẫu R01 là một điểm trong sweep. Tải outputs bằng lệnh trong audit ở mục 1 và gán đúng đường dẫn local đã xác minh.

```powershell
$ErrorActionPreference = 'Stop'
$py = '.\.venv\Scripts\python.exe'
$root = 'work/experiments/msrvtt_stage2_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$vp = 'work/kaggle/videoprism_msr_audit_output_20261115/dev'
$qwen = 'work/kaggle/qwen_msr_audit_output_20261115/query_plan_cache.jsonl'
$qwenRev = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

function Invoke-Bench {
    & $py -B @args
    if ($LASTEXITCODE -ne 0) { throw "Benchmark command failed ($LASTEXITCODE)" }
}

$labels = "$root/rules/evaluation/clip_vp_dev_full_859_labels.jsonl"
$lock = "$root/rules/evaluation/clip_vp_dev_full_859_split_lock.json"
Invoke-Bench scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --temporal-rerank-lambda 0 --rules-only --out-dir "$root/rules/pools" --eval-dir "$root/rules/evaluation"
Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/rules/pools" --pool a5 --k 20 --alpha 0 --out-dir "$root/R00_k20"
Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/rules/pools" --pool clip --k 20 --alpha 0 --out-dir "$root/R10_k20"
Invoke-Bench scripts/compare_msr_runs.py --baseline "$root/R00_k20" --challenger "$root/R10_k20" --labels $labels --split-lock $lock --out-dir "$root/compare_R00_R10_k20"

# VP đã qua kiểm tra artifact
Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/rules/pools" --pool a5 --k 20 --alpha 0.60 --vp-dir $vp --vp-coverage-policy preserve-unscored --out-dir "$root/R01_k20_a060"
$receipt = Get-Content "$root/R01_k20_a060/run_receipt.json" -Raw | ConvertFrom-Json
if ($receipt.vp_dry_run -ne $false -or -not $receipt.vp_receipt_sha256 -or
    (($receipt.vp_coverage_count + $receipt.vp_partial_count) -eq 0)) {
    throw 'Replay chưa chấm được query bằng VideoPrism thật'
}
Invoke-Bench scripts/compare_msr_runs.py --baseline "$root/R00_k20" --challenger "$root/R01_k20_a060" --labels $labels --split-lock $lock --out-dir "$root/compare_R00_R01_k20_a060"

# Qwen cache đã qua kiểm tra artifact
Invoke-Bench scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --temporal-rerank-lambda 0 --query-plan-cache $qwen --query-plan-revision $qwenRev --out-dir "$root/qwen/pools" --eval-dir "$root/qwen/evaluation"
Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/qwen/pools" --pool a5 --k 20 --alpha 0 --out-dir "$root/Q00_k20"
Invoke-Bench scripts/compare_msr_runs.py --baseline "$root/R00_k20" --challenger "$root/Q00_k20" --labels $labels --split-lock $lock --out-dir "$root/compare_R00_Q00_k20"
```

Lặp với K=10 và các cặp trong ma trận. Với R11, đổi pool sang `clip` trên capture rules. Với Q10 và Q11, lấy pool `clip` từ capture Qwen. Đọc receipt và kiểm coverage riêng cho mọi cặp có VP. Chạy Q00 và Q10 ngay khi cache Qwen sẵn sàng nếu VP còn đợi GPU.

**Kết quả cần nộp.** Mỗi cặp có `run_receipt.json`, `per_query_evaluation.jsonl`, `benchmark_summary.json` và `benchmark_report.md` theo nơi pipeline ghi. Bảng chính ghi Hit@1 dạng số đúng/mẫu số, chênh lệch điểm phần trăm, thắng/thua, CI 95% bootstrap ghép cặp theo `source_group`, MRR@10 và Recall@K. Thêm số query có video đúng được thêm/mất trong pool, số query target lên hạng 1, số lần top 1 bị đổi sai, coverage VP và số plan Qwen thực dùng. Tách latencies offline trên Kaggle, replay local và request end-to-end. Với truy vấn có nhãn thời gian, ghi rõ anchor A5 có sẵn hay thiếu cho video được thăng hạng. Nếu cần quyết định trên moment tIoU, đo bằng pipeline có bước sinh anchor cho ứng viên mới.

## 4. P2 — Nâng cấp theo kiểu lỗi đo được

| Kết quả P1 | Thử nghiệm tiếp theo | Cách kiểm tra |
| --- | --- | --- |
| VP cứu một số video hạng 2–5 nhưng cũng đẩy nhiều A5 hit xuống | Thử gate đổi top 1 theo chênh lệch cosine giữa hai ứng viên được VP ưu tiên và độ chắc theo chênh lệch score A5 thực nếu snapshot đang lưu. Dùng các tín hiệu available tại inference. Bắt đầu từ gate một ngưỡng trên VP margin; thêm điều kiện A5 margin khi ablation chứng minh cần. | So gate với RRF tốt nhất trên tune. Báo số lần gate cho đổi, số thắng/thua và hiệu quả trên confirm. Giữ tie-break cố định. |
| Clip tăng coverage, ít video mới lên hạng 1 | Trước hết đo `alpha>0.5`. Nếu nhiều query mất ứng viên đúng do ghi đè vị trí K, thử đưa ứng viên clip vào pool hợp với kích thước K+1. | So clip hợp với A5 K+1 và A5 K để phân biệt hiệu ứng nguồn ứng viên với hiệu ứng thêm slot. Giữ ngân sách rerank ngang nhau trong cặp chính. |
| Qwen cải thiện nhiều query đa hành động nhưng mất ở caption đơn hoặc khi bỏ sót điều kiện | Thêm gate dựa trên đặc trưng quan sát được của query như đa hành động, thứ tự, phủ định, số lượng hoặc quan hệ vai trò. Kiểm candidate plan giữ lại hành động chính, thực thể, vai trò, số lượng, phạm vi phủ định và thứ tự. Khi validator từ chối, dùng rules. | So Qwen hiện tại với Qwen có gate, dùng cùng model revision. Kiểm mẫu lỗi theo nhóm. Thay prompt và gate trong các run riêng để quy được nguyên nhân. |
| VP và Qwen riêng lẻ đều có cải thiện | Chạy Q11 và đối chiếu Q00, Q10, R11 với cùng K và cấu hình đã khóa. | Kiểm phần thắng Qwen trùng hay bổ sung phần thắng VP. Đọc paired deltas và CI, đồng thời kiểm hồi quy QVHighlights/AVC. |
| Không thấy tăng Hit@1 đủ rõ | Đọc các ca thắng/thua và tách lỗi thiếu ứng viên khỏi lỗi xếp hạng. Đo trần Hit@1 của từng pool bằng Recall@K ở phía evaluator. | Ưu tiên đúng tầng gây mất nhiều query. Dừng sweep khi cấu hình mới làm CI xấu hơn hoặc chỉ di chuyển kết quả trong cùng top K. |

Nếu chuyển một phương án sang đường chạy trực tiếp, nối vào stage rerank hiện có trong [UnifiedMultimodalSearchPipeline](src/videntra/retrieval/unified_pipeline.py) qua tùy chọn, dùng lại logic replay đã kiểm thử. Kiểm cùng top-k, deterministic tie-break, lookup không đổi theo `video_id`, fallback thiếu dữ liệu, tạo moment anchors và schema output. Embedding nặng xử lý offline trên Kaggle. Đo p50/p95, peak RAM/VRAM và thời gian nạp trên cấu hình đã chạy thật trước khi đề xuất mở mặc định.

## 5. P3 — Tiêu chí nghiệm thu

- [ ] Xác nhận artifact thật, provenance, độ phủ và câu query gốc cho mỗi challenger. Với các so sánh VP, ghi số lần có thể đổi hạng 1 và số lần đã đổi. Với Qwen, ghi số query dùng plan và fallback.
- [ ] Lưu cấu hình cùng tiêu chí trước khi mở phần confirm. Mốc đề xuất cho hiệu quả thực dụng là Hit@1 tăng ít nhất 1 điểm phần trăm trên confirm và CI ghép cặp 95% nằm trên 0. Nếu CI còn cắt 0, ghi kết quả chưa đủ chắc và tiếp tục đánh giá trên bộ nguồn độc lập khi có.
- [ ] Kiểm QVHighlights 22 query, AVC 28 query và moment trên 22 query có nhãn ở dev. Nêu số ca và xem từng ca tụt, vì chỉ một ca cũng tạo thay đổi tỷ lệ lớn ở các tập nhỏ. Chạy split hồi quy lịch sử sau khi chốt cấu hình, kèm nguồn/hardware.
- [ ] Khi sửa code, chạy các test liên quan bên dưới. Chạy full suite trước khi đề xuất đưa vào pipeline chính. Bổ sung regression test cho contract mới như shard sai hash, thiếu shard, cache sai revision hoặc score có coverage một phần.
- [ ] Lưu output từng run dưới `work/experiments/`, logs/receipts Kaggle, và kết quả bảng trong `benchmark/`. Cập nhật `docs/PROGRESS.md`, `README.md` khi có số mới, `PROJECT.md` khi cấu trúc đổi, và `requirements.txt` khi thay runtime dependencies. Lưu rõ trạng thái của những so sánh chưa chạy.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/evaluation/test_clip_vp_replay.py tests/evaluation/test_split_lock.py tests/indexing/test_videoprism_kaggle_pipeline.py tests/query/test_llm_decomposer.py tests/retrieval/test_clip_branch.py
```
