# Kiểm chứng nhánh clip, VideoPrism và Qwen trên MSR-VTT

Ngày 15/11/2026. Dev đầy đủ có 859 query, trong đó 809 query MSR-VTT. Mỗi query truy hồi trên cùng gallery 1.324 video. A5 dùng đúng trọng số LTR của fold OOF cho 742 query và trọng số cuối cho 117 query còn lại, theo provenance gate. Temporal rerank có `lambda=0`.

## 1. Clip reserve với A5 cố định

| Candidate pool | A5 Hit@1 | Clip reserve Hit@1 | A5 Recall@K | Clip reserve Recall@K | Thêm/mất query có đáp án trong pool | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K=10 | 312/809 | 312/809 | 544/809 | 555/809 | +15/−4 | 0,4758 → 0,4771 |
| K=20 | 312/809 | 312/809 | 606/809 | 611/809 | +9/−4 | 0,4758 → 0,4758 |

Chạy lại capture rồi replay bốn tổ hợp `{a5, clip} × {10, 20}` với `alpha=0`. Hạng nhất của cả 859 query khớp từng ID với `dev_full_evaluation.jsonl` của run tham chiếu `8c40e943`. Hai phép so ghép cặp có 0 query thắng, 0 query thua và CI 95% Hit@1 bằng [0, 0] điểm phần trăm. Recall@K của MSR-VTT tăng 1,36 điểm ở K=10 và 0,62 điểm ở K=20. QVHighlights giữ 19/22, AVC giữ 26/28 và chỉ số top-1 anchor trên 22 query có nhãn thời gian giữ 16/22 trong replay. Kết quả này áp dụng cho video replay trên pool capture, với moment anchor được lấy từ A5.

Clip reserve bảo vệ K−1 vị trí đầu. Với RRF `(1−alpha)/(60+r_A5) + alpha/(60+r_VP)`, ứng viên ở vị trí K cần `alpha > 0.5` mới có khả năng vượt A5 hạng 1, ngay cả khi VideoPrism xếp nó hạng nhất và xếp A5 top 1 ở hạng cuối. Tại `alpha=0.5`, tie-break ưu tiên thứ tự A5. Vì vậy sweep nhỏ hơn hoặc bằng 0,5 có thể cải thiện các ứng viên sẵn có ở gần đầu pool và không thể đưa ứng viên reserve ở cuối lên hạng nhất.

## 2. VideoPrism trên gallery hỗn hợp

Worker ghi `files[name].sha256` trong receipt. Replay cũ chuyển object `{sha256, bytes}` thành chuỗi rồi so sánh với SHA-256, dẫn đến từ chối shard thật. Replay mới kiểm tra đủ bốn hash, số dòng, thứ tự `row_id`, hash query và trạng thái `dry_run`. Vector mô phỏng cần `--allow-synthetic` để chạy smoke và report gắn nhãn rõ.

Nếu shard chỉ có video MSR-VTT, A5 có tối đa 94/809 pool đủ toàn bộ vector ở K=10 và 22/809 ở K=20 theo membership gallery. Clip reserve có lần lượt 94/809 và 24/809. Với chính sách `full`, một video thiếu vector làm cả query quay về thứ tự A5 hoặc clip reserve. Chính sách opt-in `preserve-unscored` giữ nguyên vị trí video thiếu vector và chỉ rerank các vị trí có ít nhất hai ứng viên đủ vector. Kiểm tra membership cho thấy cả 809 pool đáp ứng điều kiện về số video MSR-VTT nếu encode đủ 1.000 video. Báo cáo ghi riêng full coverage, partial coverage và fallback, đồng thời tính Hit@1 chính trên đủ 809 query.

Kaggle dùng dataset video đã có trên tài khoản làm nguồn media, kèm đối chiếu SHA-256 với video local và fail riêng từng video lệch. Query bundle chứa ID, câu gốc và tên dataset. Checkpoint, tokenizer và code VideoPrism được pin theo runbook. [Notebook VideoPrism](https://www.kaggle.com/code/lvhuylong/videntra-msr-videoprism-dev-20261115) · [Model card chính thức](https://huggingface.co/google/videoprism-lvt-base-f16r288)

Lượt Kaggle đầu dừng ở import do thiếu `faiss` trong image Python 3.13. Phiên bản 2 pin `faiss-cpu==1.13.2` cùng môi trường notebook và đã được đưa vào hàng đợi GPU. Chưa có shard thật để báo Hit@1 sau VideoPrism tại thời điểm audit.

## 3. Qwen2.5-1.5B-Instruct cho query decomposition

Notebook chỉ cần 809 query dev, gồm `query_id`, `query_text`, `dataset`. Tải checkpoint công khai tại revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, SHA-256 weights `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. Prompt hiện dùng `compact-v1` và validator hiện có để phân rã mô tả thị giác, entity, OCR/ASR, phủ định và quan hệ thời gian. [Notebook Qwen](https://www.kaggle.com/code/lvhuylong/videntra-msr-qwen15b-dev-20261115) · [Checkpoint chính thức](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

So `--rules-only` với `--query-plan-cache … --query-plan-revision …` trên cùng SSOT, OOF và gallery. Cache được đọc riêng khỏi cache text embeddings, ở chế độ chỉ đọc. Loader kiểm tra cache key, revision, prompt version, trạng thái validator và query graph. Query thiếu plan hợp lệ dùng rules, kể cả khi máy có GPU. Pipeline giữ embedding của câu gốc và dùng các mệnh đề làm tín hiệu bổ sung. Báo số plan đã được dùng và số query thắng/thua trên cả 809 query. Lát cắt có cache hit giúp phân tích cơ chế.

Log Kaggle xác nhận Qwen đã tải đúng checksum và chạy trên Tesla T4 với Transformers 4.51.3. Mốc 140/809 query đã cho 129 plan hợp lệ, 7 fallback và 4 query xử lý bằng rules theo complexity gate. Kaggle chưa xuất cache hoàn chỉnh tại thời điểm audit. Lần thử đưa bản batch 16 vào hàng đợi bị giới hạn hai phiên GPU của tài khoản. Tỷ lệ plan hợp lệ là số đo ở bước phân rã và cần replay qua SSOT mới đo được Hit@1.

## 4. Lệnh tái lập

Các lệnh dưới đây dùng PowerShell từ thư mục gốc. Lưu mỗi capture, shard và replay vào thư mục mới. Tắt inference live khi tạo baseline đối chứng.

```powershell
$py = '.\.venv\Scripts\python.exe'
$root = 'work/experiments/msrvtt_clip_qwen_vp_repro'
$baseline = "$root/rules"
$candidate = "$root/qwen"
$revision = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
$vp = 'work/kaggle/videoprism_msr_audit_output_20261115/dev'
$qwen = 'work/kaggle/qwen_msr_audit_output_20261115/query_plan_cache.jsonl'

# Chạy sau khi từng notebook báo COMPLETE
.\.venv\Scripts\kaggle.exe kernels output lvhuylong/videntra-msr-videoprism-dev-20261115 -p (Split-Path $vp) --file-pattern '^dev/'
.\.venv\Scripts\kaggle.exe kernels output lvhuylong/videntra-msr-qwen15b-dev-20261115 -p (Split-Path $qwen) --file-pattern '^qwen_receipt\.json$|^query_plan_cache\.jsonl$'

& $py -B scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --rules-only --out-dir "$baseline/pools" --eval-dir "$baseline/evaluation"
& $py -B scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --query-plan-cache $qwen --query-plan-revision $revision --out-dir "$candidate/pools" --eval-dir "$candidate/evaluation"

$labels = "$baseline/evaluation/clip_vp_dev_full_859_labels.jsonl"
$lock = "$baseline/evaluation/clip_vp_dev_full_859_split_lock.json"
foreach ($k in 10, 20) {
    foreach ($pool in 'a5', 'clip') {
        & $py -B scripts/run_clip_vp_replay.py --pools-dir "$baseline/pools" --pool $pool --k $k --alpha 0 --out-dir "$baseline/${pool}_k$k"
        & $py -B scripts/run_clip_vp_replay.py --pools-dir "$candidate/pools" --pool $pool --k $k --alpha 0 --out-dir "$candidate/${pool}_k$k"
    }
    & $py -B scripts/compare_msr_runs.py --baseline "$baseline/a5_k$k" --challenger "$baseline/clip_k$k" --labels $labels --split-lock $lock --out-dir "$root/clip_k$k"
    & $py -B scripts/compare_msr_runs.py --baseline "$baseline/a5_k$k" --challenger "$candidate/a5_k$k" --labels $labels --split-lock $lock --out-dir "$root/qwen_k$k"
}

# Ví dụ phép so cô lập VideoPrism trên A5 K=20
& $py -B scripts/run_clip_vp_replay.py --pools-dir "$baseline/pools" --pool a5 --k 20 --alpha 0.25 --vp-dir $vp --vp-coverage-policy preserve-unscored --out-dir "$root/vp025_a5_k20"
& $py -B scripts/compare_msr_runs.py --baseline "$baseline/a5_k20" --challenger "$root/vp025_a5_k20" --labels $labels --split-lock $lock --out-dir "$root/compare_vp025_a5_k20"
```

Với VideoPrism, đo trên cùng một pool trước rồi đổi `a5` sang `clip` ở cả hai phía. Để đo đóng góp của clip khi có VideoPrism, so A5+VideoPrism với Clip+VideoPrism ở cùng K, alpha và coverage policy. Tune `K ∈ {10, 20}`, thử `alpha ∈ {0.25, 0.5, 0.6, 0.75, 1.0}`. Tách dev tune/dev confirm theo `source_group` trước khi chọn và khóa cấu hình. So Hit@1, MRR@10, Recall@K, số thắng/thua và CI ghép cặp theo nhóm nguồn. Held-out 191 query MSR-VTT đã từng được xem, nên dùng làm kiểm tra hồi quy lịch sử sau khi khóa config.

Nếu Qwen thay đổi thứ hạng theo cả hai hướng, xem nhóm caption có nhiều hành động, thứ tự, vai trò, số lượng hoặc phủ định. Thử giữ Qwen trên nhóm đó và giữ tỷ lệ embedding câu gốc. Nếu VideoPrism giúp đẩy đúng video từ hạng 2–5 lên hạng nhất nhưng làm giảm ở nhóm A5 thắng rõ, thử chỉ cho phép đổi top 1 khi VideoPrism có margin đủ lớn trên hai ứng viên đứng đầu, chọn ngưỡng trên dev tune. Gate chỉ dùng điểm ứng viên, metadata và đặc trưng query. Xác nhận riêng trên dev confirm trước khi bật mặc định.
