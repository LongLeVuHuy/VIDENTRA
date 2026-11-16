# Kế hoạch triển khai và đo Hit@1 MSR-VTT cho Antigravity

Cập nhật sau lần tái kiểm commit `7fba26f`. Antigravity hoàn tất validator giữ phủ định và số lượng, rồi benchmark Qwen2.5‑1.5B gốc với prompt `extractive-v2` trên A5. Nếu Hit@1 MSR‑VTT trên tune K=20 bằng hoặc thấp hơn A5 rules, train LoRA trên Kaggle rồi chấm lại trong cùng lượt triển khai. [Kết quả tái kiểm](docs/PROGRESS.md#tái-kiểm-bốn-bản-sửa-tại-commit-a710e37)

## Việc còn lại của validator

Ba bản sửa về mặc định trên CUDA, cache prompt version và kiểm nhãn/split đã qua tái kiểm tra. Validator đã chặn clause bịa hành động, nhưng tại `validate_extractive_events` vẫn chỉ nhìn token ngay trước span khi kiểm phủ định và số lượng. Trạng từ `really` hoặc tính từ `brown` làm hai lỗi dưới đây lọt qua. `visual_clauses` đi vào embedding truy vấn phụ, nên cần kiểm trực tiếp nội dung đã được chấp nhận. [Validator](src/videntra/query/llm.py#L1182) · [Tạo truy vấn phụ](src/videntra/retrieval/unified_pipeline.py#L2292)

1. Mở rộng kiểm tra từ token sát span lên phạm vi của cụm nguồn. Gắn từ phủ định với vị từ mà nó chi phối, kể cả qua trợ động từ hoặc trạng từ. Gắn số lượng với danh ngữ tương ứng, kể cả có tính từ xen giữa. Tái dùng định vị span và các danh sách token hiện có. Tôn trọng ranh giới mệnh đề và nội dung trong dấu trích dẫn để ràng buộc của cụm khác không lan sang clause đang xét.
2. Khi clause mất một ràng buộc có tác dụng, từ chối plan và cho pipeline quay về rules với query gốc. Prompt nên yêu cầu model trích span rộng hơn để giữ cả ràng buộc. Chỉ nới cách ghép span khi có thể xác minh quan hệ với cụm nguồn. Áp dụng cùng validator cho Qwen gốc, dữ liệu SFT và export cache LoRA. Tạo cache mới từ code đã sửa và ghi commit validator trong receipt.
3. Kiểm cả đầu ra extractor và đường fallback của pipeline. Với plan bị từ chối, assert diagnostics báo fallback, embedding câu gốc vẫn được tạo và clause mất ràng buộc không được đưa vào danh sách embedding bổ sung. Dùng các test regression hiện có bên dưới rồi chạy full suite.

| Query gốc | Clause cần kiểm | Kỳ vọng |
| --- | --- | --- |
| `the dog is not really swimming in the pool` | `swimming in the pool` | Từ chối. Clause đầy đủ giữ `not` phải được nhận. |
| `two brown dogs are swimming in the pool` | `dogs are swimming in the pool` | Từ chối. Clause đầy đủ giữ `two brown dogs` phải được nhận. Kiểm thêm cách viết `2 brown dogs`. |
| `hai con chó nâu đang bơi trong hồ` | `chó nâu đang bơi trong hồ` | Từ chối mất `hai`. Clause đầy đủ giữ số và dấu phải được nhận. |
| `con chó thật ra không hề bơi trong hồ` | `bơi trong hồ` | Từ chối mất `không`. Clause giữ `không hề bơi` phải được nhận. |
| `a man is not swimming, but a woman is running` | `a woman is running` | Không gán phủ định ở mệnh đề đầu cho mệnh đề sau. Nếu plan biểu diễn cả hai mệnh đề, clause thứ nhất phải giữ phủ định. |
| `one person points at two brown dogs` | `one person points at two brown dogs` | Nhận và giữ đúng số lượng của từng đối tượng. Thêm ca OCR/ASR có chữ số hoặc phủ định trong dấu trích dẫn để kiểm chúng không bị gán sang mô tả hình ảnh. |

## Bốn lỗi trong audit và bộ kiểm thử hồi quy

| Lỗi đã tái hiện | Sửa ở đâu và sửa thế nào | Kiểm thử bắt buộc |
| --- | --- | --- |
| **CUDA tự bật Qwen `compact-v1`** | Đặt rules mặc định cho đường chạy Unified qua cả CLI lẫn API. Chọn backend bằng cấu hình tường minh tại điểm tạo pipeline. Giữ đường bật LLM có chủ đích như `VIDENTRA_LLM_ENABLED=1` nếu vẫn là hợp đồng hiện hành. Ghi chế độ thực dùng trong diagnostics. [Backend](src/videntra/query/llm.py#L487) · [Pipeline](src/videntra/retrieval/unified_pipeline.py#L659) | Giả lập CUDA có sẵn và có model local, bỏ biến môi trường, rồi chạy cả CLI/API trên cùng query đủ dài để qua complexity gate. Dùng spy kiểm số lần gọi LLM bằng 0 và kết quả dùng rules. Kiểm thêm cấu hình bật rõ ràng và frozen cache để xác nhận mỗi đường đi đúng chế độ, cache miss về rules. |
| **Cache prompt mới bị từ chối bởi capture** | Thêm `--query-plan-prompt-version` vào capture và truyền qua factory, cache key, loader và receipt replay. Prewarm cần thêm `--prompt-version` và truyền cùng giá trị vào backend/decomposer. Mặc định tương thích cache lịch sử `compact-v1`. Ghi prompt version và model/adapter revision riêng, xác minh hash cache và receipt trước khi chọn benchmark artifact. [Capture](scripts/capture_msr_candidate_pools.py#L391) · [Loader](src/videntra/query/llm.py#L1326) · [Prewarm](scripts/prewarm_llm_query_cache.py) | Tạo cache nhỏ qua exporter rồi gọi capture CLI thật với từng version `compact-v1`, `extractive-v2` và version LoRA được khai báo trong receipt. Assert cache hit và version xuất hiện trong receipt. Đổi version, revision hoặc sửa một byte rồi kiểm bị từ chối. Kiểm cache miss dùng rules với 0 lần gọi LLM live. Artifact `dry_run=true` chỉ dùng smoke. |
| **Evaluator nhận labels bị sửa và dòng tune thiếu split** | Tính SHA-256 labels đầu vào và đối chiếu `metadata.labels_sha256` của split lock trước khi đọc target hoặc xuất metric. Khi lọc tune/confirm, mỗi ID cần thuộc partition tường minh, có trong lock và cùng nhóm nguồn với manifest đã khóa. Kiểm hai partition không trùng query ID hoặc source group. Đọc định dạng cũ theo hợp đồng của từng run. Với run mới, yêu cầu hash được ghi khi freeze. [Evaluator](scripts/compare_msr_runs.py#L306) · [Bộ lọc split](scripts/compare_msr_runs.py#L236) | Đổi một `target_videos` trên bản sao, giữ split lock và query ID, assert evaluator dừng vì sai hash. Thử JSONL thiếu split, partition lạ hoặc một source group có mặt ở cả hai phần và yêu cầu báo lỗi. Dùng labels đã khóa và manifest hợp lệ để chạy lại mốc 312/809 trên cùng gallery. |
| **Validator nhận hành động bịa khi trùng một từ** | Với `extractive-v2`, xác minh `vis` là đoạn trích từ query sau chuẩn hóa có kiểm soát về Unicode, chữ hoa và khoảng trắng. Nếu câu có chủ ngữ dùng chung hoặc quan hệ qua nhiều mệnh đề, giữ cụm gốc rộng hơn để giữ vai trò, phủ định, số lượng và thứ tự. Từ chối plan phá nghĩa rồi dùng rules. Giữ validator này cho cache LoRA và ghi version rõ ràng để cache đã sinh bằng hợp đồng cũ có thể nhận diện. [Validator](src/videntra/query/llm.py#L1115) | Với `a dog is swimming in a pool`, assert từ chối `a dog flies an airplane` và nhận clause trích đúng. Thêm ca tiếng Việt có dấu, OCR/ASR chứa trích dẫn, `two dogs` và `not swimming`, hai hành động có thứ tự. Kiểm ranking dùng query gốc khi plan bị từ chối. |

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/api/test_unified_parity.py tests/query/test_llm_decomposer.py tests/query/test_qwen_sft_lora.py tests/evaluation/test_clip_vp_replay.py tests/evaluation/test_split_lock.py
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

## Đo Qwen gốc và quyết định train LoRA

1. Khóa gallery, danh sách query và nhãn, LTR/OOF, index và cấu hình A5. Dùng cùng K, `alpha=0` và `temporal_rerank_lambda=0`. Giữ clip reserve và VideoPrism tắt cho đối chứng chính. Khóa tune/confirm theo source group trước khi đọc kết quả. Nếu chạy inference trên toàn dev, chấm tune trước và giữ nhãn confirm ở evaluator. MSR‑VTT đầy đủ có 809 query. Chấm thêm toàn dev theo quy trình sau khi đã khóa lựa chọn, cạnh báo cáo confirm.
2. Dùng checkpoint công khai Qwen2.5‑1.5B‑Instruct đã pin ở mục 2.3. Trên Kaggle GPU, chạy prewarm với `--prompt-version extractive-v2`, `--queries-file`, `--enable-llm`, `--backend transformers`, `--hf-model-id`, `--hf-revision`, `--model-revision`, `--cache-path` và `--receipt-path`. Export query text/ID bằng luồng inference hiện có. Lưu cache và receipt riêng, ghi số query đầu vào, plan hợp lệ, tier rules, fallback và lý do. Ràng buộc revision/hash của model, tokenizer, prompt và cache với receipt đã kiểm.
3. Capture A5 `--rules-only` và A5 đọc frozen cache, truyền `--query-plan-prompt-version extractive-v2` ở challenger. Dùng block A5/Qwen ở mục 3 và chấm cặp tại K=10, K=20. Với phần tune, thêm `--filter-manifest` và `--filter-split tune` vào comparator. Kiểm số cache hit, fallback và `llm_live_calls=0`. Báo Hit@1 dạng số đúng/mẫu số, delta điểm phần trăm, CI 95% theo source group, thắng/thua, Recall@K và MRR@10. Nếu mẫu số hoặc artifact sai, sửa lỗi và chạy lại phép đo.
4. **Nếu số hit của `extractive-v2` ≤ số hit của A5 rules trên cùng query MSR‑VTT tune ở K=20, train LoRA ngay trên Kaggle rồi chấm lại.** K=10 là phép kiểm tra bổ sung. Nếu số hit tăng, chốt cấu hình `extractive-v2` và chấm confirm. Báo CI để thể hiện độ bất định. Dùng số hit nguyên để quyết định khi tỷ lệ phần trăm đã làm tròn.
5. Với LoRA, dùng `scripts/prepare_qwen_sft_dataset.py --source-group-manifest` trên manifest text-only có `query_text` và partition `tune` tường minh. Chia SFT train/val theo nguồn hoặc họ template và kiểm trùng query chuẩn hóa. Xác minh hash của tập đã freeze. Chạy `scripts/kaggle_train_qwen_lora.py` bằng `--train-jsonl`, `--val-jsonl`, `--base-revision` đã pin, `--prompt-version extractive-v2`, `--model-revision` riêng cho adapter, `--export-cache-queries` và `--output-dir` mới. Thử một batch với tokenizer thật để kiểm phần JSON/nhãn assistant còn đủ token, rồi train trên GPU. Lưu adapter, cache, receipt, seed, hash và thống kê fallback.
6. Ngay khi Kaggle xuất kết quả LoRA, capture với `--query-plan-cache`, `--query-plan-receipt`, `--query-plan-prompt-version` và `--query-plan-revision` lấy theo receipt của adapter. Dùng lại các run A5 rules và Qwen gốc cùng snapshot. Replay A5 ở K=10/20 và `alpha=0`. Chấm hai cặp LoRA với rules và LoRA với Qwen gốc trên tune. Chốt checkpoint theo train/val hoặc tune trước khi chấm confirm. Báo Hit@1 theo số đúng/mẫu số, delta, thắng/thua, CI và coverage. Ghi đường dẫn tới receipt, adapter, cache và báo cáo vào plan và `docs/PROGRESS.md`.

## Tối ưu theo kênh tín hiệu

| Kênh | Xử lý và tối ưu | Đo để quyết định |
| --- | --- | --- |
| **Visual** | Dùng SigLIP2 để truy hồi frame/clip và ghép ứng viên theo video ID. Giữ query gốc trong embedding. Mệnh đề có căn cứ bổ sung tín hiệu hình ảnh. Dùng lại index và embedding đã pin theo snapshot. | So Visual only với A5 và đo Recall@10/20, Hit@1, latency. Đo riêng lỗi hình ảnh như nhầm hành động, đối tượng và thứ tự. |
| **ASR** | Faster‑Whisper/PhoWhisper tạo transcript và thời gian từ audio ở khâu ingest. Tìm trên index ASR của toàn corpus, bổ trợ hoặc hợp nhất theo ý định lời thoại. Dùng FTS5 trước. Bật E5 khi kênh có ý định phù hợp và lexical match thưa. | So A5 với A5 tắt ASR. Kiểm query có câu thoại, trùng lời ngẫu nhiên và segment ngoài moment. Dùng UNKNOWN khi thiếu transcript. Các âm thanh phi lời nói như tiếng gõ cần tín hiệu audio riêng nếu được triển khai. |
| **OCR** | OCRRouter nhận dạng chữ trong ảnh khi ingest. Tìm trên index OCR của toàn corpus. Giữ nguyên chữ trích dẫn, kiểm evidence theo frame/timestamp và ưu tiên exact match khi có. | So A5 với A5 tắt OCR. Kiểm chữ trên màn hình khác lời thoại, dấu tiếng Việt và evidence ngoài moment. |
| **Rules/Qwen và fusion** | Rules hoặc Qwen phân tích ý định, mệnh đề và ràng buộc. Các kênh chuyên biệt tìm bằng chứng. Hợp nhất theo video ID và thời gian, ghi điểm đóng góp và trạng thái thiếu của mỗi kênh. Query đa phương thức có thể dùng nhiều kênh. Gate dựa trên query và tín hiệu sẵn có qua inference. | Đo thêm `Visual + OCR`, `Visual + ASR` và A5 đầy đủ trên cùng split. Báo Hit@1, Recall@K, moment tIoU khi có nhãn và p50/p95 trên phần cứng đã chạy. Đo routing riêng sau khi đã đo prompt/LoRA để quy được tác động. |

Tối ưu chi phí bằng cách dùng lại artifact theo snapshot, cache query embedding theo model/prompt và tránh gọi lại encoder khi cùng đầu vào. Profile trước và sau mỗi thay đổi. So parity giữa lệnh search và API, bao gồm query hỗn hợp, transcript trống, OCR trống và fallback rules. Chỉ dùng ground truth phía evaluator. Giữ PTS ledger cho các phép kiểm timestamp.

## Bàn giao theo dõi Kaggle cho Antigravity

Kiểm tra từng notebook độc lập. Khi một notebook đã xuất artifact đạt kiểm tra ở mục 2, chạy ngay các cặp tương ứng ở mục 3.

```powershell
$kg = '.\.venv\Scripts\kaggle.exe'
$vpRef = 'lvhuylong/videntra-msr-videoprism-dev-20261115'
$qwenRef = $env:VIDENTRA_QWEN_NOTEBOOK_REF  # Ref notebook extractive-v2 sau khi push
foreach ($ref in $vpRef, $qwenRef) {
    if (-not $ref) { continue }
    & $kg kernels status $ref
    & $kg kernels files $ref -v
}
```

| Kaggle trả về | Antigravity làm tiếp |
| --- | --- |
| `QUEUED` | Giữ phiên đang xếp hàng. Kiểm tra lại sau vài phút. Nếu hàng đợi kéo dài, kiểm tra quota và các phiên GPU đang chiếm slot trước khi cân nhắc gửi lại. |
| `RUNNING` | Theo dõi tiến độ với `kernels logs --follow`. Ghi mốc query hoặc video mới nhất. Có thể đóng luồng log và kiểm tra lại sau. |
| `ERROR` hoặc `CANCEL_ACKNOWLEDGED` | Kiểm log của đúng version và lưu lỗi đầu tiên có thể hành động. Sửa nguyên nhân nếu thuộc notebook, tận dụng checkpoint còn hợp lệ, rồi gửi version mới. |
| `COMPLETE` | Kiểm `kernels files`, tải artifacts về thư mục mới và đối chiếu receipt ở mục 2. Nếu output còn thiếu, kiểm tra danh sách file và log export trước khi chấm. |

```powershell
# Chạy riêng lệnh của notebook đã COMPLETE và đã liệt kê các file cần tải.
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$vpOut = "work/kaggle/videoprism_msr_output_$stamp"
$qwenOut = "work/kaggle/qwen_msr_output_$stamp"
& $kg kernels output $vpRef -p $vpOut --file-pattern '^dev/'
& $kg kernels output $qwenRef -p $qwenOut --file-pattern '^qwen_receipt\.json$|^query_plan_cache\.jsonl$'
```

- [ ] **VideoPrism**. Xác minh shard tại `$vpOut/dev` theo mục 2.2. Gán đường dẫn này vào `$vp` trong lệnh benchmark và chấm R01, R11 ở K=10 và K=20. Kiểm receipt của từng replay để xác nhận VP thật đã chấm query. *(Hiện tại Kaggle `lvhuylong/videntra-msr-videoprism-dev-20261115` đang `CANCEL_ACKNOWLEDGED` — chờ artifact).*
- [x] **Qwen**. Đã tải và xác minh receipt/cache cho cả `extractive-v2` (`work/kaggle/qwen_msr_extractive_v2_output_20261116/`) và LoRA SFT `qwen2.5-1.5b-lora-sft-v1` (`work/kaggle/qwen_msr_lora_output_20261116/`). Đã chạy xong R00, R10, Q00, Q10, L00, L10 ở cả K=10 và K=20 kèm so sánh ghép cặp trên `tune`, `confirm` và `dev_full_859`.
- [x] **Báo kết quả**. Đã cập nhật bảng trạng thái bên dưới, [benchmark/msrvtt_clip_qwen_videoprism_audit.md](benchmark/msrvtt_clip_qwen_videoprism_audit.md) và [docs/PROGRESS.md](docs/PROGRESS.md).

## 1. Mốc kiểm chứng và thứ tự làm

Snapshot audit có 1.324 video trong gallery. `dev_full_859` có 809 query MSR-VTT (khóa partition `tune`/`confirm` 70/30 theo `source_group`, seed `20261115`, SHA-256 `f641d28bc96a66575bc15106422e389088e1228650a7e3135d7062bfbc3b68cd`; `tune=566`, `confirm=243` MSR-VTT). A5 dùng LTR theo fold OOF cho 742/859 query toàn dev, trọng số cuối cho phần còn lại theo provenance gate, và temporal rerank `lambda=0`.

Sau khi sửa 2 lỗi truyền `object_requirements` / `sub_clauses` và bổ sung **SigLIP2 Dual-Rank Frame+Clip Consensus Boost (`_apply_clip_consensus_boost`)** ở Stage 6 (`A5`), baseline A5 rules mới đạt **331/809 (40,91%)** (`+2,35 pp`, `21W / 2L` so với mốc cũ `312/809`; trong đó `tune`: `226 → 238/566 [+2,12 pp, 14W/2L]`, `confirm`: `86 → 93/243 [+2,88 pp, 7W/0L]`, QVH giữ `19/22`, AVC giữ `26/28`).

| Đã đo trên MSR-VTT (`16/11/2026`) | Rules A5 (`R00`/`R10`) | Qwen `extractive-v2` (`Q00`/`Q10`) | Qwen LoRA SFT (`L00`/`L10`) | Kết luận |
| --- | ---: | ---: | ---: | --- |
| Hit@1, `tune` (`n=566`), K=10 & K=20 | **238/566 (42,05%)** | **238/566 (42,05%)** (`0W/0L`, CI `[0.0, 0.0]`) | 237/566 (41,87%) (`0W/1L`, `−0,18 pp`) | `Q00 ≤ R00` → đã kích hoạt LoRA SFT |
| Hit@1, `confirm` (`n=243`), K=10 & K=20 | **93/243 (38,27%)** | **93/243 (38,27%)** (`0W/0L`, CI `[0.0, 0.0]`) | **93/243 (38,27%)** (`0W/0L`, CI `[0.0, 0.0]`) | Không tụt trên `confirm`, nhưng không vượt `+1 pp` → **giữ `rules` mặc định** |
| Hit@1, `dev_809` (`n=809`), K=10 & K=20 | **331/809 (40,91%)** | **331/809 (40,91%)** (`0W/0L`) | 330/809 (40,79%) (`0W/1L`, `−0,12 pp`) | Xóa bỏ hoàn toàn sụt giảm `−7,42 pp` của `compact-v1` |
| Recall@10 (A5 → Clip reserve) | 544 → **555/809** (`+15/−4`) | 543 → **554/809** (`+15/−4`) | 544 → **554/809** (`+14/−4`) | Clip reserve tăng coverage ổn định ở K=10 |
| Recall@20 (A5 → Clip reserve) | 607 → **611/809** (`+9/−5`) | 606 → **611/809** (`+10/−5`) | 605 → **611/809** (`+10/−4`) | Clip reserve tăng coverage ổn định ở K=20 |

| Notebook Kaggle | Trạng thái | Kết quả đã xác minh (`16/11/2026`) |
| --- | --- | --- |
| [VideoPrism v2](https://www.kaggle.com/code/lvhuylong/videntra-msr-videoprism-dev-20261115) | `CANCEL_ACKNOWLEDGED` | Chưa có shard output; `R01`, `R11`, `Q11` đang **`chờ artifact`**. |
| [Qwen2.5-1.5B `compact-v1`](https://www.kaggle.com/code/lvhuylong/videntra-msr-qwen15b-dev-20261115) | `COMPLETE` | Mốc lịch sử: A5 rules 312/809 xuống Qwen `compact-v1` 252/809 (`−7,42 pp`, `16W/76L`). |
| [`extractive-v2`](https://www.kaggle.com/code/lvhuylong/videntra-msr-qwen15b-extractive-v2-20261115) v5 | `COMPLETE` | T4×2 `849.412s`, `847` entries (`cache_sha256: e917f744...`); capture `cache_hits=205, fallback=617, rule_tier=37`; Hit@1 ngang `rules` (`331/809`, `0W/0L`). |
| [LoRA SFT `qwen2.5-1.5b-lora-sft-v1`](https://www.kaggle.com/code/lvhuylong/videntra-msr-qwen15b-lora-sft-20261116) v2 | `COMPLETE` | T4 `1209.596s`, SFT chỉ từ `tune` (`626/110`), `train_loss=0.0328`, `eval_loss: 0.0588 → 0.0408`; xuất `812` plans (`valid=821/822`, `cache_sha256: e80b6a05...`); capture `cache_hits=821, fallback=1, rule_tier=37`; Hit@1 `confirm=93/243 (0W/0L)`, `dev_809=330/809 (−0,12 pp, 0W/1L)`. |

Báo cáo đầy đủ tại [benchmark/benchmark_report.md](benchmark/benchmark_report.md) (`run_id: 0e7ec648`) và [benchmark/msrvtt_clip_qwen_videoprism_audit.md](benchmark/msrvtt_clip_qwen_videoprism_audit.md).

1. **P0, ĐÃ HOÀN TẤT.** Hoàn tất validator scope-aware, chạy lại bộ hồi quy, khóa inputs/manifest `tune`/`confirm`, xuất cache `extractive-v2` & LoRA SFT.
2. **P1, ĐÃ HOÀN TẤT PHẦN CLIP/QWEN/LORA (`R00`, `R10`, `Q00`, `Q10`, `L00`, `L10`).** Riêng `R01`/`R11`/`Q11` đang `chờ artifact` VideoPrism từ Kaggle.
3. **P2, ĐÃ TRIỂN KHAI NÂNG CẤP (`SigLIP2 Dual-Rank Frame+Clip Consensus Boost`).** Tăng MSR-VTT `confirm` `+2,88 pp` (`7W/0L`, CI dương) và `heldout_test_201` `+0,99 pp` (`38,81% → 39,80%`), giữ mặc định trong A5 (`rules`).
4. **P3, ĐÃ NGHIỆM THU (`588/588` pytest PASS).** Giữ `rules` + Consensus Boost làm mặc định trên đường chạy trực tiếp.

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
2. Xác minh pin code, checkpoint và tokenizer. Mỗi video dùng 16 frame RGB `288×288` theo recipe đã hash. Lấy frame ordinal từ decoded presentation-order ledger và timestamp từ PTS/time base. Spot-check ít nhất một video ngắn, một video dọc hoặc có rotation, và một video có frame lặp gần biên. [Runbook VideoPrism tại commit audit](https://github.com/LongLeVuHuy/VIDENTRA/blob/56a79c7/VIDENTRA_CLIP_RETRIEVAL_VIDEOPRISM_KAGGLE_RUNBOOK.md)
3. Lấy hợp ứng viên từ **cả K=10 và K=20**, cho A5 và clip. Clip reserve được tính lại theo K nên K=10 có thể chọn một video khác với reserve K=20. Nếu đã có cache cho toàn bộ video MSR-VTT thì đối chiếu cache với hợp ứng viên và chỉ encode những video thiếu trong phạm vi dataset đang thử. Liệt kê asset thiếu, lỗi decode, sai checksum theo ID. Gửi query gốc và manifest inference. Labels, gold IDs và evaluator giữ ở local.
4. Encode, tải shard về thư mục version riêng và xác minh `video_vectors.npy`, `video_rows.jsonl`, `query_vectors.npy`, `query_rows.jsonl`, `receipt.json`. So số dòng, IDs duy nhất, dim hai phía, vectors hữu hạn, norm dương, hashes, recipe, model revision, số lỗi và `dry_run=false`. Ghi receipt hoàn tất sau khi đọc lại shard. Nếu chia shard để resume, khớp pin/recipe/schema và chỉ merge khi IDs duy nhất qua mọi shard.
5. Chạy đủ hai chính sách `full` và `preserve-unscored` ở một cấu hình ban đầu để định lượng coverage. Theo audit, shard chỉ có MSR-VTT cho K=20 có tối đa 22/809 pool A5 đủ toàn bộ vectors. Giữ mẫu số 809 và ghi hit ở các lát cắt có coverage. Với mục tiêu đo VideoPrism trên toàn gallery, chuẩn bị thêm asset AVC/QVHighlights được phép sử dụng để lấp phần thiếu, giữ cùng recipe và kiểm lại baseline.

### 2.3. Qwen — khóa đầu ra phân rã trước khi chấm ranking

1. Tải cache và receipt của phiên GPU `extractive-v2` sau khi hoàn thành. Đối chiếu revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, weights SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`, tokenizer hash, prompt version `extractive-v2`, input hash và artifact hash. Checkpoint này khớp SHA-256 của weights local đã kiểm trong audit. Kiểm tương tự cho cache LoRA và thêm adapter hash/revision. [Model chính thức](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
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

Lệnh mẫu chạy sau khi validator qua kiểm thử. Gán `VIDENTRA_QWEN_CACHE_PATH` và `VIDENTRA_QWEN_RECEIPT_PATH` về cache và receipt `extractive-v2` mới, cùng `VIDENTRA_MSRVTT_FILTER_MANIFEST` về manifest tune/confirm đã khóa. Gán `VIDENTRA_VIDEOPRISM_SHARD_DIR` khi shard đã sẵn sàng để chạy thêm R01. Mỗi lần chạy tạo thư mục mới.

```powershell
$ErrorActionPreference = 'Stop'
$py = '.\.venv\Scripts\python.exe'
$root = 'work/experiments/msrvtt_stage2_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$vp = $env:VIDENTRA_VIDEOPRISM_SHARD_DIR
$qwen = $env:VIDENTRA_QWEN_CACHE_PATH
$qwenReceipt = $env:VIDENTRA_QWEN_RECEIPT_PATH
$partitionManifest = $env:VIDENTRA_MSRVTT_FILTER_MANIFEST
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

# Qwen cache đã qua kiểm tra artifact
if ($qwen -and $partitionManifest) {
    if (-not $qwenReceipt) { throw 'Gán VIDENTRA_QWEN_RECEIPT_PATH trước khi chạy đối chứng Qwen' }
    Invoke-Bench scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --temporal-rerank-lambda 0 --query-plan-cache $qwen --query-plan-receipt $qwenReceipt --query-plan-revision $qwenRev --query-plan-prompt-version extractive-v2 --out-dir "$root/qwen/pools" --eval-dir "$root/qwen/evaluation"
    foreach ($k in 10, 20) {
        if ($k -eq 10) {
            Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/rules/pools" --pool a5 --k $k --alpha 0 --out-dir "$root/R00_k$k"
        }
        Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/qwen/pools" --pool a5 --k $k --alpha 0 --out-dir "$root/Q00_k$k"
        Invoke-Bench scripts/compare_msr_runs.py --baseline "$root/R00_k$k" --challenger "$root/Q00_k$k" --labels $labels --split-lock $lock --filter-manifest $partitionManifest --filter-split tune --out-dir "$root/compare_R00_Q00_tune_k$k"
    }
}

# VP đã qua kiểm tra artifact
if ($vp) {
    Invoke-Bench scripts/run_clip_vp_replay.py --pools-dir "$root/rules/pools" --pool a5 --k 20 --alpha 0.60 --vp-dir $vp --vp-coverage-policy preserve-unscored --out-dir "$root/R01_k20_a060"
    $receipt = Get-Content "$root/R01_k20_a060/run_receipt.json" -Raw | ConvertFrom-Json
    if ($receipt.vp_dry_run -ne $false -or -not $receipt.vp_receipt_sha256 -or
        (($receipt.vp_coverage_count + $receipt.vp_partial_count) -eq 0)) {
        throw 'Replay chưa chấm được query bằng VideoPrism thật'
    }
    Invoke-Bench scripts/compare_msr_runs.py --baseline "$root/R00_k20" --challenger "$root/R01_k20_a060" --labels $labels --split-lock $lock --out-dir "$root/compare_R00_R01_k20_a060"
}
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

- [x] Xác nhận artifact thật, provenance, độ phủ và câu query gốc cho mỗi challenger. Đã ghi số query dùng plan và fallback cho cả `extractive-v2` (`205` cache hits / `617` fallback) và `lora-sft-v1` (`821` cache hits / `1` fallback). Riêng so sánh VP (`R01`/`R11`/`Q11`) đang chờ shard VideoPrism Kaggle.
- [ ] Lưu cấu hình cùng tiêu chí trước khi mở phần confirm. Mốc đề xuất cho hiệu quả thực dụng là Hit@1 tăng ít nhất 1 điểm phần trăm trên confirm và CI ghép cặp 95% nằm trên 0. *(Đã đạt với **SigLIP2 Dual-Rank Frame+Clip Consensus Boost** trên `rules`: confirm `+2,88 pp`, `7W/0L`; chưa đạt với Qwen `extractive-v2` (`+0,00 pp`) và LoRA SFT (`+0,00 pp confirm`, `−0,18 pp tune`) nên giữ `rules` mặc định).*
- [x] Kiểm QVHighlights 22 query, AVC 28 query và moment trên 22 query có nhãn ở dev (`dev_150`, `dev_full_859`), đồng thời chạy split hồi quy lịch sử `heldout_test_201` trên CPU (`run_id: 0e7ec648`: `dev_150=62,00%`, `dev_full_859=43,77%`, `heldout_test_201=39,80%`).
- [x] Chạy các test liên quan và full suite (`.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` -> **`588 passed, 3 warnings in 90.09s`**), đã bổ sung 3 regression test mới.
- [x] Lưu output từng run dưới `work/experiments/`, logs/receipts Kaggle (`work/kaggle/qwen_msr_extractive_v2_output_20261116/`, `work/kaggle/qwen_msr_lora_output_20261116/`), và cập nhật kết quả bảng trong `benchmark/`, `docs/PROGRESS.md`, `README.md`, `PROJECT.md`. Trạng thái các so sánh VP (`R01`, `R11`, `Q11`) được ghi rõ là `chờ artifact`.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/evaluation/test_clip_vp_replay.py tests/evaluation/test_split_lock.py tests/indexing/test_videoprism_kaggle_pipeline.py tests/query/test_llm_decomposer.py tests/retrieval/test_clip_branch.py
```
