# VIDENTRA — Runbook truy hồi clip và VideoPrism‑LvT‑B trên Kaggle

Ngày 15/11/2026. Runbook dành cho Antigravity, dựa trên pipeline tại `3a83b4c` và lần kiểm tra `35f7c01`. Mục tiêu là xác định độc lập hai hiệu ứng. Nhánh clip có đưa thêm video đúng vào shortlist hay không. VideoPrism có xếp video đúng lên hạng 1 trên **cùng shortlist** hay không. Các script và cờ đánh dấu **đề xuất** là phần cần triển khai. [Kế hoạch MSR‑VTT trước đó](VIDENTRA_MSRVTT_ACCURACY_AND_TUNING_PLAN.md) · [Nhật ký probe](docs/PROGRESS.md)

> Bản audit 15/11/2026 bổ sung kết quả replay đã chạy lại, kiểm tra receipt của worker, rerank khi thiếu vector và đối chứng query plans bằng Qwen2.5-1.5B. [Audit và lệnh tái lập](benchmark/msrvtt_clip_qwen_videoprism_audit.md)

## 1. Mốc cần tái lập

Lấy run `8c40e943d7104f21badda4a3fd62f623` làm mốc A5. Chỉ số chính là **MSR‑VTT global Video Hit@1** trên gallery 1.324 video, tính một lần cho mỗi query theo nhãn video hiện hành. Dev đầy đủ gồm 809 query MSR‑VTT trong `dev_full_859`. Held-out lịch sử gồm 191 query MSR‑VTT trong `heldout_test_201`. [Benchmark JSON](benchmark/benchmark_summary.json)

| Mốc | Kết quả MSR‑VTT | Dùng để kiểm tra |
| --- | --- | --- |
| A5 mặc định trên dev | Hit@1 **312/809 (38,57%)**, Hit@5 **486/809**, Hit@10 **544/809** | Đối chứng end-to-end. |
| Shortlist A5 trước LTR trên dev | Target ở Top‑10 **544/809**, Top‑20 **606/809**, Top‑50 **683/809**, union **695/809** | Chỉ báo dư địa theo độ sâu. Khi đo reranker, dùng **pool thực tế** mà reranker nhận sau A5 và ghi recall của pool đó. |
| Probe clip reserve trên dev | Giữ 9 video đầu của A5 và dành 1 chỗ cho nhánh frame + clip. Top‑10 **555/809** (+15 ca cứu, −4 ca mất), Top‑1 giữ **312/809** | Mốc kiểm tra nhánh bổ sung ở cùng K. |
| A5 mặc định trên held-out lịch sử | Hit@1 **71/191 (37,17%)** | Mốc hồi quy sau khi chốt cấu hình. |

Baseline phải dùng cùng corpus, query IDs, nhãn, FAISS, SQLite, model LTR và OOF folds. Đặt temporal rerank `lambda=0.0` như run mốc. Giữ trạng thái patches/caption nhất quán giữa các biến thể. Nếu snapshot đã bổ sung chúng, tạo baseline mới với cấu hình được ghi rõ, rồi chạy mọi challenger cùng snapshot. Bao gồm hash query set, split manifest, model weights, DB, index, feature flags và git commit trong mỗi receipt.

Lệnh dưới đây dùng CLI hiện có. Chỉ chấm dev đầy đủ và tạo thư mục mới. Đọc `per_query_evaluation.jsonl` và `dev_full_evaluation.jsonl` trong output. Trên cùng snapshot, đối chiếu cả thứ tự `ranked_videos` theo `query_id` với archive, kèm giải thích nếu lệch. [CLI runner](scripts/run_validation_benchmark.py#L1874-L1906)

```powershell
.\.venv\Scripts\python.exe -B scripts/run_validation_benchmark.py --splits dev_full_859 --temporal-rerank-lambda 0.0 --out-dir work/experiments/clip_vp_a5_reference
```

## 2. Nhánh frame + clip từ SigLIP 2 hiện có

### 2.1. Tạo gallery và tính điểm

Tách scoring vào module nhỏ dưới `src/videntra/retrieval/`, dùng lại `all_embeddings`, `video_kf_fids`, `video_keyframes`/`KeyframeMap` và `primary_q_768` của `UnifiedMultimodalSearchPipeline`. Với corpus hiện tại, một dot product NumPy và reduce theo `video_id` đủ cho pilot. Tạo ma trận clip mean một lần khi nạp snapshot. Nếu persist, lưu mapping và hash bên cạnh để khởi tạo lại đúng snapshot. [Khởi tạo pipeline](src/videntra/retrieval/unified_pipeline.py#L504-L596) · [Vector query 768d](src/videntra/retrieval/unified_pipeline.py#L897-L905)

```text
x_f = L2_normalize(frame_vector[f])
q = L2_normalize(0.75 * encode(original_query) + 0.25 * encode(frame_prompt))
clip_mean[v] = L2_normalize(mean(x_f for f in video v))
frame_score[v] = max(dot(q, x_f) for f in video v)
clip_score[v] = dot(q, clip_mean[v])
rank_frame, rank_clip = stable descending ranks by score, then video_id
branch_score[v] = 1 / (60 + rank_frame[v]) + 1 / (60 + rank_clip[v])
```

- Rank bắt đầu từ 1 trên cùng gallery hợp lệ với filter của request. Frame score tính trên toàn frame bank trong probe tái lập. Top‑120 frame hits của A5 cho thứ hạng frame khác. Log nếu đổi sang một phép xấp xỉ và đánh giá nó riêng.
- Chỉ nhận vector hữu hạn, đúng dim từ snapshot đang dùng. Dùng các FAISS row IDs đã map. Video rỗng hoặc norm quá nhỏ ghi trạng thái thiếu feature. Tie-break cố định bằng `video_id`. Frame/clip rank, điểm thành phần và video/gallery coverage phục vụ audit.
- Sau bản tái lập clip mean, có thể thử cửa sổ 3 keyframe theo presentation order. Tính mean đã chuẩn hóa trên mỗi cửa sổ rồi lấy điểm max theo video. Đo riêng ảnh hưởng của cửa sổ và tránh dùng PTS từ bất kỳ nhãn moment nào. Probe trước cho thấy cửa sổ 3 frame có tín hiệu Top‑1 nhẹ trên dev, đủ làm đối chứng bổ sung. [Số liệu probe](docs/PROGRESS.md#15112026--đối-chiếu-temporal-rerank-và-thử-nhánh-truy-hồi-clip-cho-msr-vtt)

### 2.2. Chụp pool A5 và áp dụng clip reserve

Chụp hai danh sách **trước khi chạy VideoPrism**. `a5_pool` ưu tiên thứ tự `ranked_videos` của A5 ở 10 dòng đầu để giữ đúng baseline, sau đó nối các video khác từ danh sách `all_candidate_moments` đã sort sau LTR đến K tối đa. Điều này cho phép replay ở K=10 và K=20 mà giữ nguyên head của A5. Không lấy K=20 bằng cách đổi `top_k` của một run rồi mặc định rằng 10 vị trí đầu giống nhau. Ghi `query_id`, thứ tự, video IDs, score đầy đủ, provenance và anchor có thể truy ngược. [Contract video/moment](src/videntra/retrieval/unified_pipeline.py#L1150-L1200) · [Stage 5 sau LTR](src/videntra/retrieval/unified_pipeline.py#L3200-L3244)

```text
protected = a5_pool[:K-1]
tail = first video in branch order that is absent from protected
clip_pool = protected + [tail] if tail exists else a5_pool[:K]
```

- Yêu cầu `K ≥ 2` cho policy reserve này. Nếu video ở `tail` trùng video thứ K của A5 thì pool giữ nguyên. Log `added_video`, `displaced_video`, `duplicate_skipped`, rank nhánh và lý do fallback. Nếu pool thiếu K vì gallery nhỏ, giữ đúng số video hợp lệ và báo pool size thực.
- **Replay đầu tiên** dùng export pool đã đóng băng và chạy offline trên CPU. Nhánh clip chỉ đổi thành viên ở đuôi, do đó Hit@1 của `clip_reserve` phải bằng baseline từng query và Recall@K mới là chỉ số chính của bước này. Giữ `K=10` để tái lập probe và chạy lại `K=20` để so ở ngân sách rerank lớn hơn. Dùng `reserve=1` trước khi cân nhắc `2`.
- **Tích hợp SSOT khi đã có bằng chứng.** Dùng helper tính nhánh và reranker trong `UnifiedMultimodalSearchPipeline`, giữ tính opt-in. Clip-only candidate cần một `RankedMoment` hợp lệ, gắn đại diện với frame có score visual cao nhất hoặc cửa sổ clip đã truy hồi, rồi dùng chung đường refine/resolution theo PTS. Trong cấu hình reserve này, mức điểm video clip-only không tự thay thứ tự head. Reranker chạy trước bước chọn video để refine Stage 6. Khi thắng Top‑1, ứng viên mới phải có thời điểm phát và evidence đúng video. Trace rõ `a5_pool_ids`, `clip_pool_ids`, `reranker_input_ids` và `ranked_video_ids`. Đo E2E mới sau tích hợp để phát hiện tác động của fusion hoặc moment refinement.

**Test bắt buộc.** Branch off giữ output A5 từng query. Reserve không đổi Top‑1, tối đa K video duy nhất, giữ thứ tự K−1 vị trí đầu. Thêm test filter dataset, tie, thiếu candidate, snapshot mismatch, clip-only video lên đầu sau rerank và quan hệ `ranked_moments`/`ranked_videos` đúng với kết quả cuối.

## 3. Chuẩn bị dữ liệu Kaggle theo pool đã khóa

1. Từ dev, tạo `queries.jsonl` dạng `query_id`, `query_text` và `candidates.jsonl` dạng `query_id`, `candidate_video_ids`. Dùng union của **A5 pool và clip pool tại K=20** đã dự kiến để gom video cần encode. Tạo `media_manifest.jsonl` chỉ với asset được dùng trong benchmark, gồm `video_id`, relative path, media SHA‑256, timeline revision, clip bounds và basis. Có thể lấy query từ loader hiện hành nhưng export bằng danh sách trường cho phép. Evaluator giữ `target_videos`, `target_windows`, annotation, split labels và nhãn đúng/sai ở local. Kaggle chỉ cần query text và tập video phải encode. Debug theo query IDs đã khóa và video assets từ candidate sets, kể cả ca candidate thiếu media. [Loader](scripts/run_validation_benchmark.py#L291-L376) · [Ví dụ project tách query inference khỏi evaluation](scripts/export_qvh_inference_queries.py)
2. Dùng nguồn clip/frame decode gốc tương ứng với snapshot. Nếu gửi frame bundle thay media, xuất **16 frame** từ source ledger cho mỗi video với cùng recipe và lưu từng ordinal, PTS, time base. `package_kaggle_inputs.py --mode db` hiện đóng gói keyframe mẫu cho các tác vụ khác. Chuỗi 16 frame cho LvT‑B cần một manifest/bộ lấy mẫu riêng, có thể tái dùng decoder và cơ chế registry trong project. Giữ gallery đánh giá đầy đủ kể cả video chưa có vector VideoPrism. [PTS ledger](src/videntra/timeline/ledger.py) · [Time contracts](docs/blueprint/06_DATA_SCHEMAS_AND_TIMESTAMPS.md)
3. Recipe pilot dùng 16 điểm cách đều trong khoảng clip hợp lệ `[start,end)`, lấy tâm từng bin `start + (i+0.5)*(end-start)/16`. Chọn decoded presentation-order frame gần timestamp yêu cầu, chọn ordinal nhỏ hơn khi bằng nhau và ghi timestamp thực. Cho phép frame lặp một cách xác định nếu clip quá ngắn, báo `unique_frames`, thiếu mapping hoặc lỗi decode. Mỗi video có **một** vector toàn clip trong lượt đầu. Đây là recipe của VIDENTRA. Thử cửa sổ cục bộ theo anchors ở một ablation riêng sau khi đo toàn clip. Baseline và challenger luôn dùng cùng video IDs đã đóng băng.
4. Mở notebook Kaggle riêng, chọn GPU T4 nếu khả dụng, attach bundle và checkpoint đã pin dưới `/kaggle/input`. Data mount dùng để đọc, shard output mới ghi vào `/kaggle/working`. Pilot 5–10 video và 4 query bất kỳ, lấy mẫu theo ID cố định hoặc tiêu chí chất lượng hình/decoder không dùng nhãn. Sau pilot mới encode toàn bộ video/query set của dev. Phiên test dùng bundle cùng recipe, sau khi freeze config cuối.

## 4. VideoPrism‑LvT‑B trên Kaggle

### 4.1. Phiên bản và preflight

| Thành phần | Giá trị đã đối chiếu |
| --- | --- |
| Repo JAX/Flax | `google-deepmind/videoprism@4f831e332b80d6e77bdd43882206f23aba2a67a7` |
| Config public | `videoprism_lvt_public_v1_base` |
| HF repo và revision | `google/videoprism-lvt-base-f16r288@0119cca7d1fe61aa1d5718e1f71605703e1562e7` |
| Checkpoint | `flax_lvt_base_f16r288_repeated.npz`, 990.527.124 bytes, SHA‑256 `dcc9bd31ba063221ca952e5a22370236018460faa09aa1a5649c6ccfaf071180` (LFS metadata) |
| Tokenizer | `c4_en` từ `t5-data/vocabs/cc_en.32000/sentencepiece.model`, SHA‑256 `1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec` |
| License model card | `apache-2.0`. Ghi riêng license code, tokenizer và nguồn media trong manifest của run. |

[Repo commit](https://github.com/google-deepmind/videoprism/tree/4f831e332b80d6e77bdd43882206f23aba2a67a7) · [Demo video‑text](https://github.com/google-deepmind/videoprism/blob/4f831e332b80d6e77bdd43882206f23aba2a67a7/videoprism/colabs/videoprism_video_text_demo.ipynb) · [Model card đã pin](https://huggingface.co/google/videoprism-lvt-base-f16r288/tree/0119cca7d1fe61aa1d5718e1f71605703e1562e7) · [Tokenizer chính thức](https://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model)

- Dùng environment notebook tách khỏi `.venv` local. Chọn wheel CUDA JAX tương thích với `nvidia-smi` và Python notebook, chẳng hạn `jax[cuda12]` khi driver phù hợp. Ghi phiên bản đã resolve trong lock, chạy `pip check`, sau đó dùng cùng lock cho các lần chạy đối chứng. Nếu môi trường đã có TensorFlow nên dùng một bản tương thích trong environment đó. Kiểm tra `jax.devices()` có GPU trước encode. Bắt đầu batch 1 với FP32 và batch size cố định cho từng pass. Đo compile riêng. Tăng batch từng bước khi đã thấy peak memory còn dư. [Cài JAX GPU](https://docs.jax.dev/en/latest/installation.html) · [Dependencies tại commit](https://github.com/google-deepmind/videoprism/blob/4f831e332b80d6e77bdd43882206f23aba2a67a7/requirements.txt)
- Prepare checkpoint bằng `hf_hub_download(repo_id=..., filename=..., revision=...)` và tải tokenizer từ URL chính thức ở bước chuẩn bị artifact. Xác minh SHA‑256 thực tế trước khi bundle. Chạy notebook bằng file local đã xác minh. Lệnh cài repo dưới đây dùng pin đã đối chiếu, chạy trong môi trường Kaggle sau khi chọn JAX wheel. Ghi `nvidia-smi`, `pip freeze`, version PyAV/FFmpeg và device thực trong receipt.

```bash
python -m pip install 'videoprism @ git+https://github.com/google-deepmind/videoprism.git@4f831e332b80d6e77bdd43882206f23aba2a67a7' av pillow
python -m pip check
```

### 4.2. Encode từng phía và export

RGB input có shape `[B,16,288,288,3]`, float32 trong `[0,1]`, presentation order đúng. Demo gốc làm việc với aspect ratio tương thích. Với video ngang/dọc của VIDENTRA, pilot dùng **resize giữ tỷ lệ rồi pad đen** vào 288×288 (bilinear, tâm 0,5/0,5), ghi recipe và hash. Áp rotation/display transform đã xác nhận nhất quán với media snapshot. Text dùng original query tiếng Anh từ MSR‑VTT và `vp.tokenize_texts` với `c4_en`. Giữ cùng canonicalization, token padding và revision cho tất cả cấu hình.

Mẫu cell sau minh họa API đã đối chiếu trong upstream. Sau khi nạp `clip_batch` hoặc query batch từ bundle, gọi các hàm này độc lập. Lấy **global vector** từ checkpoint LvT ở cả hai phía và xác minh cùng dim. [Upstream models.py](https://github.com/google-deepmind/videoprism/blob/4f831e332b80d6e77bdd43882206f23aba2a67a7/videoprism/models.py) · [VideoPrism paper](https://arxiv.org/abs/2402.13217)

```python
from pathlib import Path
import hashlib

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import jax
import numpy as np
from videoprism import models as vp
from videoprism import tokenizers as vp_tokenizers

assert any(device.platform == "gpu" for device in jax.devices()), jax.devices()
assets = Path("/kaggle/input/videntra-videoprism-pins")
checkpoint = assets / "flax_lvt_base_f16r288_repeated.npz"
tokenizer_file = assets / "sentencepiece.model"

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

assert sha256(checkpoint) == "dcc9bd31ba063221ca952e5a22370236018460faa09aa1a5649c6ccfaf071180"
assert sha256(tokenizer_file) == "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"

model_name = "videoprism_lvt_public_v1_base"
model = vp.get_model(model_name)
state = vp.load_pretrained_weights(model_name, checkpoint_path=str(checkpoint))
tokenizer = vp_tokenizers.SentencePieceTokenizer(str(tokenizer_file))

@jax.jit
def video_forward(frames):
    return model.apply(state, frames, None, None, train=False)[0]

@jax.jit
def text_forward(token_ids, token_paddings):
    return model.apply(state, None, token_ids, token_paddings, train=False)[1]

def normalize_rows(embeddings):
    rows = np.asarray(jax.device_get(embeddings), dtype=np.float32)
    if rows.ndim != 2 or rows.size == 0:
        raise ValueError(f"Expected [batch, embedding_dim], got {rows.shape}")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-8):
        raise ValueError("VideoPrism returned invalid embedding rows")
    return rows / norms

# video_vectors = normalize_rows(video_forward(clip_batch))
# Encode query batch bằng:
# ids, pads = vp.tokenize_texts(tokenizer, original_queries)
# query_vectors = normalize_rows(text_forward(ids, pads))
```

Mỗi shard ghi `video_vectors.npy`, `video_rows.jsonl` (`row_id`, `video_id`, `media_sha256`, frame ordinals/timestamps, `recipe_hash`), `query_vectors.npy`, `query_rows.jsonl` (`row_id`, `query_id`, `query_text_sha256`) và `receipt.json`. Có thể tách shard video và text. Ghi số rows, dim, dtype, hash từng file, commit, model/tokenizer hash, CPU/GPU và lỗi theo ID. Ghi success sau khi đã verify đọc lại. Resume theo shard hash và recipe, mỗi output dir mới. Trên local, validate unique IDs, exact query hash, media/timeline revisions, vectors hữu hạn và norm trước khi mmap. Kiểm tra một video có frame phức tạp, một clip ngắn, một query text dài và lặp encoder cùng input để kiểm parity. Một mẫu spot-check về nội dung giúp phát hiện mapping nhầm. Điểm model là tín hiệu ranking.

## 5. Rerank và ma trận benchmark từng bước

Chạy replay local bằng `dot(q_vp, v_vp)` từ vectors LvT đã L2 normalize. Giữ hai embedding spaces riêng, hợp nhất bằng thứ hạng. Quy tắc ban đầu dùng `score = (1−α)/(60+r_base) + α/(60+r_vp)` với rank trong chính pool của query, tie-break bằng base rank rồi video ID. `α=0` phải ra base order chính xác. Thử `α=0.25` trước. Mở `{0.10, 0.25, 0.40}` trên dev tune nếu có hiệu ứng. Trong replay, mọi cấu hình ghép cặp dùng cùng pool hoặc chỉ đổi đúng biến theo bảng dưới. Với query thiếu vector/video, giữ base order cho query đó, giữ query trong denominator và ghi fallback. Báo thêm kết quả trên giao coverage để phân biệt chất lượng model với thiếu artifact.

| Lượt | Cấu hình | Giữ cố định | Đọc kết quả |
| --- | --- | --- | --- |
| **1** | `a5_reference`, CLI hiện có | Snapshot, OOF LTR, splits | Tái lập Hit@1 và thứ tự Top‑10. |
| **2** | `clip_reserve` với K=10, sau đó K=20 | Query set, A5 head, K của từng cặp | So với `a5_reference` cùng K. Hit@1 bằng từng dòng. Đo Recall@K và số candidate gains/losses. |
| **3** | `videoprism_on_a5` ở K=10/20 | Chính xác cùng `a5_pool` và coverage policy | So với A5 cùng K để đo VideoPrism đưa thêm đúng lên Top‑1 được bao nhiêu. Có thể chạy `vp_only_rank` như đối chứng, vẫn dùng cùng pool. |
| **4** | `videoprism_on_clip` ở cùng K và α đã chốt trên dev | Cùng rerank logic và model, chỉ đổi pool | So với `videoprism_on_a5` để biết clip recall có chuyển thành Hit@1 hay không. So thêm `clip_reserve` để thấy riêng tác dụng của VideoPrism khi pool đã đổi. |
| **5** | Một cấu hình đã chốt tích hợp qua SSOT | Snapshot/flags đồng nhất, lần lượt baseline và challenger | Chạy `dev_full_859` trước rồi đánh giá split nguồn chưa dùng tune. Đối chiếu cả Video@K, Moment@K, API/CLI và tài nguyên end-to-end. |

- Lượt 2–4 triển khai một harness replay dưới `scripts/` với input `queries`, `a5_pool`, `clip_pool`, VP artifact/receipt, `K` và `alpha` trong config. Inference xuất query/video IDs, thứ tự và scores. Evaluator đọc `evaluation_only` labels sau. Chọn tên và giao diện khớp cấu trúc repo. Ghi lệnh thật vào README khi đã implement.
- Trước sweep, freeze dev tune và dev confirm theo canonical `source_group`, hoặc dùng inner/outer folds nhóm nguồn. `dev_150` nằm trong dev đầy đủ và dùng để smoke. `heldout_test_201` đã được quan sát trong các vòng phát triển, dùng như hồi quy lịch sử sau khi chốt. Nếu có split nguồn mới, freeze bằng `split_lock`, rà gần trùng và chỉ chạy cấu hình đã chốt. Nếu chưa có split mới, công bố mức bằng chứng là dev/held-out lịch sử. Model upstream có benchmark riêng. Kết quả tăng của VIDENTRA đo bằng cùng corpus của hai run tại đây. [Split lock](src/videntra/evaluation/split_lock.py)
- Mỗi pair kiểm tra cùng `query_id`, text hash, tập target, gallery hash, kiểu relevance và flags. Report theo toàn corpus và `dataset=msrvtt`. Lưu `base_top1`, `new_top1`, `target_rank_before`, `target_rank_after`, `base_hit`, `new_hit`, `in_pool_hit`, `fallback_reason` trong evaluator JSONL. Báo Hit@1/5/10, MRR@10, Recall@K, số video được thêm/đẩy ra, coverage, paired CI 95%. Giữ QVH/AVC và moment metrics để phát hiện hồi quy. Khi dùng chúng làm guardrails cho VideoPrism, pack các query tiếng Anh đủ điều kiện của cả hai tập và báo coverage riêng. Query ngoài phạm vi thử nghiệm đi qua baseline với lý do fallback. Kaggle offline encode time và local replay latency được báo riêng. Khi đo end-to-end, tính cả query encoding của VideoPrism theo entry point, hoặc ghi rõ query embeddings đã precompute nếu đo replay.

### Mẫu chạy replay sau khi triển khai

Các lệnh dưới đây minh họa **CLI đề xuất**. Capture dùng cùng loader và A5 OOF như baseline, ghi `queries.jsonl`, `a5_pool.jsonl`, `clip_pool.jsonl` và manifest riêng. Replay đọc ba file này, chọn `--pool a5|clip` và ghi `ranked_video_ids` theo query. Khi `--alpha` bằng 0, replay chỉ cần pool. Evaluator nhận split manifest khóa nguồn và labels qua đường dẫn local riêng.

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_msr_candidate_pools.py --split dev_full_859 --max-k 20 --out-dir work/experiments/clip_vp_pools_dev
.\.venv\Scripts\python.exe -B scripts/run_clip_vp_replay.py --pools-dir work/experiments/clip_vp_pools_dev --pool a5 --k 20 --alpha 0 --out-dir work/experiments/clip_vp_k20_baseline
.\.venv\Scripts\python.exe -B scripts/run_clip_vp_replay.py --pools-dir work/experiments/clip_vp_pools_dev --pool clip --k 20 --alpha 0 --out-dir work/experiments/clip_vp_k20_reserve
.\.venv\Scripts\python.exe -B scripts/run_clip_vp_replay.py --pools-dir work/experiments/clip_vp_pools_dev --pool a5 --k 20 --alpha 0.25 --vp-dir work/kaggle/videoprism_dev --out-dir work/experiments/clip_vp_k20_vp_a5
.\.venv\Scripts\python.exe -B scripts/run_clip_vp_replay.py --pools-dir work/experiments/clip_vp_pools_dev --pool clip --k 20 --alpha 0.25 --vp-dir work/kaggle/videoprism_dev --out-dir work/experiments/clip_vp_k20_vp_clip
```

Lặp các cặp ở K=10 với output dirs mới để đối chiếu mốc 555/809. Chấm từng cặp theo bảng trên. Ví dụ dưới đây đo tác dụng của VideoPrism trên **cùng pool A5**, sau đó dùng cách gọi tương tự cho `reserve` so với `baseline`, `vp_clip` so với `vp_a5` và `vp_clip` so với `reserve`.

```powershell
.\.venv\Scripts\python.exe -B scripts/compare_msr_runs.py --baseline work/experiments/clip_vp_k20_baseline --challenger work/experiments/clip_vp_k20_vp_a5 --labels work/evaluation-only/clip_vp_dev_full_859_labels.jsonl --split-lock work/evaluation-only/clip_vp_dev_full_859_split_lock.json --out-dir work/experiments/clip_vp_k20_compare_vp
```

### Cách kết luận về Hit@1

Tính `wins = #query đối chứng sai → challenger đúng`, `losses = #query đối chứng đúng → challenger sai`. `delta_pp = 100 × (wins−losses) / N`. Với 809 dev queries, thêm ròng 25 câu tương ứng khoảng **+3,09 điểm phần trăm**. Đây có thể là mốc giá trị của pilot. Lấy paired bootstrap 95% theo canonical source group, đề xuất 10.000 lượt và seed `20261115`. Mỗi lượt lấy lại đúng số group có hoàn lại và tính `100 × sum(new_hit − base_hit) / sum(query_count)` trên toàn bộ query trong các group đã chọn. Báo số query, số group, delta, CI và `wins/losses` cùng nhau. Nếu CI còn cắt 0, ghi là tín hiệu pilot cần xác minh. Nếu Recall@K tăng mà Hit@1 giữ nguyên, ghi đúng là tăng độ phủ. Mốc xác nhận đề xuất là cấu hình chốt trước có cận dưới CI lớn hơn 0 trên split nguồn mới. Với held-out lịch sử, diễn giải là kiểm tra hồi quy.

## 6. Bàn giao cho Antigravity

- **Code và tests.** Nhánh clip, hooks replay/SSOT, manifest guard, VideoPrism worker notebook hoặc CLI Kaggle, evaluator ghép cặp. Giữ cấu hình opt-in và bộ test tách ranking video/moment. Chạy tests retrieval, evaluation, split lock, Kaggle guards, rồi full pytest. Kiểm tắt cả hai nhánh, query duplicate, sai hash, thiếu video/query vectors, K=10/20, clip-only candidate ở hạng 1 và retry shard.
- **Artifacts.** Mỗi run có config, hashes, `per_query_evaluation.jsonl`, `benchmark_summary.json`, `benchmark_report.md` và bảng so sánh ghép cặp theo dataset. Giữ raw archive từng run trong `work/experiments`. Khi có run đầy đủ đã xác minh, đồng bộ `benchmark/`, README và `PROGRESS.md` theo quy ước repo. Với run tune chỉ có dev, lưu output riêng để bảng chuẩn vẫn tham chiếu đúng run đầy đủ. Lưu chứng cứ latency/GPU/coverage đã đo. Ghi model pins và dependency lock của notebook kèm code, lưu raw media/checkpoints ngoài Git.
- **Điều kiện đổi mặc định.** Giữ baseline nếu pilot chỉ tăng recall hoặc kết quả Hit@1 còn bất định. Khi tăng Hit@1 được xác nhận, xem thêm paired losses ở QVH/AVC và, nếu phù hợp, OCR/ASR, moment metrics, p95 và peak RAM/VRAM theo ngân sách đã chốt trước thử nghiệm. Giữ đường cấu hình để rerun A5 và rollback snapshot.
