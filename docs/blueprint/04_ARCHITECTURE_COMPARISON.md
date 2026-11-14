# 04 — Architecture comparison và decision matrix

Tất cả đánh giá qualitative dưới đây là **engineering hypotheses**, không measured leaderboards. Cùng corpus, encoder revision, qrels, budget và hardware được dùng khi so E06/E08/E10. [Pipeline chuẩn](05_VIDENTRA_DETAILED_PIPELINE.md).

## Ba kiến trúc

| Dimension | A — Basic Hybrid | B — Query-Aware Relation | C — Advanced Video Understanding |
|---|---|---|---|
| Retrieval | SigLIP+OCR+ASR independent union | A + conditional graph/routing | B + video-native proposals/reranker |
| Recall | strong baseline multimodal opportunity | có thể tăng clauses; routing có thể giảm nếu sai | không tự cứu candidate đã miss; thêm branch nếu justify |
| Temporal localization | fixed/shot/utterance windows | bounded multiscale + ordered-event matching | learned grounding native features |
| Semantic relations | co-timing, coarse similarity | role constraints; validated attribute/2D/time; unknown action nếu thiếu evidence | action/video models có khả năng hơn nhưng hallucination/domain shift |
| OCR/ASR | timestamped independent branches | query predicates + exact/semantic distinction | giống B; video model không thay verbatim OCR/ASR |
| Online latency | thấp nhất tương đối | parser/joins/refine overhead, deadlines | model load/decoding/token cost cao |
| Offline compute | decode+3 modality extraction | A + optional targeted metadata | video features/tracking/graphs có thể lớn |
| RAM/storage | FAISS+text+metadata | thêm graph evidence/cache | thêm temporal features/checkpoints |
| Engineering | thấp nhất | trung bình; explicit semantics contracts | cao; feature compatibility/training/version coupling |
| Maintainability | dễ debug | traceable nhưng nhiều schema/ontology tests | model adapters cần quality gates riêng |
| 6GB feasibility | likely batch1/sequential, chưa đo | target configuration; optional parts CPU/deferred | không default resident; quantization/candidate cap, có thể fail SLO |
| Competition | tốt cho MVP và scan nhanh | chọn làm target incremental | chỉ plugin Accurate nếu net benefit |
| Main risk | object co-occurrence false positives | parse/binding errors và over-filtering | cost tăng mà relation vẫn sai |

## Lý do chọn B tạm thời

A cần để đo mọi gain và cung cấp fallback. B xử lý lỗi về vai trò/order bằng representation/constraints mà không cần full corpus VLM. Với6GB, C chưa có bằng chứng cost/quality phù hợp. Không tuyên bố B đạt chất lượng “comparable C” trước chạy thử.

### Decision gates thay cho điểm số giả

1. Hard gates: labels isolated; exact timeline; no visual bottleneck; RAM/VRAM caps; recoverable snapshot.
2. E06: independent union vs visual-first tại matched total candidate budget. Cần Candidate Recall@1000≥0,95 trên judged dev nếu feasible; nếu chưa đạt thì chưa tune reranker che candidate misses.
3. E07: original+graph không giảm original-only Recall@10 quá1pp ở bất kỳ stratum; parser role/negation≥95% trên heldout graph set là target, chưa đạt thì soft advice only.
4. E08: B so A pairwise hard-negative accuracy tăng≥5pp với bootstrap CI không cắt0 và Moment Recall không giảm>1pp; threshold là go/no-go đề xuất.
5. C chỉ promote nếu tăng≥3pp Moment Recall@10 IoU0,5 hoặc≥5pp relation-pair accuracy so B trên independent set, không vi phạm deadlines/caps. Nếu cùng quality trong CI mà C tốn nhiều hơn, giữ B.
6. Nếu một hard gate fail, ưu tiên correctness rồi thu hẹp capability/corpus; không hạ recall silently chỉ để đạt latency.

## Candidate generation strategies

| Strategy | Ưu | Failure | Vai trò |
|---|---|---|---|
| Visual-first | ít I/O/text searches | correct text/speech moment ngoài visualTopK bị mất hẳn | ablation bắt buộc, không production default |
| Parallel modalities | coverage dễ kiểm soát, rescue independent | union lớn, score scales khác nhau | baseline A |
| Adaptive routing | giảm irrelevant branches/compute | parser/LID sai gây recall loss | B, branch floors và fallback |
| Hierarchical hybrid | coarse index giúp large corpus | coarse gate gây cascading misses | E06 challenger; always independent text bypass |

So K200/500/1000/3000, temporal coverage tIoU0,3/0,5, branch-rescue rate, union size, per-stratum recall và p95. RRF baseline không cần confidence calibration; query-aware weights chỉ train/dev. Scoring full video evidence không thay time-compatible evidence join.

## Các phương án quan hệ

Rule-only graph giúp scope và role consistency nhưng không nhận biết physical action. Learned scene graphs làm chi phí offline và annotation ontology tăng; cross-graph papers là evidence research chứ chưa ready component. Text cross-encoder chỉ đọc text. Native grounding model cần đúng feature encoder/checkpoint và không mặc định corpus-level. Target VLM kiểm tra rất ít candidates, phải chịu counterfactual tests. Chọn progressive assessments ở pipeline§13.10; mỗi relation có capability flag và confidence limitations.

## Hardware deployment alternatives

- **Preferred first:** one3060/32GB, CPU FAISS/text, GPU serial offline, warm query text tower; minimal service footprint.
- **Scale option:** media trên external store, indexes CPU shards; shared training/build jobs ở máy khác chỉ khi user chọn resources.
- **Not selected:** all models resident6GB, full corpus VLM captions+graphs, GPU Flat index hàng chục triệu vectors.
- Qdrant comparison chỉ khi cần distributed serving/filter/operations; licensing/client/runtime pin trước adoption, không thêm dependency cho planning.

Revisit B/C sau M6/E08; revisit shard/index sau1M vectors, sau corpus drift, hoặc concurrency thay đổi. Chưa thể quyết định deployment10k h cuối cùng nếu chưa có storage và throughput measurements.
