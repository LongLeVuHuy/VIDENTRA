# 08 — Benchmark, ablations và latency plan

**Trạng thái mọi benchmark VIDENTRA: NOT_MEASURED.** Những con số trong gate là design targets; không kết quả. Planning đã chạy file/schema/backup checks, không encoder/search models.

## 1. Metrics và định nghĩa

Với judged relevant set G_q, predicted ranked moments P_q:
- Video Hit@K =1 nếu có known relevant video trongK; average theo query. Recall@K cần denominator relevant videos exhaustive/known scope được ghi.
- Moment Hit@K@tIoUτ =1 nếu đúng video và max interval IoU≥τ, τ∈{0,3;0,5;0,7}. Multi-instance recall = matched relevant moments/known positives, bipartite one-to-one matching tránh một proposal cover hết occurrences.
- tIoU([a,b),[c,d)) = max(0,min(b,d)−max(a,c))/(max(b,d)−min(a,c)).
- MRR reciprocal rank first relevant,0 nếu không hit; mAP/nDCG chỉ khi qrels đủ cho declared task/judgment coverage.
- Boundary absolute error: |s_pred−s_gt| và |e_pred−e_gt| trên matched positives; report unmatched rate riêng, không bỏ failures để làm error thấp.
- Relation pair accuracy: positive moment outranks its controlled negative; ties=0,5; report per role/order/speaker/attribute type. Predicate support precision/recall, unknown coverage, selective risk và false-supported rate.
- OCR CER=(substitution+deletion+insertion)/reference_chars, WER tương tự units; detection/spotting separately. Vietnamese word segmentation convention ghi rõ; whitespace-based syllable error không gọi word-segmented WER mà không nêu.
- ASR normalized/raw CER/WER, timestamp error/coverage và hallucination-in-silence; no-audio không gán WER0.
- Parser Predicate F1 (canonical relation+roles), entity-binding exact accuracy, temporal/negation accuracy, valid-JSON rate, hallucinated predicates/query, residual coverage.
- ANN Recall@K = |ANN_topK∩exact_topK|/K, khác task relevance Recall. Report ANN-vs-exact precision loss riêng.
- Latency end-to-end p50/p95, deadline miss/failure, time-to-first-useful-results, playback start; throughput input hours/wall hour, stagefps/RTF, peak RAM/VRAM, disk/network bytes.

## 2. Experiments và go/no-go

| ID | Arms/controlled variables | Data/outputs | Gate đề xuất |
|---|---|---|---|
| E01 Timeline/recovery | ffprobe/PyAV recipes; CFR/VFR/nonzeroPTS/audio offset/corruption; restart | synthetic media fixtures + small real clips; ledger/frame hashes/restart receipt | no wrong frame mapping;0 silent failures; idempotency100% fixtures |
| E02 Sampling | uniform vs scene vs diff/pHash vs hybrid; gap1/2/4s; flow on/off; query refinement on/off | human event/text spans on development videos; matched frame count **and** compute reports | save≥30% selected frames vs1fps while relevant-event coverage loss≤1pp; if fail keep denser policy |
| E03 Visual | Base224/Base384/Large256/So400m384; FP32/FP16; frame vs pooled auxiliary | fixed samples; exact retrieval | choose Pareto quality/memory/time, no unit mixing |
| E04 OCR/ASR | primary-only, routing, fallback, controlled-dual; detection resolution; VAD/align; vi/en/mixed/noise | VinText+FLEURS+joint dev; CER/WER/timing/routing capture | alternate enabled only net gain at declared compute cap; full Vietnamese charset gate |
| E05 Index | Flat/HNSW/IVF-Flat/OPQ-IVF-PQ, nprobe/efSearch/PQ;100k/1M/10M/target | exact independent query vectors; build/load/update | ANN Recall@100≥0,95 target; quality loss≤1pp taskHit; peakRAM cap; if fail increase budget or scope |
| E06 Candidates | visual-first/parallel/adaptive/hierarchical-bypass; K200/500/1k/3k; RRF vs calibrated | fixed real query set and index; candidate provenance | union≥visual recall, target temporal candidateHit@1000≥0,95; routing loss≤1pp per stratum |
| E07 Parser | original-only/decomp-only/original+graph; rules/Qwen0.6/1.7/Qwen2.5; Q4/FP16 | human structured queries, paraphrase/source-disjoint | role/temporal/negation target≥95%; no>1pp end-to-end regression; timeout fallback correct |
| E08 Relations | A vs graph rules vs track+temporal vs native grounder vs targeted VLM | real hard-negative pairs; initial smoke then confirmatory | B gain≥5pp pair accuracy CI excludes0, MomentHit regression≤1pp; C marginal gate [04] |
| E09 Localization | fixed/shot/multiscale/scorecurve/learned; boundary refine; NMS | QVH/Charades and jointdev; exact frame exports | no frame mapping errors, report IoU+MAE+recall; no merge disjoint repeated events |
| E10 Hardware/scale | warm/cold; cachehit/miss; concurrent1/2/4; background ingest off/on;100k→1M→10M | target3060/32GB; latency distributions | warm p50 ≤1 s, p95 ≤3 s; báo riêng deadline misses của Fast allocation 2 s và Accurate 8 s; peakRAM≤24GiB,VRAM≤5GiB; otherwise disable costly arms/replan |
| E11 Competition UI | click/seek/stop/replay/frame-step/keyboard/export/media404 | real media fixtures and server contracts | every displayed ID/frame matches backend; opening result triggers no inference |
| E12 Leakage | GT path deny, poison labels, gold-guided gallery flag, split hashes | pipeline imports/process permissions and audit | no qrels in index/query dependencies; predictions immutable before scoring |
| E13 Source/rights | pin releases, charset, checkpoint compatibility, dataset receipts | model/data manifests | no unapproved rights/unknown provenance activated |
| E14 Migration | full archive SHA+member hashes, selected restore; off-device copy later | local backup receipts now; independent restore pending | local hash/restore pass; deletion remains approval-gated and off-device pending |

Targets với sample ít phải kèm uncertainty; “0 failures” trên fixtures không proof production. E08≥5pp chưa hợp lý về statistical power nếu chỉ20pairs; expand trước confirmatory claim. Bootstrap2.000 resamples clustered source_video, paired differences same queries; seed recorded. Không multiple-test chọn winner rồi dùng cùng test báo p-value confirmatory.

## 3. Minimal baseline sequence

1. Build query-blind media/index from24 QVH dev assets; original-query SigLIP exact Flat, windows independent GT.
2. Freeze predictions, evaluate known-positive video/moment hits; label supplied-video localizer track separately.
3. Add OCR and ASR singly, then both; same selected visual frames to isolate marginal value.
4. Parallel union vs visual-first; track “recovered outside visualTopK” **before** reranking.
5. Add parser graph to same union; compare original fallback.
6. Add relation matcher; record oracle diagnostics separately: gold candidates can estimate ranking ceiling in evaluator-only analysis, never production or normal baseline.
7. Add optional native localizer/video verifier only after candidate reachability measured.
8. Fine-tune parser only if E07 attributable gap survives baseline/prompts and E2E gain potential shown.

Ablation axes không chạy full Cartesian explosion: one factor at a time initial, then selected interactions sampler×localizer and route×fusion. Config freeze trước test, negative artifacts/version preserved.

## 4. Measurement protocol

Hardware receipt: CPU model/threads, RTX3060 exact variant/TGP/driver, OS, CUDA/cuDNN, disk model/interface, free RAM/VRAM, thermal/power mode. Local machine tạm không đủ CUDA; không chuyển old CPU latency thành3060 forecast.

Warmup30 queries (not counted, logged), then≥200 timed unique queries or all validated set if smaller;3 randomized passes, cold tests separate process/index load. Fixed query order seed, cache states labeled; report distribution+failure/timeouts and query length/modalities/candidate counts. CUDA events stagekernel+sync; wall monotonic for full request. Do not subtract network/JSON/UI when reporting user-perceived latency.

GPU memory record process allocations/reserved and device total; batch adapt events; disk RSS+mmap resident+page faults. CPU thread caps prevent FAISS×OCR oversubscription. Ingest30min→10h stratified codec/resolution/noise pilot; measure decode/encode/OCR/ASR separately and pipeline wall time, peak storage and retries.

Scale test synthetic random vectors can validate index memory/ANN behavior but **cannot** establish semantic recall on10k h real video. Repeat corpus-size series with real distractor videos chosen independent qrels; label incomplete relevance and near-duplicates. Native full10k h acceptance requires real manifests/hours/index stats, not extrapolation.

## 5. Run artifact contract

`experiments/<run_id>/`: config.json, environment.json, model_manifest.json, data_manifest.json, predictions.jsonl, candidate_trace.jsonl, stage_timings.jsonl, metrics.json, failures.jsonl, report.md, hashes.json. RunID immutable; failed attempts retained. No overwrite successful metrics, no test-selected threshold.

Metrics JSON fields include `status=measured|not_measured`, actual n_queries/n_videos/hours, exposure, task_scope, judgment_coverage, hardware_id, cache_state, code_commit, seed, CI method. Report authoritative original labels/evaluator version. Missing metric is null with reason, never0 fabricated.

Benchmark unit tests use tiny known intervals/ranks to detect off-by-one and invalid IoU; property tests monotonicRecall@K and boundedIoU. Test no empty relevant set division; no-audio/no-text metrics reported NA. Third-party published scores ở research table tách khỏi our run artifacts.

## 6. Stop criteria

CandidateRecall insufficient→fix sampling/branches/index before expensive reranker. Relation gain absent→keep baseline and restrict capability claims. Parser latency too high→rules/cache/fallback in Fast, Accurate optional. OOM→batch/model scheduling, not silent dropping evidence. Storage insufficient→external-media plan before full corpus. Any GT leakage→invalidate affected evaluation, preserve incident artifacts, rerun with new scoped split where scientifically valid.
