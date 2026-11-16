# VIDENTRA — Audit sau bản nâng cấp của Hydrogen (HEAD `4ba4ebc`) & Hướng phát triển tiếp theo

> Ngày: 16/11/2026. Đã đọc: `docs/PROGRESS.md`, `benchmark/benchmark_report.md`, `benchmark/msrvtt_clip_qwen_videoprism_audit.md`, `models/ranking/videntra_ltr_v1.json`, và code `src/videntra/retrieval/{unified_pipeline,clip_branch}.py`.
> Phạm vi chưa phủ: `query/llm.py` (~2 000 dòng mới), `storage/database.py`, `api/ui_html.py`, scripts Kaggle — chỉ đọc qua PROGRESS. Mình **không chạy lại benchmark**; các số lấy từ report/PROGRESS. Kết quả pytest chạy trong lượt này được ghi ở mục A.

---

## A. Tình trạng hiện tại

**Đã sửa tốt so với audit trước (B1–B11):** benchmark nay có provenance rõ (run id, SHA LTR, `Eval Mode: oof`, `Overlap With Eval`, kênh bị degrade); không còn đánh giá in-sample một cách âm thầm; `heldout_test_201` `clean_disjoint`; có split `tune/confirm` khóa theo `source_group` + SHA-256; replay/capture có kiểm hash labels; VietOCR/PhoWhisper được xử lý lại (theo PROGRESS, `tests/ocr/test_ocr_asr_no_stubs.py`). Phương pháp đánh giá đã tiến bộ rõ rệt — đây là điểm mạnh lớn nhất của bản này.

**Số hiện tại (A5, report run `0e7ec648`):**

| Tập | Hit@1 | MRR | nDCG@10 | Ghi chú |
|---|---|---|---|---|
| `dev_150` | 62.00% | 0.6548 | 0.6752 | 129/150 query dùng trọng số OOF, 21 query dùng trọng số fit-all |
| `dev_full_859` | 43.77% | 0.5142 | 0.5558 | MSR-VTT 40.91% (331/809) |
| `heldout_test_201` | 39.80% | 0.4805 | 0.5291 | MSR-VTT 37.70% (72/191) — **bằng đúng Phase 6** |

Latency Non-Neural p50/p95 = 53.71/124.35 ms (mục tiêu 20/45: FAIL); end-to-end 319.9/652.4 ms trên CPU 4 nhân.

**Pytest:** xem cuối file (mục F).

---

## B. Lỗi / rủi ro còn lại

### B1 (HIGH) — LTR gần như là no-op, và đã lỗi thời so với pipeline
`videntra_ltr_v1.json` có 24 trọng số, tất cả ở cỡ `1e-4 … 1e-3` (lớn nhất `vis_lexical_synergy = 0.001067`). Tổng `|w|` ≈ 0.0074. Với đặc trưng đã chuẩn hóa (|z| ≲ 3) thì `|ltr_sc| ≲ 0.022`, và đóng góp vào điểm là `0.015 × tanh(ltr_sc)` ≲ **3.3e-4** (ước lượng cận trên của mình từ trọng số, chưa đo trên dữ liệu thật). Trong khi các cổng sau đó dùng độ lệch điểm cỡ **0.03–0.045** (`a5_gap`). Nghĩa là LTR hầu như không thể đổi thứ hạng. Nguyên nhân: loss phần dư `(base_i − base_j) + 0.015·(Δx@w)` cộng `l2_reg = 250000` ép `w` về ~0 — bài toán tối ưu "học" được rằng không nên can thiệp.
Thêm nữa, `trained_at = 2026-11-15T07:59Z`, **trước** các thay đổi 16/11 (lọc COCO-80, loại sub-clause trùng nghĩa, trộn 768d+864d ở Stage 2) → phân phối đặc trưng đã đổi nhưng LTR chưa train lại.
**Cách xử lý:** (1) chạy ablation `--ltr-blend-weight 0` vs hiện tại trên `dev_full_859` OOF; nếu Δ≈0 thì ghi nhận LTR không đóng góp (và không gọi là "bứt phá"); (2) nếu muốn LTR có tác dụng, bỏ cách "cộng phần dư nhỏ": train LTR như bộ chấm điểm chính trên Top-K ứng viên (blend 1.0, thang điểm chuẩn hóa), hoặc LambdaMART; (3) train lại sau mỗi thay đổi pipeline và ghi `pipeline_git_sha` vào metadata LTR; provenance gate nên cảnh báo khi `ltr.trained_at < pipeline_last_change`.

### B2 (HIGH) — Consensus Boost là luật ngưỡng cứng, chưa chứng minh tổng quát hóa
`_apply_clip_consensus_boost` (`unified_pipeline.py:1767-1846`) lật Top-1 bằng `a5_gap + 0.002` khi `rf_b==1 && rc_b==1 && f_diff≥0.008 && 0<a5_gap≤0.045` hoặc `rf_b≤2 && rc_b==1 && f_diff≥0.005 && c_diff≥0.015 && a5_gap≤0.030`. Các hằng số này có dạng đã chỉnh theo dev. Bằng chứng:
- Tăng trên dev: +2.35 pp (21W/2L), nhưng bộ `tune/confirm` là hai nửa của **cùng** `dev_full_859`; cổng được thiết kế sau khi đã xem kết quả dev, nên `confirm` không còn là kiểm tra độc lập thật sự.
- Trên `heldout_test_201` chỉ tăng 38.81% → 39.80% = **+2 query/201**, CI 95% bao phủ 0; riêng MSR-VTT held-out vẫn 72/191 = 37.70% như Phase 6.
- "Clip" là trung bình cửa sổ của **cùng vector SigLIP2 768d** mà A5 đã dùng → không có thông tin độc lập; đây là tái xếp hạng bằng cùng tín hiệu, rủi ro overfit cao. Nguồn tín hiệu độc lập (VideoPrism) vẫn **chưa có artifact** (`CANCEL_ACKNOWLEDGED`).
**Cách xử lý:** coi boost là giả thuyết, chưa phải cải tiến. (1) Báo cáo bootstrap theo `source_group` cho heldout; (2) thay luật tay bằng bộ gate học được (logistic nhỏ trên `f_diff, c_diff, a5_gap, rank_*`) huấn luyện trên `tune` và chỉ đánh giá trên `confirm` + heldout **một lần**; (3) bật mặc định chỉ khi lợi ích heldout có CI > 0; nếu không, để `enable_consensus_boost=False` và ghi rõ là thử nghiệm.

### B3 (MEDIUM) — Benchmark không tái lập từ một commit
`benchmark_report.md`: `Git SHA 02803a2 (Dirty: True)`; HEAD hiện là `4ba4ebc`; cây làm việc hiện còn thay đổi chưa commit (`AGENTS.md` sửa, 8 file `.md` bị xóa). Số liệu trong report không gắn với một commit sạch.
**Cách xử lý:** `run_validation_benchmark.py` từ chối ghi đè `benchmark/` khi `dirty=True` (hoặc ghi vào `work/experiments/` kèm cờ `unofficial`); chạy lại bản chính thức trên cây sạch.

### B4 (MEDIUM) — Ba kênh quan trọng vẫn `missing_artifact`
Report: `e5_caption_index`, `vlm_caption_lexical`, `spatial_patches` đều thiếu (ảnh hưởng 1210/1210 query), `temporal_rerank` tắt. Nghĩa là các nâng cấp Phase 8 P8-5 (caption Florence-2, late-interaction) **chưa đóng góp gì vào số liệu**; code đã viết nhưng chưa được kiểm chứng bằng dữ liệu. `rrf_vis_caption` có trọng số LTR = 0, `s_late_interaction` = 0.
**Cách xử lý:** chạy Kaggle P8-5 (đã có pipeline), rồi ablation từng kênh. Cho đến khi có artifact, đừng liệt kê chúng như thành phần "đã đo".

### B5 (MEDIUM) — Chi phí mỗi truy vấn của `ClipBranchIndex`
Mỗi lần search, `_ensure_clip_branch_index()` gọi `matches()` → `compute_snapshot_signature()` duyệt toàn bộ `video_kf_fids` (1 324 video, 11 581 id) và băm lại chuỗi. (Suy luận từ code; chưa profile.) Trong khi p50 non-neural đã tăng từ 24 ms (Phase 6) lên 54 ms. Ngoài ra chữ ký chỉ băm 3 hàng embedding (đầu/giữa/cuối), nên không phát hiện được sửa đổi ở hàng khác.
**Cách xử lý:** tính chữ ký một lần khi nạp snapshot và cache theo `id(emb)`/mtime; dùng hash toàn mảng (hoặc `fid` + SHA của file) lúc build index; profile bằng `cProfile` để xác nhận đóng góp trước khi tối ưu.

### B6 (MEDIUM) — Cache phân rã Qwen: hầu hết query vẫn đi rule/fallback
Capture thực tế: `cache_hits=205` (base) / `821` (LoRA), `fallback=617`/`1`, `rule_tier=37` trên 859 query. Hai điều đáng lưu ý: (a) base `extractive-v2` hòa hoàn toàn với rules (0W/0L) → LLM chưa mang lại gì cho Hit@1; (b) LoRA có 821/822 plan hợp lệ nhưng `tune` −0.18 pp và Moment@0.5 giảm 16/22 → 15/22. LoRA train trên `tune`, đánh giá trên cùng `tune` ra −1 query nên chưa có bằng chứng overfit/cải thiện; kết luận "giữ rules" của Hydrogen là hợp lý.
**Cách xử lý:** thôi đầu tư thêm vào phân rã bằng LLM cho MSR-VTT (caption ngắn, 1 mệnh đề); chỉ giữ cho truy vấn dài/đa sự kiện (QVH/AVC/TVR), và đánh giá riêng nhóm đó (n nhỏ — cần mở rộng tập đa sự kiện trước).

### B7 (LOW) — Mục tiêu nghiệm thu vẫn FAIL, và vài số mới chỉ là ước lượng nhỏ mẫu
`Moment nDCG@5 = 0.5121 < 0.53`, latency FAIL, `Pure-ASR FP = 0/2` (n=2 pure-ASR nên chưa đủ để kết luận "FP = 0 tuyệt đối"), Moment đo trên 22 query (CI [54.6%, 90.9%]).
**Cách xử lý:** ghi rõ n trong README/benchmark (đã ghi `0/2` ở report — giữ cách này); thêm truy vấn QVH/TVR có nhãn moment để n ≥ 100 trước khi so sánh các biến thể định vị.

### B8 (LOW) — Tài liệu & repo
Cây làm việc chứa thay đổi chưa commit: `AGENTS.md` (thêm hai quy tắc về README/benchmark) và 8 file `.md` bị xóa (bao gồm các báo cáo audit/roadmap trước đó). Nếu việc xóa là có chủ ý thì nên commit; nếu không, khôi phục bằng `git restore`. Cần đối chiếu với quy tắc "Preserve ... historical runs" — các báo cáo audit là tài liệu lịch sử.

---

## C. Điểm mới thực sự (đã có đo đạc)

1. **Giao thức đánh giá chống rò rỉ cấp hệ thống**: OOF LTR theo `source_group`, split khóa SHA-256, `labels_sha256` kiểm tra, provenance ghi kênh bị degrade. Đây là điểm khác biệt rõ so với đa số prototype retrieval.
2. **Phân tích trung thực**: nhiều thí nghiệm kết thúc bằng "không cải thiện, giữ rules" (Qwen, LoRA, clip reserve với `alpha=0`) — đúng tinh thần khoa học.
3. **Cải thiện thật nhưng nhỏ trên MSR-VTT**: lọc COCO-80 và loại sub-clause trùng nghĩa (sửa lỗi phạt oan `−0.10`, tránh nhiễu `sub_clauses`) → +2.35 pp trên dev; trên heldout MSR-VTT chưa thấy thay đổi (37.70%), nên chưa xác nhận tổng quát hóa.
4. Hạ tầng Kaggle offload có receipt/hash (revision, weights SHA, cache SHA) — tái lập tốt.

Chưa nên gọi là bứt phá: Consensus Boost (B2), LTR 24d (B1), Qwen LoRA (B6), các kênh P8-5 (B4).

---

## D. Hướng phát triển tiếp theo (xếp theo giá trị kỳ vọng / rủi ro)

**Chẩn đoán gốc rễ trước.** Trên `dev_full_859`, MSR-VTT: Recall@20 = 611/809 (75.5%) nhưng Hit@1 = 40.9%. Vậy 25% video không vào pool, và 35 điểm còn lại thất thoát ở khâu xếp hạng giữa các ứng viên gần nhau. Mọi tinh chỉnh bằng luật trên cùng một vector SigLIP2 đều đụng trần (Hit@1 heldout MSR-VTT đứng yên 37.7% qua ba phase). Cần tín hiệu **mới và độc lập**, không thêm luật.

**D1. Tín hiệu video thực sự (ưu tiên số 1).**
Hoàn tất VideoPrism-LvT-B (hoặc InternVideo2/ViCLIP) trên Kaggle cho 1 324 video; dùng làm điểm độc lập trong rerank Top-K. Giả thuyết kiểm chứng: Hit@1 MSR-VTT held-out ≥ +3 pp, CI > 0 theo `source_group`. Cần: shard thật, replay với `alpha ∈ {0.25,0.5,0.75}` chọn trên `tune`, khóa, rồi báo `confirm` + heldout một lần. (Tài liệu của Hydrogen đã chỉ ra reserve với `alpha ≤ 0.5` không thể lật hạng 1 — nên rerank trực tiếp Top-5 thay vì "reserve ở đuôi".)

**D2. Cross-encoder / VLM làm reranker Top-5 (offline trên Kaggle, chấm sẵn cache).**
Dùng VLM nhỏ (Qwen2-VL-2B / SmolVLM / LLaVA-OneVision-0.5B) chấm "caption khớp video?" trên 4–8 khung mỗi ứng viên Top-5, lưu cache theo `(query_id, video_id)`. Đây là cách duy nhất nhắm trực tiếp vào lỗi `F4_FINE_GRAINED` (34–35% lỗi). Đánh giá: so Top-1 trước/sau trên cùng pool, paired bootstrap. Chi phí: ~1 060 × 5 × 8 khung — khả thi trên 2×T4.

**D3. Mở rộng recall (25% video ngoài pool).**
(i) Thêm kênh truy hồi từ caption Florence-2 (B4: chạy P8-5 thật); (ii) pseudo-relevance feedback (Rocchio) trong không gian 864d: lấy trung bình Top-3 làm truy vấn mở rộng, kiểm tra Recall@20; (iii) truy vấn đa prompt (3–5 diễn đạt lại bằng luật/LLM) rồi hợp nhất RRF. Chỉ số: Recall@20 từ 75.5% → ≥ 80% trước khi nghĩ tới rerank.

**D4. Fine-tune nhẹ biểu diễn (LoRA/projection) với split theo `source_group`.**
Huấn luyện projection head 768→768 hoặc LoRA trên text tower SigLIP2 bằng contrastive loss trên `tune` (MSR-VTT có 20 caption/video — hiện chỉ dùng 1 caption/query?). Đánh giá `confirm` + heldout. Đây thường là bước tăng lớn nhất cho zero-shot retrieval trong miền hẹp; rủi ro: phải giữ nguyên quy tắc không dùng GT để chọn candidate.

**D5. Sửa LTR cho đúng vai trò (liên quan B1).**
Train LambdaMART trên Top-20 pool (`capture_msr_candidate_pools.py` đã có), đặc trưng gồm điểm nhiều nguồn (SigLIP2, VideoPrism, VLM-score, lexical), blend 1.0, OOF theo `source_group`; thay thế cổng ngưỡng cứng của Consensus Boost bằng đầu ra của mô hình này. Tiêu chí chấp nhận: heldout CI > 0.

**D6. Định vị khoảnh khắc.**
Mở rộng nhãn moment (hiện 22 query) bằng QVHighlights đầy đủ (hàng nghìn query); thử DP đơn điệu + NMS đa đỉnh với ngưỡng học được; mục tiêu `Moment nDCG@5 ≥ 0.53` trên n ≥ 100.

**D7. Hiệu năng.**
Profile (B5), cache chữ ký snapshot, vector hóa Stage 4–6 (hiện vòng lặp Python theo video), đo latency end-to-end trên RTX 3060 6GB riêng cho GPU path; mục tiêu non-neural p50 ≤ 30 ms.

**D8. Quy trình.**
Chỉ chạy heldout sau khi khóa cấu hình; benchmark từ chối ghi khi cây `dirty`; mỗi cải tiến phải đi kèm bảng `tune → confirm → heldout` với CI theo nhóm nguồn, và ghi rõ n.

### Thứ tự đề xuất
`B1/B3/B8` (nửa ngày) → `D1` (VideoPrism, cần Kaggle) song song `D3` (P8-5) → `D2` → `D5` → `D4`. Dừng đầu tư thêm vào phân rã Qwen cho MSR-VTT (B6).

---

## F. Kết quả pytest trong lượt audit
`pytest -q -q -x -p no:cacheprovider` trên HEAD (cây làm việc hiện tại) kết thúc với exit code 0, tức không có test lỗi (do dùng `-q -q` nên không in số lượng; PROGRESS ghi 588 passed). Chưa chạy lại benchmark.

