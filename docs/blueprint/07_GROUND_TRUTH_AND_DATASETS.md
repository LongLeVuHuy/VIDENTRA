# 07 — Ground Truth, datasets và leakage prevention

## 1. Stack tối thiểu và audit cũ

Bắt đầu **24 video QVHighlights local + annotation tương ứng**, chỉ development đã tiếp xúc, để kiểm timeline→query→rank→playback. Không đợi10k giờ. Thêm Charades-CG cho compositional grounding, TextVR cho scene-text retrieval, VinText cho Vietnamese OCR và FLEURS vi_vn cho ASR theo capability gap, không download mọi dataset cùng lúc.

Đã đọc old `docs/rebuild_2026/10_DATASET_GROUND_TRUTH_STRATEGY.md`, DATASET_REGISTRY, journal và raw/manifests. Giữ insight về exposure/rights nhưng viết blueprint mới. Các sự kiện local cần bảo toàn:
- QVH24 video và validation JSONL1550 rows; chỉ24 queries match media local theo audit lịch sử. Supplemental restore đọc lại1550JSON rows, không chạy retrieval mới.
- AVC300-video gallery cũ chọn160 known targets+140 distractors; cache229 từ Top15 của36 queries theo audit cũ. **Gold-informed**, không fresh corpus evaluation.
- Old review packets/AI-reviewed query graphs không tự thành human GT; nhiều form pending. Old VSR self-review/official-dev repetitions không là independent labels.
- Test samples VinText/VIVOS và FLEURS **English** từng dùng trong decisions; không gọi vi_vn đã benchmark chỉ vì có FLEURS report.
- Source paths/manifest mappings cũ giữ byte-identical trong [backup](01_OLD_PROJECT_RETIREMENT.md); new derivative manifest ghi parent hashes/exposure.
- Old dataset registry từng ghi TextVR42,2k videos; nguồn tác giả ghi **42,2k sentence queries cho10,5k videos**. Dùng số tác giả, không copy lỗi. [TextVR](https://github.com/callsys/TextVR)

## 2. Dataset matrix

`SOURCE_VERIFIED` chỉ xác nhận tài liệu/annotation release metadata; không raw download/decode mới.

| Dataset/source | Task và annotation | Splits/evaluator | Rights/media/readiness | Vai trò |
|---|---|---|---|---|
| [QVHighlights](https://github.com/jayleicn/moment_detr/tree/main/data) | human query, relevant_windows, saliency trên2s clips | train/val/test; test GT withheld; official standalone_eval | annotations CC-BY-NC-SA4.0; raw archive URL official được repo liệt kê, media rights riêng;24local exposed | core temporal development; official single-video score riêng |
| [Charades-STA/TALL](https://github.com/jiyanggao/TALL) | sentence+[start,end] cho household video | train/test release; repo ghi updated cleaned annotations, phải pin đúng version | base Charades media/terms phải check trước acquisition; annotation links verified, chưa tải | action grounding challenger |
| [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) | temporally localized event descriptions | train/val/test; [official dense-caption evaluator](https://github.com/ranjaykrishna/densevid_eval) đánh caption khác retrieval | URLs availability/rights và annotation release revision cần xác nhận thêm; không assume YouTube còn đủ | later scale/action coverage, không initial download |
| [Ego4D NLQ](https://ego4d-data.org/docs/benchmarks/episodic-memory/) | language query + answer-visible interval trên egocentric video | official train/val/test protocol và [episodic-memory](https://github.com/EGO4D/episodic-memory) evaluator | access agreement; official site hiện thông báo chuyển access sang Lab22; chưa media-ready | long egocentric localization, không OCR/ASR complete truth |
| [Charades-CG / ActivityNet-CG](https://github.com/YYJMJC/Compositional-Temporal-Grounding) | temporal labels, novel-composition/novel-word split definitions | train/test-trivial/novel-composition/novel-word, theo author release | split files verified docs; base-media rights inherit riêng; code/weights model không presumed | compositional generalization; không exhaustive actor-role reversal labels |
| [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning) | image-caption spatial binary labels | random/zero-shot official splits; preserve released rows và pin counts | Apache-2.0 repo; COCO image rights riêng; old dev exposure | component2D spatial only, không video moment retrieval |
| [TVR](https://github.com/jayleicn/TVRetrieval) | human video-subtitle moment queries, modality attribution | video retrieval/SVMR/VCMR riêng; official evaluator | TV episode rights/access hạn chế; code license không cấp raw media; native subtitles không ASR verbatim GT | optional multimodal corpus-moment evaluation |
| [TextVR](https://github.com/callsys/TextVR) | video/query relevance, scene-text reading; supplied OCR from API | author train/test JSON; actual unscrambled test annotation now linked | resized5,4GB/full85GB links listed, chưa tải; dataset license/media redistribution chưa fully verified | OCR contribution to video retrieval; không precise moment/OCR CER GT |
| [VinText](https://github.com/VinAIResearch/dict-guided) | Vietnamese static scene-text polygons+transcripts | official train/test and conversion; recognition vs detection protocols tách | research/education only, no redistribution theo README; acquire controlled path, not republish | diacritics/detection/crop CER; không temporal text tracking |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | speech audio + raw/normalized transcripts | vi_vn và en_us train/validation/test; corpus config pin | CC-BY4.0; official parquet/source access; no new audio downloaded | ASR CER/WER, không speaker-to-visual identity |
| [VIVOS](https://zenodo.org/records/7068130) | Vietnamese read speech, transcripts | train11660/test760 utterances theo source | CC-BY-NC-SA4.0, academic; official1,5GB archive; old subset exposure | optional ASR domain control, không noisy spontaneous video proof |
| [AVC](https://www.avc.gg/) | aligned machine-generated multimodal layers và retrieval/QA benchmark với human review scope theo release | pin v1 official query/corpus IDs/evaluator; không mix website v2 | corpus media rất lớn, không full download; local300 available but gold-informed | historical diagnostics; audit per-item human provenance trước use |

ActivityNet official page initial request gặp502; evaluator repo verified, nhưng raw data/rights status vẫn pending. Không bịa media download receipts hoặc license gộp cho mọi asset. Link khả dụng không chứng minh bytes/decode/index readiness.

### AVC conflicting provenance

Current website mô tả4237 human-reviewed benchmark items; old pinned audit đọc200 audit items (100 retrieval/100QA), với7 retrieval items overlap local300. Hai thông tin có thể là khác release/scope. **Không nâng toàn v1 labels thành human-reviewed dựa current landing page.** Resolve release hash/date/per-item receipts trong E13; giữ old audit và current claim tách. “Same video±10s=.5” trong old evaluator là protocol-derived relevance, không human relevance judgment.

## 3. Capability coverage

| Capability | Best initial evidence | Gap |
|---|---|---|
| accurate moment temporal IoU | QVH/Charades-STA | corpus discovery và Vietnamese relevance không được đo tự động |
| novel predicate compositions | Charades-CG | explicit swapped giver/receiver/speaker cần custom contrast labels |
| exact Vietnamese scene text | VinText | video persistence/brief text cần video annotations |
| Vietnamese speech | FLEURS vi_vn/VIVOS | code-switch/noise/actor alignment cần joint video set |
| OCR-assisted retrieval | TextVR | exact frame bounds/transcription human status khác nhau |
| speech+visual moments | TVR/AVC diagnostics | subtitles/machine layers không verbatim truth |
| negation/absence | bounded human-reviewed cases | sparse samples không xác minh universal absence |
| cross-modal binding | joint human video review | chưa có ready local benchmark đầy đủ |

Không dataset đơn lẻ chứng minh toàn query “đưa túi→người nhận cảm ơn→SALE phía sau”. MSR-VTT/SugarCrepe/VSI-Bench cũ giữ archive; video-level captions, image contrast và VideoQA là diagnostics khác task.

## 4. Joint development set và human protocol

Initial target:40–60 legally reusable clips, tổng1–3h, khoảng200 queries Việt/Anh; numbers là kế hoạch, **hiện chưa tạo nhãn mới**. Chọn corpus độc lập target labels; đủ motion/static/text/speech/no-audio/no-text/noise/code-switch. Có ít nhất20 pairs cho mỗi stratum hard-negative để smoke; confirmatory≥100 independent pairs/type hoặc power analysis trước claims.

Annotation two-pass:
1. Annotators độc lập xem raw video+audio, ghi event intervals, actor/object identities, role bindings, OCR polygon/text observations và speech transcription/times. Không xem model ranking.
2. Query author tạo query từ observation hợp lệ; translator Việt/Anh được người thứ2 review semantic equivalence, giữ quoted literals gốc.
3. Negative construction: opposite roles, attribute swap, reverse order, different speaker, object co-occurrence without interaction, text at wrong time, near-duplicate scenes, bounded negation. Dùng video thật khác hoặc alternate query cho same clip; edited videos/synthetic examples gắn riêng, không claim natural-data performance.
4. Hai reviewers đánh relevance trên fixed corpus/pool: relevant/irrelevant/partial/unknown + reason + evidence refs. Adjudicator xử lý disagreements; lưu originals và adjudication, không overwrite.
5. Measure inter-annotator agreement (categorical kappa/Krippendorff alpha theo task), boundary disagreement median/p95, coverage; acceptance target≥0,8 category agreement statistic chỉ mục tiêu, inspect nếu không đạt.

Human annotation ở inference không được dùng; inference evidence là model/rule outputs. Truth và query-structure labels độc lập: graph correct không đảm bảo video match. “Behind” 3D chỉ annotate supported khi video đủ cues; unknown cases dùng để test abstention, không ép binary.

## 5. Split và exposure governance

Split trước tuning theo **source_video_id**, near-duplicate cluster, clip family và query paraphrase family; không chia random frames cùng video sang train/test. Initial custom allocation60/20/20 train/dev/test, calibration tách từ train nếu đủ; nhỏ quá thì dùng cross-validation grouped, giữ untouched final test riêng. Tất cả labels/doc inspections/existing benchmark exposure ghi dataset ledger.

Official splits giữ nguyên; local subset manifest ghi selection reason, seed, source revision, missing media IDs, download/rights/decode states. Không thay official label vì model disagree; ghi suspected issue sidecar và report sensitivity khi có authoritative correction.

Calibration và threshold tuning chỉ train/dev; test không dùng chọn parser/OCR thresholds/index nprobe/sampler. LLM-generated labels là pseudo, không tier human_verified dù được machine “review”. Annotation source official/human status/review status/exposure là các axes độc lập.

## 6. Evaluation isolation và corpus relevance

Hai tracks riêng:
- **Official single-video localization:** input explicitly supplied video theo official task, evaluate localizer. Gold video có thể là task input ở track này, nhưng không gọi là corpus retrieval.
- **Unrestricted/declared-corpus search:** query không nhận target ID hoặc gold interval; index corpus chosen independently, all branches global. Qrels chỉ evaluator process đọc sau predictions frozen.

QVH original annotations không exhaustively label every cross-video negative. Với24-video development corpus báo **known-positive Hit@K** và temporal hit, không claim exhaustive corpus mAP/nDCG. Unjudged≠negative. Nếu muốn corpus metrics, tạo frozen judged evaluation universe bằng pooling candidates từ nhiều baselines + random negatives + human review; báo judgment coverage và hạn chế pool, không coi unknown ngoài pool bằng0 âm thầm.

Prediction artifact ghi query_id, candidates, scores, moments, snapshot/config hashes; evaluator mount labels read-only ở path runtime bị deny. Independent evaluation code tái tính official metrics và synthetic unit fixtures. Gold-informed candidate galleries có task label “oracle/diagnostic”, không đi vào generalization table.

## 7. Acquisition gates và ưu tiên

M0: preserve existing assets; QVH existing24 dev, no new bulk download. M3: small official VinText/FLEURS samples sau source+rights check. M6: Charades-CG source split and joint annotations for real relation claims. TVR/Ego4D/ActivityNet only if specific coverage gap and resource access established.

Per dataset receipt: source URL/revision/license snapshot, file hash/bytes, official IDs/splits, expected count vs actual, media availability/decode gaps, annotation_type, human provenance, exposure, evaluator revision. Đạt annotation-ready không tự thành media-ready; partial downloads phải có missing manifest. Không gọi all10k h benchmark khi chỉ shard/distractor-expanded pilot.
