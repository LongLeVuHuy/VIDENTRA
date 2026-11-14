# 03 — Research and model review

**Mốc tra cứu:04/11/2026 (Asia/Bangkok)**, bao gồm nghiên cứu đã tìm được tới ngày này, không tuyên bố bao phủ phần còn lại tháng11/2026. Nguồn ưu tiên paper gốc, repo tác giả, official model/dataset cards. API release snapshot và HF revision snapshots được lưu tại [audit](../audit/research_release_snapshot.json), [model revisions](../audit/model_revision_snapshot.json).

`SOURCE_VERIFIED` nghĩa là đọc được metadata/docs/code entry points; **không** có nghĩa đã download weights, chạy inference, benchmark3060 hoặc xác minh mọi media URL. Các README đưa benchmark của tác giả chỉ là external reported results, không chuyển thành số liệu VIDENTRA. Một số CVF HTML trả403 khi open; đã đối chiếu official indexed page/arXiv/author pages, không coi mọi paper được reproduction.

## 1. Visual, sampling và indexing

| Source/model | Đã xác minh | Quyết định / giới hạn |
|---|---|---|
| [SigLIP 2 paper](https://arxiv.org/abs/2502.14786), [official variants](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) | image-text multilingual encoder; standard/NaFlex, B/L/So400m/g weights | first-stage visual candidate, không model video temporal/action binding |
| [Base224](https://huggingface.co/google/siglip2-base-patch16-224) / [Base384](https://huggingface.co/google/siglip2-base-patch16-384) | public safetensors, Apache-2.0 card,768-d config defaults | Base224 initial; Base384 tests small object/text context vs compute; OCR vẫn native pixels |
| [Large256](https://huggingface.co/google/siglip2-large-patch16-256) / [So400m384](https://huggingface.co/google/siglip2-so400m-patch14-384) | config1024/1152-d và weights files; pinned HF SHAs | later E03 challengers, không assume6GB batch lớn; dimension change→new index |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | latest API release v0.7.2, BSD-3-Clause; Content/Adaptive detectors | scene hints, không semantic event detector; timeline authoritative từ PyAV |
| [PyAV timing](https://pyav.org/docs/stable/api/time.html), [FFprobe](https://ffmpeg.org/ffprobe.html) | PTS/time_base distinction; ffprobe installed local qua restore probe | docs stable path có version cũ; pin actual FFmpeg/PyAV at M1; test VFR, không assume library frame_num globally correct |
| [FAISS](https://github.com/facebookresearch/faiss), [selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) | latest release v1.15.1, MIT; Flat/IVF/PQ/HNSW; vector IDs integer | Flat truth for ANN recall, IVF-PQ hypothesis cho18M+ vectors; metadata cần separate store |
| [SQLite FTS5](https://sqlite.org/fts5.html) | BM25/phrase/unicode61, disk-backed text index | simple MVP; Vietnamese diacritics separate normalized/folded fields; BM25 lower is better |
| [Qdrant](https://github.com/qdrant/qdrant) | Apache-2.0 repo, payload/filter/query operations | service alternative cho operations/distribution, không bắt buộc thay FAISS |

Sampling alternatives: uniform (stable baseline), scene-only (miss intra-shot), pHash/difference (cheap redundancy but no semantics), flow (camera-motion confound/cost), learned event-aware (training/domain risk), hybrid (coverage floor + event bursts), query-time refinement (chỉ cứu candidates reachable). Hướng chọn là hybrid với budget/coverage experiments, không tuyên bố superiority trước E02.

## 2. OCR: cập nhật quan trọng năm2026

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) repo Apache-2.0; latest release v3.7.0 có PP-OCRv6. API versions khác model architecture versions; không pin “PaddleOCR3.7” rồi cho rằng mọi model checkpoint giống nhau. Baseline PP-OCRv5 mobile/Latin, challenger PP-OCRv6 small, shared detection. Document VLM như PaddleOCR-VL phục vụ document parsing, chưa có lý do chạy mọi video frame.

**Vietnamese charset gate:** [issue18254](https://github.com/PaddlePaddle/PaddleOCR/issues/18254) báo thiếu tone-marked characters trong v6 dictionaries; maintainer nói sửa cần model-level work, không đơn giản đổi dictionary. Issue closed không phải bằng chứng model đã sửa. Probe đọc current official dictionary ngày04/11/2026 vẫn không thấy một số ký tự `ả, ạ, ớ, ở, ậ, ố`; [receipt](../audit/paddle_v6_charset_probe.json). Đây là static sampled-character check, không đầy đủ charset/inference benchmark. **Không promote v6 làm recognizer tiếng Việt chính** trước kiểm charset và test diacritics. Không tự append/reorder charset vì class IDs gắn với weights.

| Recognizer | Availability/license verified | Benchmark đầu tiên và limitation |
|---|---|---|
| [VietOCR](https://github.com/pbcquoc/vietocr) | Apache-2.0 code; README pretrained crop recognizer; latest GitHub tag v0.36 khác PyPI package version | shared crops, VinText/dev real scene; autoregressive latency, không detector |
| [VieSVTR v7](https://huggingface.co/ngocthua830/viesvtr) | ONNX + safetensors listed;7,66M params card; **CC-BY-NC-4.0** | lightweight challenger khi rights phù hợp; card reports chưa reproduce; model card nêu English training overlap, không clean English benchmark |
| PP-OCRv5 Latin | general multilingual path được docs liệt kê; exact checkpoint/charset phải pin M3 | primary EN/control; Vietnamese route cần actual complete diacritics support |
| PP-OCRv6 small | official weights links/docs hiện có | general challenger, VI charset blocker ở trên |

VietOCR/VieSVTR nhận **text-line crop**, không phải detector hoàn chỉnh. Bounding boxes lấy detector chung; line-merge phải giữ polygons/source transforms. General-only vs language-route vs fallback vs selective-dual được so trên cùng detected crops và end-to-end spotting riêng, để không nhầm recognition gain với detector recall.

Khuyến nghị initial: Paddle detection + general recognition, Vietnamese specialist trên crop nghi VI/ambiguous. Khi general model không có đủ charset, không dựa high confidence để bỏ fallback Vietnamese. Thresholds model-specific, có random audit để đo missed routing. Benchmark VinText ảnh và video-text human annotations riêng; không suy từ static CER sang brief text temporal recall.

## 3. ASR và alignment

| Source | Verification | Cách dùng |
|---|---|---|
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | MIT, release1.2.1; CTranslate2, batch, VAD, word timestamps, Transformers conversion docs | small multilingual reference trên3060, FP16/int8_float16 compare; medium/large-v3-turbo secondary experiments |
| [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper), [small card](https://huggingface.co/vinai/PhoWhisper-small) | ICLR2024 Tiny Papers; public5 model sizes; small244M; BSD-3-Clause code/card | Vietnamese specialist; published WER không dự đoán retrieval, noise/code-switch cần custom strata |
| [WhisperX](https://github.com/m-bain/whisperX) | current repo BSD-2-Clause, GitHub release3.8.6; word alignment and diarization interfaces | optional language-specific alignment; không assume Vietnamese aligner ready; submodels có license/access riêng |

Whisper speech language detection, no-speech/avg-logprob/compression diagnostics không calibrated correctness. Alignment chỉ đặt thời gian lên transcript được cung cấp, không chứng minh transcript đúng. Diarization acoustic clusters không gắn identity visual. Caption/subtitle có thể paraphrase/translation. Shared timeline phải giữ audio offset và VAD chunk map.

Model conversion PhoWhisper→CTranslate2 là route được API hỗ trợ ở family level, **runtime chưa test** với chosen revision. M3 phải compare fixed utterance transcripts/timing trước primary switch. Không dùng model thừa hưởng English-only distillation cho Việt mà không benchmark.

## 4. Query parsing và inference runtime

| Candidate/source | Verification | Decision |
|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0, weights listed, thinking/non-thinking switch | first parser candidate; disable thinking, constrained compact JSON, source spans |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0, public weights | quality challenger; Q4 vs FP16 tests, larger memory/latency |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0 card | non-thinking control, Vietnamese role accuracy chưa đo |
| [Gemma3-1B-it](https://huggingface.co/google/gemma-3-1b-it) | text model, custom Gemma license/access terms | optional comparison, không assume Apache hoặc compatible terms |
| [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) | JSON Schema subset→GBNF, server APIs | compile compact schema supported subset; full Python semantic/schema validation sau decode vẫn bắt buộc |
| [PEFT](https://github.com/huggingface/peft), [TRL](https://huggingface.co/docs/trl/index) | LoRA/quantization integration và SFT APIs | M9 conditional, không fine-tune trong planning |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | MIT card,384-d, query/passage prefix | text dense retrieval, không visual relationship verifier |

Chưa chọn backend/prompt bằng benchmark. GGUF quantization phải có source revision, conversion tool SHA, recipe, checksum; không chọn random third-party quantization không provenance. Context512–2048 tokens nên đủ parsing short query; full graph JSON nhiều tokens nên compact wire+deterministic expansion. Grammar sửa syntax, không tự sửa swapped roles.

## 5. Temporal và compositional research

| Work/source | Task/verified contribution | Engineering lesson / runnable status |
|---|---|---|
| [Moment-DETR/QVHighlights](https://github.com/jayleicn/moment_detr) (NeurIPS2021) | supplied-video moment/highlight prediction; raw-video demo + checkpoint path, train/eval code | valid temporal baseline; code MIT, annotations CC-BY-NC-SA4.0; not unrestricted corpus retrieval; actual execution pending |
| [Lighthouse](https://github.com/line/lighthouse) (EMNLP2024 demo) | model wrappers, supported temporal predictors, CPU/GPU APIs, feature-specific recipes; release1.2 | practical adapter/reproduction harness; Apache-2.0 code; individual weights/features/license checks required; pretrained SigLIP substitution invalid |
| [Compositional Temporal Grounding](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Compositional_Temporal_Grounding_With_Structured_Variational_Cross-Graph_Correspondence_Learning_CVPR_2022_paper.html), [arXiv](https://arxiv.org/abs/2203.13049), [splits repo](https://github.com/YYJMJC/Compositional-Temporal-Grounding) | CVPR2022, structured cross-graph correspondence; Charades-CG/ActivityNet-CG novel-composition/novel-word splits | use compositional splits/diagnostics; repo README confirms split release, not a verified turnkey graph model+weights |
| [DeCo](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_DeCo_Decomposition_and_Reconstruction_for_Compositional_Temporal_Grounding_via_Coarse-To-Fine_CVPR_2023_paper.html), [author page](https://www.ut-vision.org/publication/2023-yang-deco/) | CVPR2023 compositional temporal grounding, decomposition/reconstruction and coarse-to-fine contrastive ranking | motivate full-query+components ablation; executable code/weights/license not established this audit, not implementation dependency |
| [Multi-Modal Relational Graph](https://openaccess.thecvf.com/content/CVPR2021/html/Zeng_Multi-Modal_Relational_Graph_for_Cross-Modal_Video_Moment_Retrieval_CVPR_2021_paper.html), [author site](https://cvpr-2021.wixsite.com/mmrg) | CVPR2021 object visual/text relational graphs, matching+boundary regression | graph representation useful, training/perception expensive; author site says code reorganizing/contact, no verified ready weights |
| [ModaRoute](https://arxiv.org/abs/2507.13374), [ICCV2025 workshop](https://openaccess.thecvf.com/content/ICCV2025W/MRR%202025/html/Dela_Rosa_Smart_Routing_for_Multimodal_Video_Retrieval_When_to_Search_What_ICCVW_2025_paper.html) | multimodal routing across visual/OCR/ASR, paper uses GPT-4.1 | motivation for adaptive budgets; **not** evidence0.6B reproduces routing quality or actor binding; paper performance not our forecast |
| [HiMu](https://github.com/DanBenAmi/HiMu), [paper](https://arxiv.org/abs/2603.18558) | hierarchical logic frame selection for **long VideoQA**; README says accepted WACV2027, MIT, query/expert examples | fuzzy tree/temporal composition informative; not corpus retrieval benchmark. Do not copy README fps-based index conversion into VFR contract; offline2fps experts costly at10k h |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Apache-2.0 card, public multimodal weights/APIs | optional candidate-level verifier; no proven reliability for handover/speaker or6GB latency; quantization/context tests needed |

Graph matching paper success trên supplied video không chứng minh identify correct corpus video. Scene graph/tracking chỉ cung cấp representational primitives; không semantic truth. Specialized action models cần correct role labels; bounding boxes không đủ direction of transfer. Stage3 phải đánh giá counterfactual swaps và time reversal thay vì chỉ hỏi leading yes/no.

## 6. Tài liệu mới2026 và ảnh hưởng tới thiết kế

- [ExtremeWhenBench — Natural-Language Temporal Grounding in Hour-Long Videos is a Search Problem](https://aclanthology.org/2026.emnlp-main.2112/), EMNLP2026 October. Nguồn mô tả search bottleneck trong hour-long supplied-video grounding. **Suy luận engineering:** candidate coverage nên đo trước investing localizer; không mang mức gain paper sang corpus10k h. Dataset/raw rights/runnable pipeline chưa audit để core adoption.
- [Driving Video Retrieval for Complex Queries with Structured Grounding](https://arxiv.org/abs/2606.09109v3), revised23/10/2026; arXiv comments báo accepted NeurIPS2026, chưa kiểm proceedings. Domain driving và data-calibrated rules. **Suy luận:** rule reliability phải domain-calibrate; không chuyển event rules lái xe sang người trao túi.
- [Towards One-to-Many Temporal Grounding](https://proceedings.mlr.press/v306/xu26cc.html), ICML2026 official proceedings. Nghiên cứu nhiều disjoint moments. **Suy luận:** giữ repeated events và evaluate multiple-instance recall, không NMS toàn video thành một result.
- PaddleOCRv6 và VieSVTRv7 được đưa vào comparison vì public releases hiện hữu; lỗi charset và contamination note quan trọng hơn marketing “newer/better”.

Các papers mới là direction review, không bắt buộc tải thêm datasets/models. Không tuyên bố survey exhaustive hoặc tất cả source claim đã reproduction.

## 7. Reproducibility/availability gate

Mỗi model trước activation phải có: repository+license riêng code/weights/data; exact commit/revision; SHA-256 bytes; processor/charset/tokenizer; dependency lock/OS/CUDA; tiny inference output; peak memory; input/output tensor shape; known failure; accepted use. “Weights files listed on Hub” khác “download/load thành công”.

Model benchmark order cho3060: SigLIP Base224→Base384; Paddle detector+PP5 general vs VietOCR→VieSVTR rights-gated→PP6 charset-gated; Whisper-small vs PhoWhisper-small→medium/turbo nếu gain; Qwen3-0.6B→1.7B/Qwen2.5 control. Action/video reranker sau independent union và human hard-negative set. Không chạy model benchmark nào trong planning.

Nguồn request ban đầu đều được xem xét ở bảng trên; unavailable code/weights được ghi chưa xác minh thay vì giả định runnable. Khi source cập nhật sau04/11, tạo registry revision mới, không sửa ngược receipt lịch sử.

