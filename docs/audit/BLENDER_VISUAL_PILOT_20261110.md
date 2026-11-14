# Blender visual-only pilot — 10/11/2026

## Phạm vi và nguồn

Đây là phép thử visual retrieval có tính thăm dò, không phải A5, relation verifier hay đánh giá human-gold. Corpus được cố định trước khi đọc nhãn: Big Buck Bunny (dev) và Elephants Dream (test), hai phim Blender có receipt và attribution tại `data/raw/blender_pilot_20261110`. Hai phim nổi tiếng và số source quá ít; exposure trước huấn luyện và near-duplicate chưa được loại trừ.

Query lấy từ [MomentBench 0.2.5](https://github.com/mixpeek/momentbench), file film scored đã pin commit và kiểm SHA-256. Nguồn công bố nhãn do model tạo và model adjudicate, **chưa human review**. Dự đoán không được dùng làm nhãn người. Corpus/query chỉ tiếng Anh.

## Quy trình inference

- Hai ledger video được decode tuần tự hai lượt PyAV: 14.315 và 15.294 frame, đối chiếu PTS/ordinal không sai khác. Seek bit-exact, audio và source-relative chưa được xác minh.
- Dataset Kaggle private chứa hai phim có attribution và 130 cặp query ID/text; không chứa qrels, timestamps nhãn hoặc hard negatives.
- Kernel private version 2 trên NVIDIA Tesla T4: SigLIP 2 `google/siglip2-base-patch16-224` revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weights SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`. PyAV 19.0.2 wheel kiểm hash; torch 2.11.0+cu130. Lượt 1 thất bại trước inference do thiếu PyAV; giữ log. Kernel 2 hoàn tất.
- Lấy frame tuần tự theo PTS cách tối thiểu 10 giây, không dựa query/qrels: 60 frame Big Buck Bunny, 66 frame Elephants Dream. Mọi 126 frame đã đối chiếu PTS/timestamp với ledger local và media hash. Tính cosine visual của ensemble query raw/frame prompt (0.75/0.25); chọn frame điểm cao nhất mỗi video, window cố định ±5 giây. Không OCR, ASR, verifier hay boundary expansion.
- Chạy đủ 130 query trước khi evaluator đọc qrels. Predictions JSONL SHA-256 `fed260afddb62b2a2bc09e9e40691875a9bdd4016b7248b0a9253e50ffaefda6`. Evaluator kiểm input hashes và phân nhóm sau inference.

## Kết quả provisional theo nhãn model

| Nhóm | Số query có positive trong corpus | Video Hit@1 | Moment Hit@1 tIoU >= 0.5 |
| --- | ---: | ---: | ---: |
| Dev — Big Buck Bunny | 12 | 12/12 | 4/12 |
| Test — Elephants Dream | 23 | 23/23 | 1/23 |
| Cả hai nhóm | 35 | 35/35 | 5/35 |

35/105 query có positive công bố thuộc corpus (33,33%); 70 positive chỉ ở ngoài corpus và 25 không có positive công bố được tách riêng, không coi là negative. Kết quả video Hit@1 trên hai phim rất khác nhau và chỉ nhóm positive trong corpus không đại diện khả năng tìm trên corpus lớn. Không có kết luận về abstention, tiếng Việt, multimodal hay relation. Kiểm tra IoU độc lập tính lại 4/12 và 1/23 trùng evaluator. Không đo SLA/latency, không suy rộng Tesla T4 sang RTX 3060.

## Truy vết

- GPU artifacts: `work/kaggle/blender_visual_gpu_v2/videntra_visual_pilot/manifest.json` và file embeddings/coordinates đã hash.
- Prediction: `work/experiments/blender_visual_predictions_20261110.jsonl`.
- Kết quả/per-query: `work/experiments/blender_visual_evaluation_20261110.json`.
- Source qrels evaluator-only: `work/evaluation-only/momentbench_0.2.5_film_scored/`.
- Dataset/kernel Kaggle là private; output và log giữ lại để tái kiểm tra.