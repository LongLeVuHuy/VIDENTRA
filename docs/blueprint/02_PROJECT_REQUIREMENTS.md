# 02 — Requirements, scale và resource envelope

## Scope và acceptance

VIDENTRA tìm timestamped moments trong >10.000 giờ; Việt/Anh; semantic visual+scene-text+speech; quan hệ/temporal với evidence limitations rõ. Planning không triển khai engine, không fine-tune/download all datasets. [Pipeline](05_VIDENTRA_DETAILED_PIPELINE.md) là contract chuẩn.

| ID | Requirement | Verification |
|---|---|---|
| FR01 | stable asset/source IDs, resumable ingest | restart/idempotency fixtures E01 |
| FR02 | VFR/nonzero PTS/audio offsets, exact decoded frame ordinal | timeline round-trip E01 |
| FR03 | adaptive frames + independent text probes/audio | coverage matched budget E02/E04 |
| FR04 | SigLIP 2 local frame representations | precision/model comparison E03 |
| FR05 | selective OCR/ASR second model | routing gain/cost E04 |
| FR06 | OCR/ASR recover visual misses | forced visual-miss tests and corpus Recall E06 |
| FR07 | graph roles/AND/OR/negation/original fallback | parser corpus E07 |
| FR08 | compositional/temporal ranking, unknown state | contrast pairs E08 |
| FR09 | multi-moment boundaries, frame mapping | IoU/boundary MAE E09 |
| FR10 | UI TopK, playback, frame-step/copy/export | integration E11 |
| NFR01 | no labels during inference | process ACL/import/poison-label tests E12 |
| NFR02 | atomic snapshot/rollback | crash injection E01/E05 |
| NFR03 | warm p50 ≤1 s, p95 ≤3 s; Fast allocation 2 s, Accurate 8 s; resources bounded | p50/p95/cold/warm E10 |
| NFR04 | licenses/revisions/receipts | model/data registry gates E13 |
| NFR05 | old-project recovery | archive hash/restore E14 |

Cấu hình đích **RTX3060 6GB VRAM, RAM32GB** theo user. Máy tạm Intel HD620 không được dùng để suy ra GPU speed. CPU/TGP/OS/storage máy đích chưa biết. Giả định concurrency1, local NVMe cho index active và media proxy cache; chưa có paid/cloud commitment. GPU tasks nặng serialized. RAM24GiB/VRAM5GiB là caps planning, đo lại khi nhận hardware.

## Phép tính dung lượng (decimal GB/TB; không phải benchmark)

H=10.000h; T=36.000.000s; N=T×d samples/s. SigLIP Base d_embed=768. FP16=2N×768, FP32/Flat=4N×768. IVF-PQ96x8+ID payload=N×104. HNSW32 approximate=N×(4×768+8×32); metadata512B/sample và thumbnail40.000B/sample chỉ scenario inputs.

| Sample/s | N vectors | FP16 GB | FP32/Flat GB | PQ+ID GB | HNSW32 GB | Metadata GB | All thumbnails GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
|0,10|3.600.000|5,5296|11,0592|0,3744|11,9808|1,8432|144|
|0,25|9.000.000|13,824|27,648|0,936|29,952|4,608|360|
|0,50|18.000.000|27,648|55,296|1,872|59,904|9,216|720|
|1,00|36.000.000|55,296|110,592|3,744|119,808|18,432|1440|
|2,00|72.000.000|110,592|221,184|7,488|239,616|36,864|2880|

Default max-gap2s có floor xấp xỉ0,5/s; burst tăng density. Hai dòng0,1/0,25 chỉ sensitivity lower-density policies, không đáp ứng default safeguard. PQ numbers **chưa gồm** coarse centroids/codebooks/OPQ/direct map/allocator, training peak hoặc FP16 store để exact rerank. HNSW thêm IDMap overhead. RAM32GB không chứa Flat18M vectors; FP16 trên disk không làm CPU Flat tự dùng FP16.

Sharding1M vectors giúp build bounded RAM, nhưng fan-out mọi shard có CPU cost. Baseline capacity tiers:100k Flat~0,307GB;1M Flat~3,072GB;10M Flat~30,72GB;18M~55,3GB. Với IVF4096,d768 mỗi coarse quantizer khoảng12,58MB FP32/shard;18 shards~226MB trước graph/direct maps. Training200k vectors FP32~614MB; workspace và replicated arrays cần đo.

### Text, metadata và media

- Frame ledger full decode ở30fps:1,08 tỷ rows; packed24B/row≈25,92GB trước compression/index. Không giữ tất cả rows trong RAM; parquet by asset/row group, hot seek anchors.
- ASR scenario60% speech, utterance trung bình5s→4,32M segments. Nếu UTF-8 text100B+metadata300B→1,728GB; E5 FP16 d384 thêm3,318GB; FTS indexes/storage overhead đo thực tế. Word-level timing có thể tăng lớn.
- OCR scenario0,05/0,2/1 unique tracks mỗi giây→1,8/7,2/36M tracks. Với1KB text+metadata/track→1,8/7,2/36GB; E5 FP16 thêm1,3824/5,5296/27,648GB. Observations riêng có thể nhiều hơn tracks; budget probes không đồng nghĩa unique text rows.
- Probes1fps=36M OCR frame inspections, có thể lớn hơn18M visual samples; detector crops multiplicity chưa biết. Không bỏ chi phí này khỏi estimate.
- Thumbnails mọi frame tốn720GB ở0,5fps scenario. Chọn scene representatives + on-demand LRU cache mục tiêu20–50GB; media original vẫn là nguồn authoritative.
- Video bitrate trung bình2/5/8Mbps→9/22,5/36TB. Proxy1Mbps cho toàn corpus vẫn4,5TB. Full10k h không thể chứa trong~76GB trống của máy tạm; cần external/NAS/object storage và bandwidth budget cụ thể.
- Không giữ full decoded JPEG hoặc PCM toàn corpus. Incremental snapshots cần disk headroom old+new indexes/features, không mặc định copy tất cả raw media mỗi rebuild.

### Ingestion arithmetic, không phải tốc độ dự đoán

Đo r_decode original frames/s, r_embed selected frames/s, r_det probes/s, r_rec crops/s; ASR RTF=compute seconds/audio seconds. Thời gian stage:
`t_decode=F/r_decode; t_embed=N/r_embed; t_ocr=P/r_det+C/r_rec; t_asr=T_speech×RTF`.
Serialized GPU wall time gần tổng các GPU stages cộng I/O/stalls; pipeline overlap chỉ trừ phần đã đo, không tự max tất cả stages.

Ví dụ **sensitivity giả định**, không benchmark3060:
- N18M, r_embed50/200/500→100/25/10h chỉ embedding.
- F1,08B, r_decode100/300/1000→3000/1000/300h chỉ decode.
- 6000h speech, RTF0,05/0,2/1→300/1200/6000h primary ASR;20% dual adds60/240/1200h nếu secondary cùng RTF.
- OCR36M probes ở10/50/200fps→1000/200/50h detection, chưa recognition/tile/fallback.

Các con số cho thấy decode/OCR/ASR có thể chi phối. E10 chạy30min→10h representative corpus trên máy đích, lấy distributions/codecs/noise, rồi forecast confidence bands. Không hứa ingest trong vài ngày trước measurement.

Incremental cost ΔH gần linear selected stages; codebook drift hoặc encoder revision có thể buộc rebuild indexes/features toàn bộ. ΔH100h ở0,5fps≈180k vectors,276,48MB FP16; chưa gồm text/media. Tombstones giúp soft-delete index entry khi rights revoked, giữ audit và underlying-retention policy.

## Open requirements

Competition nào, allowed external models/network, submission frame numbering, source videos hay clips, license terms, budget disk, CPU model và target concurrent users chưa được cung cấp. Mặc định generic local competition UI; official adapter là cấu hình bổ sung, không invented schema. Không giả định “original frame index” của full source khi chỉ có downloaded clip.
