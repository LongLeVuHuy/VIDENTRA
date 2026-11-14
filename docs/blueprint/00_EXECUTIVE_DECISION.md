# 00 — Quyết định kiến trúc

Ngày04/11/2026. **Chọn tạm thời Architecture B; triển khai Architecture A làm baseline trước.** Chưa có benchmark VIDENTRA; mọi nhận định quality là giả thuyết được gắn thí nghiệm. [Pipeline chuẩn](05_VIDENTRA_DETAILED_PIPELINE.md).

## Quyết định chính

1. Repository VIDENTRA độc lập, không nhập source tree/B4 EVOI/Evidence Sufficiency Ledger từ dự án cũ. Bảo tồn data và experiment provenance qua archive; utility chỉ port sau unit test và license review.
2. Timeline chuẩn từ decoded presentation PTS và frame ledger; frame index zero-based của **ingested video**, không lấy timestamp×FPS. Source offset là mapping riêng, không thể suy ra source frame ordinal khi chưa có source media.
3. Hybrid adaptive sampling có max-gap2s, cut/burst và OCR probes độc lập; không cam kết không mất short event. So matched-compute với uniform trước promote (E02).
4. SigLIP 2 Base-224 trước; Base-384 challenger. FAISS Flat benchmark, IVF-PQ sharded là scale hypothesis (E03/E05). Không một vector cho cả long video.
5. OCR/ASR primary+selective fallback, không mặc định hai model mọi frame/chunk. Model predictions luôn giữ provenance và uncertainty.
6. Independent corpus retrieval cho visual, OCR và ASR. Temporal union+branch budgets; visualTopK không quyết định OCR/ASR được tìm ở đâu (E06).
7. Query gốc giữ nguyên; small LLM chỉ query phức tạp, schema constrained, safe timeout/fallback. Original+graph so original-only/decomposition-only (E07).
8. Relations bằng role-preserving graph assignment, interval constraints và bounded perceptual adapters. Rule-only không được coi đủ cho handover/speaker identity. C optional chỉ khi E08 chứng minh tăng quality trong budget.
9. Top-K mặc định10, Moment Ranking, exact source-frame inspection và generic export; competition schema chưa biết.
10. Ground Truth tách khỏi inference. Old QVH/AVC chỉ development/diagnostic với exposure ghi rõ; official release không đồng nghĩa mọi nhãn human-authored.

## Offline và online đã chọn

Offline: register/hash/rights→decode+frame/audio maps→adaptive visual samples và text probes→SigLIP features / shared-detector OCR / VAD+ASR→typed evidence alignment→FAISS+lexical/dense text shards→atomic snapshot→thumbnails/proxy maps.

Online: original query→rules/small-LLM QueryGraph→independent branch retrieval→time-compatible proposals→RRF→role/temporal assessment→bounded localization→optional verifier→rank→API/UI. Khi evidence thiếu, trả partial và cho xem nguồn ngay.

## Vì sao B phù hợp hơn A/C ở giai đoạn này

A kiểm chứng cơ sở, dễ chạy6GB nhưng không biểu diễn role/order đầy đủ. B thêm constraints mà vẫn reuse candidate evidence và giữ original route. C thêm models nặng làm tăng phụ thuộc feature/training/VRAM; chưa có local proof rằng gain xứng cost. B được chọn vì implementation risk thấp hơn C và có lối đo incremental value, **không vì đã đo B tốt hơn**.

Cấu hình mục tiêu do người dùng xác nhận: RTX3060 6GB +32GB RAM; không phải GPU máy tạm. Offline stages chạy tuần tự GPU; online CPU FAISS/text + limited GPU text/verifier. Không chứng minh một máy này ingest10.000h nhanh hoặc chứa toàn media; storage ngoài máy bắt buộc nếu bitrate phổ biến, xem [02](02_PROJECT_REQUIREMENTS.md).

## Gates và điều còn thiếu

- Thiết kế phải được review trước engine implementation.
- E01 timeline correctness; E02 sampling; E04 OCR/ASR routing; E05 ANN; E06 union recall; E08 relation hard negatives là gates chất lượng.
- Mục tiêu warm p50 ≤1 s, p95 ≤3 s; Fast allocation 2 s và Accurate 8 s; peak RAM ≤24 GiB/VRAM ≤5 GiB. Tất cả chưa đo trên RTX 3060.
- Chưa có joint Vietnamese video GT cho mọi relation/speaker/order; cần human annotation theo [07](07_GROUND_TRUTH_AND_DATASETS.md).
- Source media rights, actual CPU/SSD/TGP, disk budget, competition format và off-device backup còn mở.
- Không xóa dữ liệu. [01](01_OLD_PROJECT_RETIREMENT.md) ghi archive đã verify và những giới hạn phục hồi.

## Quyết định sau review

Task coding đầu tiên là **M0/M1 timeline contract + fixture-based ingest**, chưa OCR/ASR/LLM. Đóng băng24-video QVH development manifest đã tiếp xúc, tách labels và tạo CFR/VFR/nonzero-PTS/audio-delay fixtures. Acceptance cụ thể ở [10](10_IMPLEMENTATION_ROADMAP.md).
