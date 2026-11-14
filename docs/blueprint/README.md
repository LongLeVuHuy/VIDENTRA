# VIDENTRA blueprint — mục lục

Ngày04/11/2026 · v1.0 PROPOSED · phần cứng đích RTX3060 6GB +32GB RAM.

Đọc [quyết định](00_EXECUTIVE_DECISION.md) rồi [pipeline đầy đủ](05_VIDENTRA_DETAILED_PIPELINE.md), [schemas/timestamps](06_DATA_SCHEMAS_AND_TIMESTAMPS.md) và [roadmap](10_IMPLEMENTATION_ROADMAP.md). Không triển khai engine trước design review. Không có throughput/p95 hoặc retrieval accuracy mới được đo.

| Tài liệu | Nội dung |
|---|---|
| [00_EXECUTIVE_DECISION.md](00_EXECUTIVE_DECISION.md) | kiến trúc B tạm chọn, A baseline, C conditional |
| [01_OLD_PROJECT_RETIREMENT.md](01_OLD_PROJECT_RETIREMENT.md) | inventory, verified backup, migration/deletion gate |
| [02_PROJECT_REQUIREMENTS.md](02_PROJECT_REQUIREMENTS.md) | requirements, hardware và phép tính10k h |
| [03_RESEARCH_AND_MODEL_REVIEW.md](03_RESEARCH_AND_MODEL_REVIEW.md) | primary sources, releases, licenses, runnable limitations |
| [04_ARCHITECTURE_COMPARISON.md](04_ARCHITECTURE_COMPARISON.md) | A/B/C và strategies candidate generation |
| [05_VIDENTRA_DETAILED_PIPELINE.md](05_VIDENTRA_DETAILED_PIPELINE.md) |18 subsections, stages/algorithms/contracts/examples |
| [06_DATA_SCHEMAS_AND_TIMESTAMPS.md](06_DATA_SCHEMAS_AND_TIMESTAMPS.md) | exact IDs/PTS/frame/source maps/evidence schemas |
| [07_GROUND_TRUTH_AND_DATASETS.md](07_GROUND_TRUTH_AND_DATASETS.md) | dataset stack, human review, exposure/anti-leakage |
| [08_BENCHMARK_AND_LATENCY_PLAN.md](08_BENCHMARK_AND_LATENCY_PLAN.md) | E01–E14, metrics, ablations, resource profiling |
| [09_COMPETITION_UI_SPEC.md](09_COMPETITION_UI_SPEC.md) | wireframes, TopK, playback/frame-step/API/export |
| [10_IMPLEMENTATION_ROADMAP.md](10_IMPLEMENTATION_ROADMAP.md) | M0–M9 executable milestone contracts |
| [11_RISKS_AND_DECISION_LOG.md](11_RISKS_AND_DECISION_LOG.md) | D01–D14, risks, assumptions và validation |
| [12_NEW_REPOSITORY_STRUCTURE.md](12_NEW_REPOSITORY_STRUCTURE.md) | actual planning vs planned code structure/policies |

[Repository README](../../README.md) · [Query schema](../schemas/query_graph.schema.json) · [Local backup receipt](../audit/backup_verification.json) · [Supplemental restore](../audit/restore_supplement.json).

