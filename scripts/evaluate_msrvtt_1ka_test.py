"""Single-shot evaluation of sealed MSR-VTT 1K-A (JSFusion) after Champion lock.

Protocol guarantees:
1. Fails closed unless `champion_lock_receipt.json` exists and `test_1ka_evaluated`
   is False (enforcing single-shot sealed test evaluation).
2. Verifies `eval_only/test_1ka_split_lock.json` integrity before scoring.
3. Evaluates strictly B1 (VideoPrism-LvT-B), B0 (SigLIP2), and the single locked
   Champion using the exact frozen hyperparameters and train-fit weights.
4. Reports Hit@1/5/10, MedR, MeanR, MRR@10, paired 95% bootstrap CI vs B1,
   per-category Hit@1, and error breakdown if Hit@1 < 501/1000.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_msrvtt_validation_matrix import (  # noqa: E402
    PROTOCOL_ROOT,
    _evaluate_split_matrix,
    _load_labels,
    _load_train_query_banks,
    _load_vl2_train_query_banks,
    _read_jsonl,
)
from scripts.run_validation_benchmark import get_process_memory_mb  # noqa: E402
from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.ranking.learned_ranker import LearnedRankerLTR  # noqa: E402
from videntra.retrieval.msrvtt_matrix import load_gallery_feature_bank  # noqa: E402


def _collect_existing_dirs(candidates: list[Path]) -> list[Path]:
    return [p for p in candidates if p.exists()]


def _compute_category_breakdown(
    preds: list[dict[str, Any]],
    labels_by_qid: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for p in preds:
        qid = str(p["query_id"])
        lab = labels_by_qid[qid]
        cat = str(lab.get("category_name") or lab.get("category", "unknown"))
        hit = 1 if p["ranked_video_ids"][0] in set(lab["target_videos"]) else 0
        by_cat[cat].append(hit)
    out: dict[str, dict[str, Any]] = {}
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        out[cat] = {
            "n_queries": len(vals),
            "hits_at_1": int(sum(vals)),
            "hit_at_1": round(sum(vals) / max(1, len(vals)), 6),
        }
    return out


def _build_failure_breakdown(
    preds: list[dict[str, Any]],
    labels_by_qid: dict[str, dict[str, Any]],
    queries_by_qid: dict[str, str],
) -> dict[str, Any]:
    miss_buckets: Counter[str] = Counter()
    miss_examples: list[dict[str, Any]] = []
    for p in preds:
        qid = str(p["query_id"])
        targets = set(labels_by_qid[qid]["target_videos"])
        ranked = p["ranked_video_ids"]
        if ranked and ranked[0] in targets:
            continue
        rank_pos = None
        for idx, vid in enumerate(ranked, start=1):
            if vid in targets:
                rank_pos = idx
                break
        if rank_pos is None:
            miss_buckets["outside_top20"] += 1
        elif rank_pos <= 5:
            miss_buckets["rank_2_to_5"] += 1
        elif rank_pos <= 10:
            miss_buckets["rank_6_to_10"] += 1
        else:
            miss_buckets["rank_11_to_20"] += 1
        if len(miss_examples) < 25:
            miss_examples.append({
                "query_id": qid,
                "query_text": queries_by_qid.get(qid, ""),
                "target_videos": sorted(targets),
                "top1_predicted": ranked[0] if ranked else None,
                "gold_rank_in_top20": rank_pos,
            })
    return {
        "total_misses": int(sum(miss_buckets.values())),
        "rank_buckets": dict(miss_buckets),
        "sample_misses": miss_examples,
    }


def _format_test_report(payload: dict[str, Any]) -> str:
    res = payload["results"]
    champion_id = payload["champion_id"]
    ch = res[champion_id]
    b1 = res["B1_videoprism_standalone"]
    b0 = res["B0_siglip2_standalone"]
    target_met = ch["hits_at_1"] >= 501
    lines = [
        "# MSR-VTT 1K-A (JSFusion) Official Benchmark Report\n\n",
        f"- **Evaluated At (UTC):** {payload['evaluated_at_utc']}\n",
        f"- **Champion Config:** `{champion_id}`\n",
        f"- **Target `>= 501 / 1000` (> 50% Video Hit@1):** **{'ACHIEVED' if target_met else 'NOT MET'}** ({ch['hits_at_1']}/{ch['n_queries']} = {ch['hit_at_1']*100:.2f}%)\n",
        f"- **Peak RAM:** {payload['runtime_memory']['peak_ram_mb']} MB | **Seed:** `{payload['seed']}`\n\n",
        "## Headline Comparison on `test_1ka` (1,000 Videos / 1,000 Queries)\n\n",
        "| Model | Hits / N | Hit@1 | Hit@5 | Hit@10 | S1 R@20 | MRR@10 | Delta vs B1 (pp) | 95% CI | +W/-L | p50/p95 ms |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    ]
    for cid in ("B1_videoprism_standalone", "B0_siglip2_standalone", champion_id):
        r = res[cid]
        ci = r["ci95_pp_vs_b1"]
        lines.append(
            f"| `{cid}` | {r['hits_at_1']}/{r['n_queries']} | **{r['hit_at_1']*100:.2f}%** | "
            f"{r['hit_at_5']*100:.2f}% | {r['hit_at_10']*100:.2f}% | {r['stage1_recall_at_20']*100:.2f}% | "
            f"{r['mrr_at_10']:.4f} | {r['delta_pp_vs_b1']:+.2f} | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | +{r['rescued_vs_b1']}/-{r['lost_vs_b1']} | "
            f"{r['latency_p50_ms']:.2f}/{r['latency_p95_ms']:.2f} |\n"
        )

    lines.append(f"\n### Improvement vs VideoPrism (`B1`) = {ch['hit_at_1']*100 - b1['hit_at_1']*100:+.2f} pp (+{ch['rescued_vs_b1']}/-{ch['lost_vs_b1']}), vs SigLIP2 (`B0`) = {ch['hit_at_1']*100 - b0['hit_at_1']*100:+.2f} pp\n\n")
    lines.append("## Per-Category Hit@1 (`Champion`)\n\n| Category | Hits / N | Hit@1 |\n| :--- | :---: | :---: |\n")
    for cat, st in payload["category_breakdown"].items():
        lines.append(f"| {cat} | {st['hits_at_1']}/{st['n_queries']} | {st['hit_at_1']*100:.2f}% |\n")

    if payload.get("failure_breakdown") is not None:
        fb = payload["failure_breakdown"]
        lines.append("\n## Error Breakdown (< 501/1000 Target Diagnostic)\n\n")
        lines.append(f"- Total misses: {fb['total_misses']}\n")
        lines.append(f"- Rank buckets: `{json.dumps(fb['rank_buckets'])}`\n")
    return "".join(lines)


def evaluate_sealed_test_1ka(
    *,
    protocol_root: Path = PROTOCOL_ROOT,
    experiment_dir: Path | None = None,
    benchmark_dir: Path = Path("benchmark"),
) -> dict[str, Any]:
    t0 = time.perf_counter()
    lock_path = protocol_root / "champion_lock_receipt.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Champion not locked yet: missing {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_1ka_evaluated"):
        raise RuntimeError(
            "Sealed test_1ka was ALREADY evaluated under this champion_lock_receipt.json; refusing duplicate test run."
        )

    e_dir = protocol_root / "eval_only"
    m_dir = protocol_root / "manifests" / "test_1ka"
    split_lock = json.loads((e_dir / "test_1ka_split_lock.json").read_text(encoding="utf-8"))
    expected_gal_sha = split_lock.get("gallery_manifest_sha256")
    actual_gal_sha = sha256_file(m_dir / "gallery_manifest.jsonl")
    if expected_gal_sha and actual_gal_sha != expected_gal_sha:
        raise ValueError(f"test_1ka gallery manifest hash mismatch: expected {expected_gal_sha}, got {actual_gal_sha}")

    k_out = protocol_root / "kaggle_out"
    vp_dirs = _collect_existing_dirs([k_out / "vp_test_1ka_locked", k_out / "vp_train_sub", k_out / "vp_test_1ka_queries_only"])
    mm_dirs = _collect_existing_dirs([k_out / "mm_test_1ka_locked", k_out / "mm_test_1ka_videos_only", k_out / "mm_test_1ka_queries_only"])
    asr_dirs = _collect_existing_dirs([k_out / "asr_test_1ka_videos_only", k_out / "asr_test_1ka_locked"])
    cap_dirs = _collect_existing_dirs([k_out / "cap_test_1ka_locked", k_out / "cap_val_4000", k_out / "cap_val_tune", k_out / "cap_test_1ka_queries_only"])
    vl2_dirs = _collect_existing_dirs([
        k_out / "vl2_test_1ka_locked",
        k_out / "vl2_train_sub",
        *sorted(p for p in k_out.glob("vl2_test_1ka_videos*") if p.is_dir()),
        k_out / "vl2_test_1ka_queries_only",
    ])
    if not vp_dirs:
        raise FileNotFoundError("Missing VideoPrism features for test_1ka (run vp-shard --target test_1ka_locked)")

    train_vp_bank, train_sig_bank = _load_train_query_banks(k_out / "vp_train_sub", k_out / "mm_train_sub")
    train_pe_bank, train_sig400_bank = _load_vl2_train_query_banks(k_out / "vl2_train_sub")
    ltr_path = Path(lock["train_models"]["ltr_path"])
    ltr_ranker = LearnedRankerLTR(model_path=ltr_path) if ltr_path.is_file() else None
    vedje_ckpt = Path(lock["train_models"]["vedje_ckpt"])

    test_queries_path = e_dir / "test_1ka_queries_vault.jsonl"
    test_labels_path = e_dir / "test_1ka_labels.jsonl"
    test_bank = load_gallery_feature_bank(
        gallery_name="test_1ka",
        gallery_manifest_path=m_dir / "gallery_manifest.jsonl",
        queries_path=test_queries_path,
        vp_dirs=vp_dirs,
        mm_dirs=mm_dirs,
        asr_dirs=asr_dirs,
        cap_dirs=cap_dirs,
        vl2_dirs=vl2_dirs,
        train_vp_bank=train_vp_bank,
        train_sig_bank=train_sig_bank,
        train_pe_bank=train_pe_bank,
        train_sig400_bank=train_sig400_bank,
    )
    labels_by_qid = _load_labels(test_labels_path)
    champion_id = str(lock["champion_id"])
    only_configs = ("B1_videoprism_standalone", "B0_siglip2_standalone", champion_id)

    results, _, all_preds = _evaluate_split_matrix(
        "test_1ka",
        [test_bank],
        labels_by_qid,
        train_vp_bank=train_vp_bank,
        train_sig_bank=train_sig_bank,
        ltr_ranker=ltr_ranker,
        vedje_ckpt=vedje_ckpt,
        tuned_params=lock["tuned_hyperparameters"],
        seed=int(lock.get("seed", 20261122)),
        only_configs=only_configs,
    )

    champ_preds = all_preds[champion_id]
    cat_breakdown = _compute_category_breakdown(champ_preds, labels_by_qid)
    queries_by_qid = {str(r["query_id"]): str(r["query_text"]) for r in _read_jsonl(test_queries_path)}
    failure_breakdown = (
        _build_failure_breakdown(champ_preds, labels_by_qid, queries_by_qid)
        if results[champion_id]["hits_at_1"] < 501
        else None
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_dir = experiment_dir or (Path("work/experiments") / f"{ts}_msrvtt_1ka_benchmark")
    exp_dir.mkdir(parents=True, exist_ok=True)
    rss_mb, peak_mb = get_process_memory_mb()

    payload = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(lock.get("seed", 20261122)),
        "champion_id": champion_id,
        "champion_lock_config_sha256": lock.get("config_sha256"),
        "target_hit_at_1_ge_501": bool(results[champion_id]["hits_at_1"] >= 501),
        "results": {cid: results[cid] for cid in dict.fromkeys(only_configs)},
        "category_breakdown": cat_breakdown,
        "failure_breakdown": failure_breakdown,
        "runtime_memory": {"rss_mb": rss_mb, "peak_ram_mb": peak_mb, "elapsed_seconds": round(time.perf_counter() - t0, 2)},
    }

    for cid in dict.fromkeys(only_configs):
        p_file = exp_dir / f"{cid}_test_1ka_predictions.jsonl"
        with p_file.open("w", encoding="utf-8") as fh:
            for row in all_preds[cid]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = _format_test_report(payload)
    (exp_dir / "test_1ka_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "test_1ka_report.md").write_text(md, encoding="utf-8")

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "msrvtt_1ka_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (benchmark_dir / "msrvtt_1ka_benchmark_report.md").write_text(md, encoding="utf-8")

    # Seal lock receipt against duplicate test evaluation
    lock["test_1ka_evaluated"] = True
    lock["test_1ka_evaluated_at_utc"] = payload["evaluated_at_utc"]
    lock["test_1ka_experiment_dir"] = str(exp_dir)
    lock["test_1ka_champion_hit_at_1"] = results[champion_id]["hit_at_1"]
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    print(md)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-shot sealed test_1ka benchmark evaluation.")
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    evaluate_sealed_test_1ka(
        protocol_root=args.protocol_root,
        experiment_dir=args.experiment_dir,
        benchmark_dir=args.benchmark_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
