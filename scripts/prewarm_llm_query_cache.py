"""Prewarm data/cache/query_decomposition_cache.jsonl over dev_150 validation queries.

Never touches heldout_test_201 or test_queries/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sqlite3
import statistics
import sys
from typing import Any

# Ensure src/ is importable when invoked as `python scripts/prewarm_llm_query_cache.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend  # noqa: E402


def load_indexed_videos(db_path: Path) -> set[str]:
    """Load indexed video IDs from metadata.db without initializing visual models."""
    if not db_path.exists():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT DISTINCT video_id FROM Videos").fetchall()
    return {str(r[0]) for r in rows if r and r[0]}


def load_dev150_queries(val_dir: Path, indexed_videos: set[str], max_msrvtt: int = 100) -> list[dict[str, Any]]:
    """Load dev_150 queries strictly from validation/ (never touching test_queries/)."""
    queries: list[dict[str, Any]] = []

    # 1. QVHighlights
    qvh_file = val_dir / "qvhighlights" / "highlight_val_release.jsonl"
    if qvh_file.exists():
        with qvh_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("vid") in indexed_videos:
                        queries.append(
                            {
                                "dataset": "qvhighlights",
                                "query_id": f"qvh_{item['qid']}",
                                "query_text": item["query"],
                            }
                        )

    # 2. AVC
    avc_file = val_dir / "avc" / "candidate_plan_inference.json"
    pilot_file = Path("data/raw/avc/provenance/data/processed/avc_1.8m/manifests/pilot_queries_eval_only.json")
    pilot_targets: dict[str, str] = {}
    if pilot_file.exists():
        with pilot_file.open("r", encoding="utf-8") as pf:
            for p_item in json.load(pf):
                pilot_targets[p_item["query_id"]] = p_item["known_target_window_id"]

    if avc_file.exists():
        with avc_file.open("r", encoding="utf-8") as f:
            avc_data = json.load(f)
            for item in avc_data.get("queries", []):
                target_vid = pilot_targets.get(item["query_id"])
                if not target_vid:
                    continue
                valid_cands = [c for c in item.get("candidate_ids", []) if c in indexed_videos]
                if valid_cands and target_vid in indexed_videos:
                    queries.append(
                        {
                            "dataset": "avc",
                            "query_id": f"avc_{item['query_id']}",
                            "query_text": item["query_text"],
                        }
                    )

    # 3. MSR-VTT (seed 42, max 100 -> dev_150)
    msr_file = val_dir / "msrvtt" / "test_videodatainfo.json"
    if msr_file.exists():
        with msr_file.open("r", encoding="utf-8") as f:
            msr_data = json.load(f)
            valid_sentences = [s for s in msr_data.get("sentences", []) if s.get("video_id") in indexed_videos]
            by_vid: dict[str, dict[str, Any]] = {}
            for s in valid_sentences:
                vid = s["video_id"]
                if vid not in by_vid:
                    by_vid[vid] = s
            sample_sentences = list(by_vid.values())
            rng = random.Random(42)
            rng.shuffle(sample_sentences)
            if max_msrvtt > 0:
                sample_sentences = sample_sentences[:max_msrvtt]
            for s in sample_sentences:
                queries.append(
                    {
                        "dataset": "msrvtt",
                        "query_id": f"msr_{s.get('sen_id', s['video_id'])}",
                        "query_text": s["caption"],
                    }
                )

    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Prewarm LLM query decomposition cache on dev_150.")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/metadata.db"))
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--cache-path", type=Path, default=AdaptiveQueryDecomposer.DEFAULT_CACHE_PATH)
    parser.add_argument("--deadline-ms", type=int, default=None, help="Optional explicit LLM deadline_ms override.")
    parser.add_argument("--max-msrvtt", type=int, default=100)
    args = parser.parse_args()

    indexed_videos = load_indexed_videos(args.db_path)
    queries = load_dev150_queries(args.val_dir, indexed_videos, max_msrvtt=args.max_msrvtt)
    print(f"Loaded {len(queries)} dev_150 queries from {args.val_dir} (indexed videos: {len(indexed_videos)}).")

    backend = LocalSmallLLMBackend(deadline_ms=args.deadline_ms)
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=args.cache_path,
        enable_disk_cache=True,
    )
    print(
        f"Backend: model_revision={backend.model_revision}, deadline_ms={backend.deadline_ms}, "
        f"cache_path={args.cache_path}"
    )

    tier_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    mode_counts: dict[str, int] = {"fast_rule": 0, "tri_modal_broadcast": 0, "llm_compact": 0, "fallback": 0}
    cache_hits = 0
    llm_latencies_ms: list[float] = []

    for idx, q in enumerate(queries, start=1):
        plan = decomposer.decompose(q["query_text"])
        tier_counts[plan.query_length_tier] = tier_counts.get(plan.query_length_tier, 0) + 1
        mode_counts[plan.routing_mode] = mode_counts.get(plan.routing_mode, 0) + 1
        if plan.cache_hit:
            cache_hits += 1
        elif plan.query_length_tier in ("medium", "long"):
            llm_latencies_ms.append(plan.decomposer_latency_ms)

        if idx % 25 == 0 or idx == len(queries):
            print(
                f"  [{idx}/{len(queries)}] tiers={tier_counts} "
                f"llm_compact={mode_counts['llm_compact']} fallback={mode_counts['fallback']}"
            )

    mean_lat = statistics.mean(llm_latencies_ms) if llm_latencies_ms else 0.0
    p50_lat = statistics.median(llm_latencies_ms) if llm_latencies_ms else 0.0

    print("\n=== Prewarm Summary (dev_150) ===")
    print(f"Total queries      : {len(queries)}")
    print(f"Tiers              : short={tier_counts['short']}, medium={tier_counts['medium']}, long={tier_counts['long']}")
    print(
        f"Routing modes      : fast_rule={mode_counts['fast_rule']}, "
        f"tri_modal_broadcast={mode_counts['tri_modal_broadcast']}, "
        f"llm_compact={mode_counts['llm_compact']}, fallback={mode_counts['fallback']}"
    )
    print(f"Cache hits         : {cache_hits}")
    print(f"LLM calls (uncached medium+long): {len(llm_latencies_ms)}")
    print(f"LLM latency mean   : {mean_lat:.2f} ms")
    print(f"LLM latency p50    : {p50_lat:.2f} ms")
    print(f"Cache file exists  : {args.cache_path.exists()} ({args.cache_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
