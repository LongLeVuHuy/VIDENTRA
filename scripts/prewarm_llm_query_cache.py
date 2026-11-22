"""Prewarm data/cache/query_plans/query_decomposition_plans.jsonl across validation and/or heldout queries.

Supports:
- dev_150 (validation with max_msrvtt=100)
- dev_full_859 (full validation/)
- heldout_test_201 (full test_queries/)
- all_1060 (dev_full_859 + heldout_test_201 = 1,060 queries)

Strictly extracts only query_id / dataset / query_text (never stores or passes
ground-truth target_videos, target_windows, or timestamps to the LLM or cache).
Writes/merges the JSONL cache atomically and supports resumable checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
from typing import Any, Sequence

# Ensure src/ is importable when invoked as `python scripts/prewarm_llm_query_cache.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.query.llm import AdaptiveQueryDecomposer, LocalSmallLLMBackend  # noqa: E402

DEFAULT_PILOT_FILE = Path(
    "data/raw/avc/provenance/data/processed/avc_1.8m/manifests/pilot_queries_eval_only.json"
)
VALID_SPLITS = ("dev_150", "dev_full_859", "heldout_test_201", "all_1060")


def _get_validator_commit() -> str | None:
    """Return full HEAD git commit hash of the validator/repository if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def load_indexed_videos(db_path: Path) -> set[str]:
    """Load indexed video IDs from metadata.db without initializing visual models."""
    if not db_path.exists():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT DISTINCT video_id FROM Videos").fetchall()
    return {str(r[0]) for r in rows if r and r[0]}


def _load_queries_from_dir(
    root_dir: Path,
    indexed_videos: set[str] | None,
    max_msrvtt: int | None = 100,
    pilot_file: Path = DEFAULT_PILOT_FILE,
) -> list[dict[str, str]]:
    """Load query texts from a validation/ or test_queries/ directory without GT leakage."""
    queries: list[dict[str, str]] = []

    def _is_indexed(vid: str | None) -> bool:
        if indexed_videos is None:
            return bool(vid)
        return vid in indexed_videos

    # 1. QVHighlights (only qid, vid, query; never read relevant_windows)
    qvh_file = root_dir / "qvhighlights" / "highlight_val_release.jsonl"
    if qvh_file.exists():
        with qvh_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if _is_indexed(item.get("vid")):
                        queries.append(
                            {
                                "dataset": "qvhighlights",
                                "query_id": f"qvh_{item['qid']}",
                                "query_text": str(item["query"]),
                            }
                        )

    # 2. AVC (pilot_targets used solely for candidate/video membership filter; never stored)
    avc_file = root_dir / "avc" / "candidate_plan_inference.json"
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
                valid_cands = [c for c in item.get("candidate_ids", []) if _is_indexed(c)]
                if valid_cands and _is_indexed(target_vid):
                    queries.append(
                        {
                            "dataset": "avc",
                            "query_id": f"avc_{item['query_id']}",
                            "query_text": str(item["query_text"]),
                        }
                    )

    # 3. MSR-VTT (seed 42; max_msrvtt=100 -> dev_150, None/0 -> all)
    msr_file = root_dir / "msrvtt" / "test_videodatainfo.json"
    if msr_file.exists():
        with msr_file.open("r", encoding="utf-8") as f:
            msr_data = json.load(f)
            valid_sentences = [
                s for s in msr_data.get("sentences", []) if _is_indexed(s.get("video_id"))
            ]
            by_vid: dict[str, dict[str, Any]] = {}
            for s in valid_sentences:
                vid = s["video_id"]
                if vid not in by_vid:
                    by_vid[vid] = s
            sample_sentences = list(by_vid.values())
            rng = random.Random(42)
            rng.shuffle(sample_sentences)
            if max_msrvtt is not None and max_msrvtt > 0:
                sample_sentences = sample_sentences[:max_msrvtt]
            for s in sample_sentences:
                queries.append(
                    {
                        "dataset": "msrvtt",
                        "query_id": f"msr_{s.get('sen_id', s['video_id'])}",
                        "query_text": str(s["caption"]),
                    }
                )

    return queries


def load_dev150_queries(
    val_dir: Path,
    indexed_videos: set[str],
    max_msrvtt: int | None = 100,
) -> list[dict[str, str]]:
    """Backwards-compatible loader for dev_150 queries strictly from validation/."""
    return _load_queries_from_dir(val_dir, indexed_videos, max_msrvtt=max_msrvtt)


def load_split_queries(
    split: str,
    val_dir: Path,
    test_dir: Path,
    indexed_videos: set[str] | None,
    max_msrvtt: int | None = 100,
    pilot_file: Path = DEFAULT_PILOT_FILE,
) -> list[dict[str, str]]:
    """Load queries for dev_150, dev_full_859, heldout_test_201, or all_1060 without GT leakage."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {VALID_SPLITS}")

    if split == "dev_150":
        return _load_queries_from_dir(val_dir, indexed_videos, max_msrvtt=max_msrvtt, pilot_file=pilot_file)
    if split == "dev_full_859":
        return _load_queries_from_dir(val_dir, indexed_videos, max_msrvtt=None, pilot_file=pilot_file)
    if split == "heldout_test_201":
        return _load_queries_from_dir(test_dir, indexed_videos, max_msrvtt=None, pilot_file=pilot_file)

    # all_1060 = dev_full_859 + heldout_test_201
    val_queries = _load_queries_from_dir(val_dir, indexed_videos, max_msrvtt=None, pilot_file=pilot_file)
    test_queries = _load_queries_from_dir(test_dir, indexed_videos, max_msrvtt=None, pilot_file=pilot_file)
    return val_queries + test_queries


def load_queries_jsonl(queries_file: Path) -> list[dict[str, str]]:
    """Load pre-exported queries JSONL (strictly query_id, dataset, query_text)."""
    if not queries_file.is_file():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")
    queries: list[dict[str, str]] = []
    with queries_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("query_text") or item.get("query") or "").strip()
            if not text:
                continue
            queries.append(
                {
                    "dataset": str(item.get("dataset", "custom")),
                    "query_id": str(item.get("query_id") or item.get("qid") or f"q_{idx}"),
                    "query_text": text,
                }
            )
    return queries


def load_cache_records(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load valid JSONL cache records keyed by cache key, skipping blank/corrupted lines."""
    records: dict[str, dict[str, Any]] = {}
    if not cache_path.is_file():
        return records
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("key") and isinstance(rec.get("plan"), dict):
                records[str(rec["key"])] = rec
    return records


def atomic_write_jsonl(target_path: Path, rows: Sequence[dict[str, Any]]) -> int:
    """Atomically write JSONL records via temp file + fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return len(rows)


def atomic_write_json(target_path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON object via temp file + fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_cache(cache_path: Path, records_by_key: dict[str, dict[str, Any]]) -> int:
    """Atomically persist deduplicated cache records to cache_path."""
    return atomic_write_jsonl(cache_path, list(records_by_key.values()))


def merge_cache_files(target_path: Path, source_paths: Sequence[Path]) -> dict[str, int]:
    """Atomically merge one or more JSONL cache files into target_path, deduplicating by key."""
    merged = load_cache_records(target_path)
    existing_count = len(merged)
    source_records = 0
    for src in source_paths:
        src_map = load_cache_records(src)
        source_records += len(src_map)
        merged.update(src_map)
    total = atomic_write_cache(target_path, merged)
    return {
        "existing": existing_count,
        "source_records": source_records,
        "added": max(0, total - existing_count),
        "total": total,
    }


def _resolve_hf_model_dir(hf_model_id: str, hf_revision: str | None) -> Path:
    """Download/resolve a HuggingFace model snapshot for Kaggle GPU execution."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": hf_model_id}
    if hf_revision:
        kwargs["revision"] = hf_revision
    return Path(snapshot_download(**kwargs))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prewarm LLM query decomposition cache across dev_150, dev_full_859, heldout_test_201, or all_1060."
    )
    parser.add_argument("--split", choices=VALID_SPLITS, default="dev_150")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/catalog/corpus_metadata.db"))
    parser.add_argument("--allow-missing-db", action="store_true", help="Allow loading queries without metadata.db filtering.")
    parser.add_argument("--val-dir", type=Path, default=Path("validation"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_queries"))
    parser.add_argument("--pilot-file", type=Path, default=DEFAULT_PILOT_FILE)
    parser.add_argument("--queries-file", type=Path, default=None, help="Optional pre-exported queries JSONL (bypasses val/test dirs).")
    parser.add_argument("--export-queries-only", type=Path, default=None, help="Atomically export selected split queries (zero GT) to JSONL and exit.")
    parser.add_argument("--cache-path", type=Path, default=AdaptiveQueryDecomposer.DEFAULT_CACHE_PATH)
    parser.add_argument("--merge-from", type=Path, nargs="*", default=None, help="Merge external JSONL cache file(s) into --cache-path atomically.")
    parser.add_argument("--merge-only", action="store_true", help="Only run --merge-from into --cache-path and exit.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Atomically compact/checkpoint cache every N queries.")
    parser.add_argument("--deadline-ms", type=int, default=None, help="Optional explicit LLM deadline_ms override.")
    parser.add_argument("--max-msrvtt", type=int, default=100, help="Cap MSR-VTT queries for dev_150 split.")
    parser.add_argument("--model-path", type=Path, default=None, help="Local GGUF file or HuggingFace model directory.")
    parser.add_argument("--model-revision", type=str, default=None, help="Model revision tag stored in cache key.")
    parser.add_argument("--prompt-version", type=str, default="compact-v1", help="Query decomposer prompt version (e.g. compact-v1, extractive-v2).")
    parser.add_argument("--receipt-path", type=Path, default=None, help="Optional output JSON receipt path.")
    parser.add_argument("--hf-model-id", type=str, default=None, help="Optional HF repo ID (e.g. Qwen/Qwen2.5-1.5B-Instruct) to resolve on Kaggle.")
    parser.add_argument("--hf-revision", type=str, default=None, help="Optional HF revision/commit hash.")
    parser.add_argument("--backend", choices=("auto", "llama_cpp", "transformers"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--enable-llm", dest="enabled", action="store_true", default=None)
    parser.add_argument("--no-enable-llm", dest="enabled", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Inspect query counts and existing cache hits without running LLM inference.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.merge_from:
        stats = merge_cache_files(args.cache_path, args.merge_from)
        print(
            f"Merged cache into {args.cache_path}: existing={stats['existing']}, "
            f"source_records={stats['source_records']}, added={stats['added']}, total={stats['total']}"
        )
        if args.merge_only:
            return 0
    elif args.merge_only:
        raise ValueError("--merge-only requires --merge-from")

    if args.queries_file is not None:
        queries = load_queries_jsonl(args.queries_file)
        source_desc = str(args.queries_file)
    else:
        indexed_videos: set[str] | None = None
        if args.db_path.exists():
            indexed_videos = load_indexed_videos(args.db_path)
        elif not args.allow_missing_db:
            raise FileNotFoundError(f"Metadata DB not found: {args.db_path} (pass --allow-missing-db or --queries-file)")
        queries = load_split_queries(
            split=args.split,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            indexed_videos=indexed_videos,
            max_msrvtt=args.max_msrvtt,
            pilot_file=args.pilot_file,
        )
        source_desc = f"split={args.split} (val={args.val_dir}, test={args.test_dir})"

    print(f"Loaded {len(queries)} queries from {source_desc}.")

    if args.export_queries_only is not None:
        written = atomic_write_jsonl(args.export_queries_only, queries)
        print(f"Exported {written} GT-free queries to {args.export_queries_only}")
        return 0

    model_path = args.model_path
    if model_path is None and args.hf_model_id and not args.dry_run:
        model_path = _resolve_hf_model_dir(args.hf_model_id, args.hf_revision)

    backend = LocalSmallLLMBackend(
        model_path=model_path,
        model_revision=args.model_revision,
        prompt_version=args.prompt_version,
        backend=args.backend,
        deadline_ms=args.deadline_ms,
        enabled=False if args.dry_run else args.enabled,
        max_new_tokens=args.max_new_tokens,
    )
    decomposer = AdaptiveQueryDecomposer(
        llm_backend=backend,
        cache_path=args.cache_path,
        model_revision=args.model_revision,
        prompt_version=args.prompt_version,
        enable_disk_cache=not args.dry_run,
    )
    print(
        f"Backend: model_revision={backend.model_revision}, prompt_version={decomposer.prompt_version}, "
        f"deadline_ms={backend.deadline_ms}, enabled={backend.enabled}, cache_path={args.cache_path}"
    )

    if args.dry_run:
        existing_records = load_cache_records(args.cache_path)
        cached_count = sum(1 for q in queries if decomposer._make_cache_key(q["query_text"]) in existing_records)
        print(f"Dry-run: {cached_count}/{len(queries)} queries already present in {args.cache_path}")
        if args.receipt_path is not None:
            atomic_write_json(
                args.receipt_path,
                {
                    "schema_version": "1.0",
                    "dry_run": True,
                    "split": args.split,
                    "model_revision": decomposer.model_revision,
                    "hf_model_id": args.hf_model_id,
                    "hf_revision": args.hf_revision,
                    "prompt_version": decomposer.prompt_version,
                    "validator_commit": _get_validator_commit(),
                    "total_queries": len(queries),
                    "cached_matches": cached_count,
                    "cache_path": str(args.cache_path),
                    "cache_records": len(existing_records),
                    "cache_sha256": sha256_file(args.cache_path) if args.cache_path.is_file() else None,
                },
            )
        return 0

    # Compact/sanitize any existing cache file before starting resumable loop
    records_by_key = load_cache_records(args.cache_path)
    if args.cache_path.exists() and records_by_key:
        atomic_write_cache(args.cache_path, records_by_key)

    tier_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    mode_counts: dict[str, int] = {"fast_rule": 0, "tri_modal_broadcast": 0, "llm_compact": 0, "fallback": 0}
    cache_hits = 0
    llm_latencies_ms: list[float] = []
    ckpt_every = max(1, int(args.checkpoint_every))

    for idx, q in enumerate(queries, start=1):
        plan = decomposer.decompose(q["query_text"])
        key = decomposer._make_cache_key(q["query_text"])
        q_sha = hashlib.sha256(plan.original_query.encode("utf-8")).hexdigest()
        records_by_key[key] = {
            "key": key,
            "model_revision": decomposer.model_revision,
            "prompt_version": decomposer.prompt_version,
            "query_text_sha256": q_sha,
            "plan": plan.to_dict(),
        }

        tier_counts[plan.query_length_tier] = tier_counts.get(plan.query_length_tier, 0) + 1
        mode_counts[plan.routing_mode] = mode_counts.get(plan.routing_mode, 0) + 1
        if plan.cache_hit:
            cache_hits += 1
        elif plan.query_length_tier in ("medium", "long"):
            llm_latencies_ms.append(plan.decomposer_latency_ms)

        if idx % ckpt_every == 0 or idx == len(queries):
            atomic_write_cache(args.cache_path, records_by_key)
            print(
                f"  [{idx}/{len(queries)}] tiers={tier_counts} "
                f"llm_compact={mode_counts['llm_compact']} fallback={mode_counts['fallback']} "
                f"cached_total={len(records_by_key)}"
            )

    mean_lat = statistics.mean(llm_latencies_ms) if llm_latencies_ms else 0.0
    p50_lat = statistics.median(llm_latencies_ms) if llm_latencies_ms else 0.0

    print(f"\n=== Prewarm Summary ({args.split}) ===")
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
    if not queries:
        atomic_write_cache(args.cache_path, records_by_key)

    cache_sha256 = sha256_file(args.cache_path) if args.cache_path.is_file() else None
    if args.receipt_path is not None:
        atomic_write_json(
            args.receipt_path,
            {
                "schema_version": "1.0",
                "dry_run": False,
                "split": args.split,
                "model_revision": decomposer.model_revision,
                "hf_model_id": args.hf_model_id,
                "hf_revision": args.hf_revision,
                "prompt_version": decomposer.prompt_version,
                "validator_commit": _get_validator_commit(),
                "total_queries": len(queries),
                "tier_counts": tier_counts,
                "routing_mode_counts": mode_counts,
                "cache_hits": cache_hits,
                "llm_calls": len(llm_latencies_ms),
                "cache_path": str(args.cache_path),
                "cache_records": len(records_by_key),
                "cache_sha256": cache_sha256,
            },
        )

    print(f"Cache file exists  : {args.cache_path.exists()} ({args.cache_path}, {len(records_by_key)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
