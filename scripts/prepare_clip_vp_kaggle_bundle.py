"""Prepare inference-safe Kaggle bundle and 16-frame bin-center recipes for VideoPrism-LvT-B."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.evaluation.provenance import sha256_file  # noqa: E402
from videntra.ingestion.decode import decode_frames  # noqa: E402
from videntra.timeline.ledger import exact_seek_frame  # noqa: E402

NUM_CLIP_FRAMES = 16

FORBIDDEN_GT_KEYS = frozenset({
    "target_videos",
    "target_windows",
    "relevant_videos",
    "relevant_windows",
    "split",
    "split_label",
    "gold",
    "annotations",
    "labels",
    "is_relevant",
    "target_video_ids",
    "ground_truth",
})


def sha256_text(text: str) -> str:
    """Return canonical UTF-8 SHA-256 hex digest for query text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Return deterministic SHA-256 hex digest for JSON-serializable metadata."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_no_gt_leakage(records: Sequence[dict[str, Any]], *, context: str = "bundle") -> None:
    """Fail closed if any ground-truth or split-label keys appear in inference records."""
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        leaked = sorted(FORBIDDEN_GT_KEYS.intersection(rec.keys()))
        if leaked:
            raise ValueError(f"Inference leakage guard triggered in {context}[{idx}]: {leaked}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_inference_queries(queries_path: Path) -> list[dict[str, str]]:
    """Load queries and export strictly (query_id, query_text, query_text_sha256)."""
    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    raw_rows = _read_jsonl(queries_path)
    seen_ids: set[str] = set()
    exported: list[dict[str, str]] = []

    for row in raw_rows:
        qid = str(row.get("query_id") or row.get("qid") or "").strip()
        if not qid:
            raise ValueError("Query record missing query_id")
        if qid in seen_ids:
            raise ValueError(f"Duplicate query_id in {queries_path}: {qid}")
        seen_ids.add(qid)

        qtext = row.get("query_text")
        if qtext is None:
            qtext = row.get("text")
        if qtext is None:
            qtext = row.get("query")
        if not isinstance(qtext, str) or not qtext.strip():
            raise ValueError(f"Empty or invalid query_text for {qid}")

        expected_hash = sha256_text(qtext)
        provided_hash = row.get("query_text_sha256")
        if provided_hash is not None and str(provided_hash) != expected_hash:
            raise ValueError(f"query_text_sha256 mismatch for {qid}")

        exported.append({
            "query_id": qid,
            "query_text": qtext,
            "query_text_sha256": expected_hash,
        })

    assert_no_gt_leakage(exported, context="queries")
    return exported


def _extract_pool_video_ids(row: dict[str, Any]) -> list[str]:
    for key in ("candidate_video_ids", "ranked_video_ids", "video_ids", "pool_video_ids"):
        val = row.get(key)
        if isinstance(val, list):
            return [str(v) for v in val if str(v).strip()]
    raise ValueError(f"Pool row for {row.get('query_id')} missing candidate video list")


def load_and_merge_candidate_pools(
    query_ids: Sequence[str],
    pool_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge locked pool files into inference-safe (query_id, candidate_video_ids) rows."""
    if not pool_paths:
        raise ValueError("At least one candidate pool file is required")

    per_query_seen: dict[str, set[str]] = {qid: set() for qid in query_ids}
    per_query_order: dict[str, list[str]] = {qid: [] for qid in query_ids}

    for pool_path in pool_paths:
        if not pool_path.is_file():
            raise FileNotFoundError(f"Pool file not found: {pool_path}")
        seen_in_file: set[str] = set()
        for row in _read_jsonl(pool_path):
            qid = str(row.get("query_id") or row.get("qid") or "").strip()
            if not qid:
                raise ValueError(f"Pool row missing query_id in {pool_path}")
            if qid in seen_in_file:
                raise ValueError(f"Duplicate query_id {qid} in {pool_path}")
            seen_in_file.add(qid)
            if qid not in per_query_seen:
                continue
            for vid in _extract_pool_video_ids(row):
                if vid not in per_query_seen[qid]:
                    per_query_seen[qid].add(vid)
                    per_query_order[qid].append(vid)

    candidates_rows: list[dict[str, Any]] = []
    union_videos_seen: set[str] = set()
    union_videos: list[str] = []

    for qid in query_ids:
        vids = per_query_order[qid]
        candidates_rows.append({
            "query_id": qid,
            "candidate_video_ids": vids,
        })
        for vid in vids:
            if vid not in union_videos_seen:
                union_videos_seen.add(vid)
                union_videos.append(vid)

    assert_no_gt_leakage(candidates_rows, context="candidates")
    return candidates_rows, union_videos


def select_bin_center_frames(
    frame_records: Sequence[tuple[int, float | None]],
    clip_start_sec: float,
    clip_end_sec: float,
    num_frames: int = NUM_CLIP_FRAMES,
) -> dict[str, Any]:
    """Select `num_frames` frames at uniform bin centers `start + (i + 0.5)*(end - start)/N`.

    Args:
        frame_records: Sequence of `(ordinal, timestamp_sec)` in presentation order.
        clip_start_sec: Valid clip start bound in seconds.
        clip_end_sec: Valid clip end bound in seconds (`clip_end_sec > clip_start_sec`).
        num_frames: Number of uniform bin centers (default 16).

    Returns:
        Deterministic recipe dictionary with selected ordinals, timestamps,
        `unique_frames`, `missing_mapping`, and `recipe_hash`.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if clip_end_sec <= clip_start_sec:
        raise ValueError(f"Invalid clip interval [{clip_start_sec}, {clip_end_sec})")

    valid_frames: list[tuple[int, float]] = []
    missing_mapping = 0
    for ordinal, ts_sec in frame_records:
        if ts_sec is None:
            missing_mapping += 1
            continue
        ts_val = float(ts_sec)
        if clip_start_sec <= ts_val < clip_end_sec:
            valid_frames.append((int(ordinal), ts_val))

    # Fall back to any mapped frame if strict [start, end) had none (e.g., single-frame clip)
    if not valid_frames:
        valid_frames = [(int(o), float(ts)) for o, ts in frame_records if ts is not None]

    if not valid_frames:
        return {
            "num_frames": num_frames,
            "clip_start_sec": float(clip_start_sec),
            "clip_end_sec": float(clip_end_sec),
            "target_timestamps_sec": [],
            "selected_ordinals": [],
            "selected_timestamps_sec": [],
            "unique_frames": 0,
            "missing_mapping": missing_mapping if missing_mapping > 0 else 1,
            "status": "missing_mapping",
            "recipe_hash": None,
        }

    # Sort by ordinal to guarantee deterministic tie-breaking (smaller ordinal on tie)
    valid_frames.sort(key=lambda item: item[0])

    duration = clip_end_sec - clip_start_sec
    target_timestamps: list[float] = []
    selected_ordinals: list[int] = []
    selected_timestamps: list[float] = []

    for i in range(num_frames):
        target_t = clip_start_sec + (i + 0.5) * duration / num_frames
        best_ord, best_ts = min(
            valid_frames,
            key=lambda item: (round(abs(item[1] - target_t), 12), item[0]),
        )
        target_timestamps.append(round(target_t, 6))
        selected_ordinals.append(best_ord)
        selected_timestamps.append(round(best_ts, 6))

    recipe_payload = {
        "formula": "start + (i + 0.5) * (end - start) / num_frames",
        "num_frames": num_frames,
        "clip_start_sec": round(float(clip_start_sec), 6),
        "clip_end_sec": round(float(clip_end_sec), 6),
        "selected_ordinals": selected_ordinals,
        "selected_timestamps_sec": selected_timestamps,
    }
    return {
        "num_frames": num_frames,
        "clip_start_sec": float(clip_start_sec),
        "clip_end_sec": float(clip_end_sec),
        "target_timestamps_sec": target_timestamps,
        "selected_ordinals": selected_ordinals,
        "selected_timestamps_sec": selected_timestamps,
        "unique_frames": len(set(selected_ordinals)),
        "missing_mapping": missing_mapping,
        "status": "ok",
        "recipe_hash": sha256_json(recipe_payload),
    }


def compute_video_recipe_from_ledger(
    ledger_path: Path,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
    num_frames: int = NUM_CLIP_FRAMES,
) -> dict[str, Any]:
    """Compute 16-frame uniform bin-center recipe from a Parquet timeline ledger."""
    table = pq.read_table(
        str(ledger_path),
        columns=["original_decoded_frame_index", "timestamp_seconds", "presentation_timestamp"],
    )
    ordinals = table.column("original_decoded_frame_index").to_pylist()
    ts_secs = table.column("timestamp_seconds").to_pylist()
    pts_list = table.column("presentation_timestamp").to_pylist()

    records: list[tuple[int, float | None]] = []
    valid_ts: list[float] = []
    for o, ts, pts in zip(ordinals, ts_secs, pts_list):
        if pts is None or ts is None:
            records.append((int(o), None))
        else:
            t = float(ts)
            records.append((int(o), t))
            valid_ts.append(t)

    if not valid_ts:
        return {
            "num_frames": num_frames,
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
            "target_timestamps_sec": [],
            "selected_ordinals": [],
            "selected_timestamps_sec": [],
            "unique_frames": 0,
            "missing_mapping": len(records) if records else 1,
            "status": "missing_mapping",
            "recipe_hash": None,
        }

    start_sec = float(clip_start_sec) if clip_start_sec is not None else min(valid_ts)
    end_sec = float(clip_end_sec) if clip_end_sec is not None else (max(valid_ts) + 1e-3)
    if end_sec <= start_sec:
        end_sec = start_sec + 1e-3

    return select_bin_center_frames(records, start_sec, end_sec, num_frames=num_frames)


def compute_video_recipe_from_media(
    media_path: Path,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
    num_frames: int = NUM_CLIP_FRAMES,
) -> tuple[dict[str, Any], dict[int, Any]]:
    """Decode presentation-order frames from media and compute 16-frame recipe."""
    records: list[tuple[int, float | None]] = []
    decoded_pixels: dict[int, Any] = {}
    valid_ts: list[float] = []

    for frame in decode_frames(media_path, include_pixels=True):
        ts = frame.timestamp_seconds
        records.append((frame.ordinal, ts))
        if ts is not None:
            valid_ts.append(ts)
        if frame.pixels is not None:
            decoded_pixels[frame.ordinal] = frame.pixels

    if not valid_ts:
        return {
            "num_frames": num_frames,
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
            "target_timestamps_sec": [],
            "selected_ordinals": [],
            "selected_timestamps_sec": [],
            "unique_frames": 0,
            "missing_mapping": len(records) if records else 1,
            "status": "missing_mapping",
            "recipe_hash": None,
        }, {}

    start_sec = float(clip_start_sec) if clip_start_sec is not None else min(valid_ts)
    end_sec = float(clip_end_sec) if clip_end_sec is not None else (max(valid_ts) + 1e-3)
    if end_sec <= start_sec:
        end_sec = start_sec + 1e-3

    recipe = select_bin_center_frames(records, start_sec, end_sec, num_frames=num_frames)
    needed = set(recipe["selected_ordinals"])
    selected_pixels = {ord_val: decoded_pixels[ord_val] for ord_val in needed if ord_val in decoded_pixels}
    return recipe, selected_pixels


def _load_video_registry(
    *,
    video_registry_path: Path | None = None,
    db_path: Path | None = None,
    media_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build video_id -> asset metadata lookup from JSON/JSONL registry or SQLite DB."""
    registry: dict[str, dict[str, Any]] = {}

    if video_registry_path is not None:
        if not video_registry_path.is_file():
            raise FileNotFoundError(f"Video registry not found: {video_registry_path}")
        if video_registry_path.suffix == ".jsonl":
            entries = _read_jsonl(video_registry_path)
        else:
            doc = json.loads(video_registry_path.read_text(encoding="utf-8"))
            entries = doc.get("records", doc) if isinstance(doc, dict) else doc
        for entry in entries:
            vid = str(entry.get("video_id") or "").strip()
            if vid:
                registry[vid] = dict(entry)

    if db_path is not None and db_path.is_file():
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT video_id, file_path, duration_ms FROM Videos"
            ).fetchall()
        for r in rows:
            vid = str(r["video_id"])
            if vid not in registry:
                duration_ms = r["duration_ms"]
                registry[vid] = {
                    "video_id": vid,
                    "file_path": str(r["file_path"]),
                    "clip_start_sec": 0.0,
                    "clip_end_sec": float(duration_ms) / 1000.0 if duration_ms else None,
                    "basis": "metadata_db",
                }

    if media_root is not None and media_root.is_dir():
        for entry in registry.values():
            raw_fp = entry.get("relative_path") or entry.get("file_path") or entry.get("media_uri")
            if raw_fp and not Path(str(raw_fp)).is_file():
                candidate = media_root / Path(str(raw_fp)).name
                if candidate.is_file():
                    entry["_resolved_media_path"] = str(candidate)

    return registry


def _resolve_media_path(entry: dict[str, Any], media_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if entry.get("_resolved_media_path"):
        candidates.append(Path(str(entry["_resolved_media_path"])))
    for key in ("media_path", "file_path", "relative_path", "media_uri"):
        val = entry.get(key)
        if val:
            p = Path(str(val))
            candidates.append(p)
            if media_root is not None:
                candidates.append(media_root / p)
                candidates.append(media_root / p.name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _resolve_ledger_path(video_id: str, media_sha256: str | None, ingest_dir: Path | None) -> Path | None:
    if ingest_dir is None or not ingest_dir.is_dir():
        return None
    patterns = []
    if media_sha256:
        patterns.append(f"timeline/*{media_sha256}*/*/frames.parquet")
    patterns.append(f"timeline/*{video_id}*/*/frames.parquet")
    for pat in patterns:
        matches = sorted(ingest_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def prepare_kaggle_bundle(
    *,
    queries_path: Path,
    pool_paths: Sequence[Path],
    output_dir: Path,
    video_registry_path: Path | None = None,
    db_path: Path | None = None,
    media_root: Path | None = None,
    ingest_dir: Path | None = None,
    copy_media: bool = False,
    compute_recipe: bool = False,
    export_frames: bool = False,
) -> dict[str, Any]:
    """Export inference-only Kaggle bundle (queries, candidates, media_manifest, optional recipe)."""
    queries = load_inference_queries(queries_path)
    query_ids = [q["query_id"] for q in queries]
    candidates, union_video_ids = load_and_merge_candidate_pools(query_ids, pool_paths)

    registry = _load_video_registry(
        video_registry_path=video_registry_path,
        db_path=db_path,
        media_root=media_root,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_out_dir = out_dir / "media"
    frames_out_dir = out_dir / "frames"

    media_manifest_rows: list[dict[str, Any]] = []
    recipe_rows: list[dict[str, Any]] = []
    missing_media_video_ids: list[str] = []
    decode_errors: list[dict[str, str]] = []

    for vid in union_video_ids:
        entry = registry.get(vid, {})
        media_path = _resolve_media_path(entry, media_root)
        media_sha = entry.get("media_sha256") or entry.get("sha256")
        if media_path is not None and not media_sha:
            media_sha = sha256_file(media_path)

        if media_path is None and not media_sha:
            missing_media_video_ids.append(vid)

        rel_path_str = str(entry.get("relative_path") or "")
        if not rel_path_str and media_path is not None:
            if media_root is not None:
                try:
                    rel_path_str = media_path.resolve().relative_to(media_root.resolve()).as_posix()
                except ValueError:
                    rel_path_str = media_path.name
            else:
                rel_path_str = media_path.name

        if copy_media and media_path is not None:
            media_out_dir.mkdir(parents=True, exist_ok=True)
            dest_media = media_out_dir / media_path.name
            if not dest_media.exists():
                shutil.copy2(media_path, dest_media)
            rel_path_str = f"media/{media_path.name}"

        timeline_rev = str(entry.get("timeline_revision") or "tl_v1")
        clip_start = float(entry.get("clip_start_sec", 0.0))
        clip_end_raw = entry.get("clip_end_sec")
        clip_end = float(clip_end_raw) if clip_end_raw is not None else None
        basis = str(entry.get("basis") or "full_clip")

        ledger_path = _resolve_ledger_path(vid, media_sha, ingest_dir)
        recipe: dict[str, Any] | None = None
        if compute_recipe or export_frames:
            try:
                if ledger_path is not None:
                    recipe = compute_video_recipe_from_ledger(
                        ledger_path, clip_start_sec=clip_start, clip_end_sec=clip_end
                    )
                    if export_frames and media_path is not None and recipe["selected_ordinals"]:
                        from PIL import Image

                        frames_out_dir.mkdir(parents=True, exist_ok=True)
                        for ord_val in sorted(set(recipe["selected_ordinals"])):
                            frame_img_path = frames_out_dir / f"{vid}__{ord_val:06d}.png"
                            if not frame_img_path.exists():
                                dec = exact_seek_frame(media_path, ledger_path, target_ordinal=ord_val)
                                if dec.pixels is not None:
                                    Image.fromarray(dec.pixels).save(frame_img_path, format="PNG")
                elif media_path is not None:
                    recipe, decoded_pixels = compute_video_recipe_from_media(
                        media_path, clip_start_sec=clip_start, clip_end_sec=clip_end
                    )
                    if export_frames and decoded_pixels:
                        from PIL import Image

                        frames_out_dir.mkdir(parents=True, exist_ok=True)
                        for ord_val, pix in decoded_pixels.items():
                            frame_img_path = frames_out_dir / f"{vid}__{ord_val:06d}.png"
                            if not frame_img_path.exists():
                                Image.fromarray(pix).save(frame_img_path, format="PNG")
                else:
                    recipe = {
                        "num_frames": NUM_CLIP_FRAMES,
                        "clip_start_sec": clip_start,
                        "clip_end_sec": clip_end,
                        "selected_ordinals": [],
                        "selected_timestamps_sec": [],
                        "unique_frames": 0,
                        "missing_mapping": 1,
                        "status": "missing_media",
                        "recipe_hash": None,
                    }
            except Exception as exc:
                decode_errors.append({"video_id": vid, "error": str(exc)})
                recipe = {
                    "num_frames": NUM_CLIP_FRAMES,
                    "clip_start_sec": clip_start,
                    "clip_end_sec": clip_end,
                    "selected_ordinals": [],
                    "selected_timestamps_sec": [],
                    "unique_frames": 0,
                    "missing_mapping": 1,
                    "status": "decode_error",
                    "recipe_hash": None,
                }

            if clip_end is None and recipe.get("clip_end_sec") is not None:
                clip_end = float(recipe["clip_end_sec"])

            recipe_rows.append({
                "video_id": vid,
                "media_sha256": media_sha,
                "timeline_revision": timeline_rev,
                **recipe,
            })

        media_manifest_rows.append({
            "video_id": vid,
            "relative_path": rel_path_str,
            "media_sha256": media_sha,
            "timeline_revision": timeline_rev,
            "clip_start_sec": clip_start,
            "clip_end_sec": clip_end,
            "basis": basis,
        })

    assert_no_gt_leakage(media_manifest_rows, context="media_manifest")

    queries_out = out_dir / "queries.jsonl"
    candidates_out = out_dir / "candidates.jsonl"
    media_manifest_out = out_dir / "media_manifest.jsonl"

    _write_jsonl(queries_out, queries)
    _write_jsonl(candidates_out, candidates)
    _write_jsonl(media_manifest_out, media_manifest_rows)

    files_sha256 = {
        "queries.jsonl": sha256_file(queries_out),
        "candidates.jsonl": sha256_file(candidates_out),
        "media_manifest.jsonl": sha256_file(media_manifest_out),
    }

    if recipe_rows:
        recipe_out = out_dir / "frame_recipe.jsonl"
        _write_jsonl(recipe_out, recipe_rows)
        files_sha256["frame_recipe.jsonl"] = sha256_file(recipe_out)

    bundle_manifest = {
        "bundle_format": "videntra_clip_vp_kaggle_v1",
        "inference_only": True,
        "num_queries": len(queries),
        "num_candidate_videos": len(union_video_ids),
        "missing_media_video_ids": missing_media_video_ids,
        "decode_errors": decode_errors,
        "files_sha256": files_sha256,
    }
    (out_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return bundle_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare inference-safe Kaggle bundle for VideoPrism-LvT-B."
    )
    parser.add_argument("--pools-dir", type=Path, default=None, help="Directory with queries.jsonl, a5_pool.jsonl, clip_pool.jsonl")
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--pool", type=Path, action="append", default=[], help="Candidate pool JSONL (repeatable)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-registry", type=Path, default=None, help="JSON/JSONL video metadata registry")
    parser.add_argument("--db-path", type=Path, default=None, help="Optional SQLite metadata.db")
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--ingest-dir", type=Path, default=None)
    parser.add_argument("--copy-media", action="store_true")
    parser.add_argument("--compute-recipe", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    queries_path = args.queries or (args.pools_dir / "queries.jsonl" if args.pools_dir else None)
    if queries_path is None:
        raise ValueError("Provide --queries or --pools-dir")

    pool_paths = list(args.pool)
    if not pool_paths and args.pools_dir is not None:
        for name in ("a5_pool.jsonl", "clip_pool.jsonl", "candidates.jsonl"):
            candidate = args.pools_dir / name
            if candidate.is_file():
                pool_paths.append(candidate)

    summary = prepare_kaggle_bundle(
        queries_path=queries_path,
        pool_paths=pool_paths,
        output_dir=args.out_dir,
        video_registry_path=args.video_registry,
        db_path=args.db_path,
        media_root=args.media_root,
        ingest_dir=args.ingest_dir,
        copy_media=args.copy_media,
        compute_recipe=args.compute_recipe,
        export_frames=args.export_frames,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
