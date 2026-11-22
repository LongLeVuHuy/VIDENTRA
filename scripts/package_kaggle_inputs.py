"""Package keyframes and benchmark queries for Kaggle GPU embedding & caption inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence

from PIL import Image
import pyarrow.parquet as pq

# Ensure src/ is importable when invoked as `python scripts/package_kaggle_inputs.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from videntra.timeline.ledger import exact_seek_frame  # noqa: E402


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _save_resized_keyframe(pixels: Any, img_path: Path, size: tuple[int, int] = (224, 224)) -> None:
    pil_img = Image.fromarray(pixels)
    if size is not None:
        pil_img = pil_img.resize(size, Image.Resampling.BILINEAR)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(img_path, format="JPEG", quality=90)


def _decode_frames_by_ordinal(media_path: Path, target_ordinals: set[int]) -> dict[int, Any]:
    """Decode zero-based presentation-order frames without timestamp*FPS estimation."""
    import av

    decoded: dict[int, Any] = {}
    if not target_ordinals or not media_path.is_file():
        return decoded
    max_ord = max(target_ordinals)
    with av.open(str(media_path)) as container:
        for f_idx, frame in enumerate(container.decode(video=0)):
            if f_idx in target_ordinals:
                decoded[f_idx] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(target_ordinals):
                    break
            if f_idx >= max_ord:
                break
    return decoded


def _infer_dataset_bucket(video_id: str) -> str:
    """Infer dataset bucket (msrvtt, avc, qvhighlights) from video_id."""
    if video_id.startswith("video"):
        return "msrvtt"
    if "__s" in video_id:
        return "avc"
    return "qvhighlights"


def _select_stratified_pilot_rows(rows: Sequence[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    """Select balanced keyframes across datasets and distinct videos up to limit."""
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    by_ds_vid: dict[str, dict[str, list[sqlite3.Row]]] = {}
    for row in rows:
        vid = str(row["video_id"])
        ds = _infer_dataset_bucket(vid)
        by_ds_vid.setdefault(ds, {}).setdefault(vid, []).append(row)

    ds_queues: dict[str, list[list[sqlite3.Row]]] = {}
    for ds, vid_map in by_ds_vid.items():
        # Deterministic video order sorted by earliest faiss_id
        v_lists = sorted(vid_map.values(), key=lambda items: int(items[0]["faiss_id"]))
        ds_queues[ds] = v_lists

    ds_names = sorted(ds_queues.keys())
    selected: list[sqlite3.Row] = []
    pos_by_ds: dict[str, int] = {ds: 0 for ds in ds_names}
    while len(selected) < limit:
        progressed = False
        for ds in ds_names:
            if len(selected) >= limit:
                break
            v_lists = ds_queues[ds]
            if not v_lists:
                continue
            # Scan round-robin within dataset for next remaining keyframe
            checked = 0
            while checked < len(v_lists):
                idx = pos_by_ds[ds] % len(v_lists)
                pos_by_ds[ds] = (pos_by_ds[ds] + 1) % len(v_lists)
                checked += 1
                if v_lists[idx]:
                    selected.append(v_lists[idx].pop(0))
                    progressed = True
                    break
        if not progressed:
            break
    selected.sort(key=lambda r: int(r["faiss_id"]))
    return selected


def package_db_keyframes(
    db_path: Path,
    output_dir: str | Path = "work/kaggle/p85_inputs",
    media_root: str | Path | None = None,
    resize_to: tuple[int, int] = (224, 224),
    limit: int | None = None,
    dry_run: bool = False,
    stratified_pilot: bool = False,
) -> dict[str, Any]:
    """Package keyframes from an existing metadata.db (ordered by faiss_id) for Kaggle P8-5."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Metadata DB not found: {db_path}")

    out_p = Path(output_dir)
    images_p = out_p / "images"
    media_root_p = Path(media_root) if media_root is not None else None

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT k.faiss_id, k.keyframe_id, k.video_id, k.source_frame_index,
                   k.timestamp_ms, v.file_path
            FROM Keyframes k
            JOIN Videos v ON k.video_id = v.video_id
            ORDER BY k.faiss_id ASC
            """
        ).fetchall()

    if limit is not None and limit > 0:
        rows = (
            _select_stratified_pilot_rows(rows, int(limit))
            if stratified_pilot
            else rows[:limit]
        )

    if dry_run:
        return {
            "mode": "db",
            "db_path": str(db_path),
            "output_dir": str(out_p),
            "num_keyframes": len(rows),
            "stratified_pilot": bool(stratified_pilot),
            "dry_run": True,
        }

    images_p.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_video.setdefault(str(r["video_id"]), []).append(r)

    keyframe_records: list[dict[str, Any]] = []
    for vid, v_rows in by_video.items():
        raw_fp = Path(str(v_rows[0]["file_path"]))
        candidates = [raw_fp]
        if media_root_p is not None:
            candidates.insert(0, media_root_p / raw_fp.name)
        media_path = next((c for c in candidates if c.is_file()), None)

        needed_ordinals: set[int] = set()
        for r in v_rows:
            fid = int(r["faiss_id"])
            img_path = images_p / f"{r['keyframe_id']}.jpg"
            if not img_path.exists():
                needed_ordinals.add(int(r["source_frame_index"]))

        decoded_map = _decode_frames_by_ordinal(media_path, needed_ordinals) if (media_path and needed_ordinals) else {}

        for r in v_rows:
            kf_id = str(r["keyframe_id"])
            img_filename = f"{kf_id}.jpg"
            img_path = images_p / img_filename
            ord_val = int(r["source_frame_index"])
            if not img_path.exists() and ord_val in decoded_map:
                _save_resized_keyframe(decoded_map[ord_val], img_path, size=resize_to)
            if img_path.exists():
                ts_ms = int(r["timestamp_ms"])
                keyframe_records.append({
                    "faiss_id": int(r["faiss_id"]),
                    "keyframe_id": kf_id,
                    "video_id": vid,
                    "ordinal": ord_val,
                    "timestamp_ms": ts_ms,
                    "timestamp_seconds": round(ts_ms / 1000.0, 4),
                    "image_file": img_filename,
                    "image_sha256": sha256_file(img_path),
                })

    keyframe_records.sort(key=lambda r: int(r["faiss_id"]))
    kf_manifest_path = out_p / "keyframes_manifest.jsonl"
    with kf_manifest_path.open("w", encoding="utf-8") as f:
        for r in keyframe_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    package_summary = {
        "mode": "db",
        "num_keyframes": len(keyframe_records),
        "stratified_pilot": bool(stratified_pilot),
        "keyframes_manifest_sha256": sha256_file(kf_manifest_path),
    }
    (out_p / "package_manifest.json").write_text(
        json.dumps(package_summary, indent=2), encoding="utf-8"
    )
    print(f"Packaged {len(keyframe_records)} DB keyframes in {out_p}")
    return package_summary


def package_inputs(
    manifest_path: str = "manifests/benchmark/smoke/benchmark_smoke.media.json",
    ingest_dir: str = "work/benchmark/smoke/ingest",
    media_root: str | None = None,
    output_dir: str = "work/kaggle/inputs",
    max_samples_per_video: int = 100,
    gt_output_dir: str = "work/benchmark/smoke",
    skip_gt: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if media_root is None:
        media_root = os.environ.get("VIDENTRA_QVH_MEDIA_DIR", "data/raw/qvhighlights/media")
    out_p = Path(output_dir)
    images_p = out_p / "images"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    ingest_p = Path(ingest_dir)
    media_root_p = Path(media_root)

    if dry_run:
        records = manifest_doc.get("records", [])
        return {
            "mode": "manifest",
            "manifest_path": str(manifest_path),
            "num_videos": len(records),
            "output_dir": str(out_p),
            "dry_run": True,
        }

    images_p.mkdir(parents=True, exist_ok=True)
    keyframe_records = []
    print("Extracting keyframes for Kaggle packaging...")

    for rec in manifest_doc["records"]:
        vid = rec["video_id"]
        sha = rec["sha256"]
        media_name = Path(rec["media_uri"]).name
        media_path = media_root_p / media_name
        if not media_path.exists():
            continue

        tl_matches = list(ingest_p.glob(f"timeline/*{sha}*/*/frames.parquet"))
        samples_matches = list(ingest_p.glob(f"samples/*{sha}*/*/samples.parquet"))
        if not tl_matches or not samples_matches:
            continue

        tl_path = tl_matches[0]
        samples_table = pq.read_table(str(samples_matches[0]))

        ords = samples_table.column("original_decoded_frame_index").to_pylist()
        s_ids = samples_table.column("sampled_frame_id").to_pylist()
        ts_secs = samples_table.column("timestamp_seconds").to_pylist()

        # Sample frames up to max_samples_per_video with uniform stride
        stride = max(1, len(ords) // max(1, max_samples_per_video))
        selected_indices = list(range(0, len(ords), stride))[:max_samples_per_video]

        for idx in selected_indices:
            ord_val = ords[idx]
            s_id = s_ids[idx]
            t_sec = ts_secs[idx]

            img_filename = f"{s_id}.jpg"
            img_path = images_p / img_filename

            if not img_path.exists():
                try:
                    frame = exact_seek_frame(media_path, tl_path, target_ordinal=ord_val)
                    if frame.pixels is not None:
                        _save_resized_keyframe(frame.pixels, img_path, size=(224, 224))
                except Exception:
                    continue

            if img_path.exists():
                t_float = float(t_sec) if t_sec is not None else 0.0
                keyframe_records.append({
                    "faiss_id": len(keyframe_records),
                    "keyframe_id": s_id,
                    "sampled_frame_id": s_id,
                    "video_id": vid,
                    "source_video_name": media_name,
                    "ordinal": ord_val,
                    "timestamp_ms": int(round(t_float * 1000.0)),
                    "timestamp_seconds": round(t_float, 4),
                    "image_file": img_filename,
                })

    # Save keyframes manifest
    kf_manifest_path = out_p / "keyframes_manifest.jsonl"
    with open(kf_manifest_path, "w", encoding="utf-8") as f:
        for r in keyframe_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Define stratified benchmark queries
    benchmark_queries = [
        # Visual-dominant
        {"qid": "Q_VIS_01", "query": "A person makes a gift package while sitting at a table", "modality_focus": "visual"},
        {"qid": "Q_VIS_02", "query": "Asian woman holds up beauty products to the camera", "modality_focus": "visual"},
        {"qid": "Q_VIS_03", "query": "A group of men is walking in a deserted path", "modality_focus": "visual"},
        {"qid": "Q_VIS_04", "query": "Mike Pence is sitting in front of a blue background", "modality_focus": "visual"},
        # OCR-dominant
        {"qid": "Q_OCR_01", "query": "VIDENTRA 2026", "modality_focus": "ocr"},
        # ASR-dominant
        {"qid": "Q_ASR_01", "query": "thông lệ không một người đàn ông", "modality_focus": "asr"},
        {"qid": "Q_ASR_02", "query": "nguyên liệu nhảy khiến chị thật thú", "modality_focus": "asr"},
        # Multimodal
        {"qid": "Q_MM_01", "query": "Asian woman holds up beauty products and talks about ingredients", "modality_focus": "multimodal"},
        {"qid": "Q_MM_02", "query": "A person makes a gift package with sign VIDENTRA 2026", "modality_focus": "multimodal"},
    ]

    queries_path = out_p / "benchmark_queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in benchmark_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Save manifest with checksums
    package_summary = {
        "mode": "manifest",
        "num_keyframes": len(keyframe_records),
        "num_queries": len(benchmark_queries),
        "keyframes_manifest_sha256": sha256_file(kf_manifest_path),
        "queries_sha256": sha256_file(queries_path),
    }

    with open(out_p / "package_manifest.json", "w", encoding="utf-8") as f:
        json.dump(package_summary, f, indent=2)

    if not skip_gt:
        # Separately create Ground Truth for Evaluator (strictly isolated from Kaggle inputs)
        gt_dir = Path(gt_output_dir)
        gt_dir.mkdir(parents=True, exist_ok=True)
        ground_truth = [
            {
                "qid": "Q_VIS_01",
                "relevant_videos": [{"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[96.0, 126.0]]}],
            },
            {
                "qid": "Q_VIS_02",
                "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 22.0]]}],
            },
            {
                "qid": "Q_VIS_03",
                "relevant_videos": [{"video_id": "sha256:724761572dc8714aa2b8083239469cfa6b83da3988e8903c2e13871f94f0d766", "relevant_windows": [[48.0, 68.0]]}],
            },
            {
                "qid": "Q_VIS_04",
                "relevant_videos": [{"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[6.0, 14.0], [16.0, 24.0], [36.0, 48.0], [72.0, 82.0], [96.0, 100.0], [108.0, 112.0], [118.0, 124.0], [136.0, 144.0]]}],
            },
            {
                "qid": "Q_OCR_01",
                "relevant_videos": [
                    {"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[0.0, 150.0]]},
                    {"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 150.0]]},
                    {"video_id": "sha256:724761572dc8714aa2b8083239469cfa6b83da3988e8903c2e13871f94f0d766", "relevant_windows": [[0.0, 150.0]]},
                    {"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[0.0, 150.0]]},
                ],
            },
            {
                "qid": "Q_ASR_01",
                "relevant_videos": [{"video_id": "sha256:0985cda8436a32a31e6ec063a58dc5f30daa937dd8a18bb4798d3347523780e7", "relevant_windows": [[0.0, 30.0]]}],
            },
            {
                "qid": "Q_ASR_02",
                "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 30.0]]}],
            },
            {
                "qid": "Q_MM_01",
                "relevant_videos": [{"video_id": "sha256:ac7b8b63bf2a4f80d77b9f850f6da179a31c6fc610d5d822021d6d83c0390c73", "relevant_windows": [[0.0, 22.0]]}],
            },
            {
                "qid": "Q_MM_02",
                "relevant_videos": [{"video_id": "sha256:c12895f168c00faf3fa7bab1c3a69a62e3ce3a1f5e735baee9f003928489b93a", "relevant_windows": [[96.0, 126.0]]}],
            },
        ]
        gt_file = gt_dir / "ground_truth.jsonl"
        with open(gt_file, "w", encoding="utf-8") as f:
            for item in ground_truth:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Ground truth isolated and saved to {gt_file}")

    print(f"Packaged {len(keyframe_records)} keyframes and {len(benchmark_queries)} queries in {out_p}")
    return package_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package keyframes and queries for Kaggle GPU inference.")
    parser.add_argument("--mode", choices=("manifest", "db"), default="manifest")
    parser.add_argument("--manifest-path", type=str, default="manifests/benchmark/smoke/benchmark_smoke.media.json")
    parser.add_argument("--ingest-dir", type=str, default="work/benchmark/smoke/ingest")
    parser.add_argument("--media-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="work/kaggle/inputs")
    parser.add_argument("--max-samples-per-video", type=int, default=100)
    parser.add_argument("--gt-output-dir", type=str, default="work/benchmark/smoke")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/catalog/corpus_metadata.db"))
    parser.add_argument("--limit", type=int, default=None, help="Optional keyframe limit for --mode db.")
    parser.add_argument("--stratified-pilot", action="store_true", help="Sample balanced keyframes across datasets/videos when --limit is set.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "db":
        summary = package_db_keyframes(
            db_path=args.db_path,
            output_dir=args.output_dir,
            media_root=args.media_root,
            limit=args.limit,
            dry_run=args.dry_run,
            stratified_pilot=args.stratified_pilot,
        )
    else:
        summary = package_inputs(
            manifest_path=args.manifest_path,
            ingest_dir=args.ingest_dir,
            media_root=args.media_root,
            output_dir=args.output_dir,
            max_samples_per_video=args.max_samples_per_video,
            gt_output_dir=args.gt_output_dir,
            skip_gt=args.skip_gt,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
