"""Create a new, label-free private Kaggle input bundle; never upload it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from videntra.evaluation.corpus_gate import verify_inventory
from videntra.evaluation.provenance import sha256_file


def package(inventory_path: Path, media_dir: Path, worker_path: Path, output: Path, *, include_media: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    if not verify_inventory(inventory_path):
        raise ValueError("Invalid or modified corpus inventory")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    receipt = media_dir / "source_receipts.jsonl"
    attribution = media_dir / "ATTRIBUTION.md"
    if not receipt.is_file() or not attribution.is_file() or not worker_path.is_file():
        raise FileNotFoundError("Missing receipt, attribution or worker")
    source_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["video_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("Duplicate source receipt")
    receipt_hash = sha256_file(receipt)
    assets = []
    for entry in inventory["entries"]:
        source = by_id.get(entry["video_id"])
        if (entry["rights_status"] != "allowed" or entry["eligibility"] != "eligible_pending_independent_review"
                or entry["rights_receipt_sha256"] != receipt_hash or source is None):
            raise ValueError("Asset rights or source receipt requires review")
        filename = source.get("name", "")
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Invalid media filename")
        file = media_dir / filename
        if (not file.is_file() or file.stat().st_size != source["bytes"] or
                sha256_file(file) != entry["sha256"] or source["sha256"] != entry["sha256"]):
            raise ValueError(f"Media checksum mismatch: {entry['video_id']}")
        assets.append((file, source))
    if not assets:
        raise ValueError("No eligible assets")

    output.mkdir(parents=True, exist_ok=False)
    if include_media:
        media_out = output / "videos"
        media_out.mkdir()
        for path, source in assets:
            target = media_out / path.name
            with path.open("rb") as inp, target.open("xb") as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
            if sha256_file(target) != source["sha256"]:
                raise RuntimeError("Copied media checksum mismatch; bundle is incomplete")
    for src in (receipt, attribution, inventory_path, worker_path):
        with src.open("rb") as inp, (output / src.name).open("xb") as out:
            shutil.copyfileobj(inp, out, 1024 * 1024)
    manifest = {"status": "PREPARED_NOT_UPLOADED", "media_included": include_media,
                "inventory_sha256": sha256_file(inventory_path),
                "source_receipts_sha256": receipt_hash, "worker_sha256": sha256_file(worker_path),
                "assets": [{key: row[key] for key in ("video_id", "name", "bytes", "sha256", "license", "attribution")}
                           for _, row in assets],
                "limitations": ["No query labels or ground truth are included.",
                    "Private upload and GPU run require review; rights receipt hash is not a legal decision.",
                    "Worker's OCR, ASR, object models and runtime are not pinned; do not claim full provenance.",
                    "Source grouping, timeline origin and human labels remain unverified."]}
    with (output / "bundle_manifest.json").open("x", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-media", action="store_true", help="Copy licensed videos into a new bundle")
    args = parser.parse_args()
    result = package(args.inventory, args.media_dir, args.worker, args.output, include_media=args.include_media)
    print(json.dumps({"status": result["status"], "assets": len(result["assets"]), "output": str(args.output)}))
