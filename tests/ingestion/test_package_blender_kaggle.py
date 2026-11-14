import hashlib
import json

import pytest

from scripts.package_blender_kaggle import package
from scripts.kaggle_multimodal_offline_worker import load_bundle_videos
from videntra.evaluation.corpus_gate import freeze_inventory


def make_fixture(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    movie = media / "movie.mov"
    movie.write_bytes(b"synthetic")
    digest = hashlib.sha256(movie.read_bytes()).hexdigest()
    receipt = media / "source_receipts.jsonl"
    receipt.write_text(json.dumps({"video_id": "v1", "name": movie.name, "bytes": movie.stat().st_size,
                                   "sha256": digest, "license": "CC-BY-3.0", "attribution": "Author"}) + "\n")
    (media / "ATTRIBUTION.md").write_text("Author")
    worker = tmp_path / "worker.py"
    worker.write_text("# pinned worker fixture")
    inventory = tmp_path / "inventory.json"
    freeze_inventory([{"video_id": "v1", "sha256": digest, "rights_status": "allowed",
                       "rights_receipt_path": str(receipt.resolve()),
                       "rights_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                       "source_group": "v1", "duplicate_group": "v1"}], inventory, snapshot_id="fixture")
    return inventory, media, worker


def test_review_bundle_excludes_media_and_rejects_overwrite(tmp_path):
    inventory, media, worker = make_fixture(tmp_path)
    output = tmp_path / "out"
    manifest = package(inventory, media, worker, output)
    assert manifest["media_included"] is False
    assert not (output / "videos").exists()
    assert not any("query" in path.name or "gold" in path.name for path in output.iterdir())
    with pytest.raises(FileExistsError):
        package(inventory, media, worker, output)


def test_bundle_rejects_changed_media_before_creating_output(tmp_path):
    inventory, media, worker = make_fixture(tmp_path)
    (media / "movie.mov").write_bytes(b"tampered")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="checksum mismatch"):
        package(inventory, media, worker, output, include_media=True)
    assert not output.exists()


def test_bundle_worker_reads_only_verified_listed_mov(tmp_path):
    inventory, media, worker = make_fixture(tmp_path)
    output = tmp_path / "out"
    package(inventory, media, worker, output, include_media=True)
    (output / "videos" / "unlisted.mov").write_bytes(b"do not process")
    assert load_bundle_videos(output / "bundle_manifest.json") == [str(output / "videos" / "movie.mov")]
    (output / "videos" / "movie.mov").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size|checksum"):
        load_bundle_videos(output / "bundle_manifest.json")


def test_bundle_worker_accepts_flat_kaggle_layout_but_rejects_ambiguous_copy(tmp_path):
    inventory, media, worker = make_fixture(tmp_path)
    output = tmp_path / "out"
    package(inventory, media, worker, output, include_media=True)
    nested = output / "videos" / "movie.mov"
    flat = output / "movie.mov"
    nested.rename(flat)
    assert load_bundle_videos(output / "bundle_manifest.json") == [str(flat)]
    nested.write_bytes(flat.read_bytes())
    with pytest.raises(ValueError, match="ambiguous"):
        load_bundle_videos(output / "bundle_manifest.json")
