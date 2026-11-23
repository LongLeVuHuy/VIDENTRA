import json
import subprocess
import sys

from videntra.cli import validate_manifests


def _write_synthetic_msrvtt_manifests(manifest_dir, media):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    digest_1 = "a" * 64
    digest_2 = "b" * 64
    rec_1 = {
        **media,
        "video_id": f"sha256:{digest_1}",
        "source_video_id": "msrvtt:clip:video7010",
        "sha256": digest_1,
        "media_uri": "asset://msrvtt/video7010.mp4",
        "official_split": "test",
        "exposure": "sealed_evaluation",
        "rights_status": "allowed",
        "rights_reference": "receipt://msrvtt-academic",
    }
    rec_2 = {
        **media,
        "video_id": f"sha256:{digest_2}",
        "source_video_id": "msrvtt:clip:video7011",
        "sha256": digest_2,
        "media_uri": "asset://msrvtt/video7011.mp4",
        "official_split": "test",
        "exposure": "sealed_evaluation",
        "rights_status": "allowed",
        "rights_reference": "receipt://msrvtt-academic",
    }
    media_doc = {
        "kind": "media_asset",
        "revision": "msrvtt-synthetic-v1",
        "records": [rec_1, rec_2],
    }
    dataset_doc = {
        "kind": "dataset",
        "revision": "msrvtt-synthetic-v1",
        "records": [
            {
                "dataset_id": "msrvtt_test_1ka_sample",
                "source_url": "https://huggingface.co/datasets/friedrichor/MSR-VTT",
                "source_revision": "pinned-msrvtt-1ka",
                "official_split": "test",
                "selection_method": "msrvtt_1k_a_partition",
                "annotation_license": "msr-vtt-academic",
                "media_license_status": "public_academic",
                "media_ids": [rec_1["video_id"], rec_2["video_id"]],
                "exposure": "sealed_evaluation",
                "rights_status": "allowed",
                "rights_reference": "receipt://msrvtt-academic",
                "annotation_access": "evaluation_only",
                "parent_manifest_sha256": "0" * 64,
            }
        ],
    }
    (manifest_dir / "01_media.json").write_text(json.dumps(media_doc), encoding="utf-8")
    (manifest_dir / "02_dataset.json").write_text(json.dumps(dataset_doc), encoding="utf-8")


def test_synthetic_msrvtt_manifest_validation(tmp_path, media):
    manifest_dir = tmp_path / "manifests"
    _write_synthetic_msrvtt_manifests(manifest_dir, media)
    docs = validate_manifests(manifest_dir)
    media_records = next(d for d in docs if d["kind"] == "media_asset")["records"]
    dataset_records = next(d for d in docs if d["kind"] == "dataset")["records"]
    assert len(media_records) == 2
    assert len(dataset_records) == 1
    assert set(dataset_records[0]["media_ids"]) == {r["video_id"] for r in media_records}
    assert all(r["rights_status"] == "allowed" for r in media_records)


def test_cli_from_other_working_directory(tmp_path, media):
    manifest_dir = tmp_path / "manifests"
    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    _write_synthetic_msrvtt_manifests(manifest_dir, media)
    completed = subprocess.run(
        [sys.executable, "-m", "videntra", "manifests", "validate", "--all", "--directory", str(manifest_dir)],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["records"] == 3


def test_cli_failure_is_not_empty_success(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "videntra", "manifests", "validate", "--all", "--directory", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "error" in json.loads(completed.stderr)

