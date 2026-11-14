import hashlib
import json
import pytest

from scripts.acquire_qvh_quarantine import acquire
from videntra.evaluation.split_lock import freeze_split


def test_verified_download_is_quarantined_and_not_overwritten(tmp_path, monkeypatch):
    data = b"synthetic-video-bytes"
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"qid": 1, "vid": "abcdefghijk_0.0_10.0"}) + "\n", encoding="utf-8")
    lock = tmp_path / "split.json"
    freeze_split([{"query_id": "qvh_1", "split": "test", "source_group": "abcdefghijk"}], lock,
                 metadata={"source_files_sha256": {"test": hashlib.sha256(annotations.read_bytes()).hexdigest()}})

    import io
    monkeypatch.setattr("scripts.acquire_qvh_quarantine.metadata", lambda url: (hashlib.sha256(data).hexdigest(), len(data)))
    monkeypatch.setattr("scripts.acquire_qvh_quarantine.urlopen", lambda *a, **kw: io.BytesIO(data))
    output = tmp_path / "quarantine"
    acquire(lock, annotations, output)
    assert (output / "abcdefghijk_0.0_10.0.mp4").read_bytes() == data
    receipt = json.loads((output / "source_receipts.jsonl").read_text())
    assert receipt["rights_status"] == "unknown"
    with pytest.raises(FileExistsError):
        acquire(lock, annotations, output)
    annotations.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen split"):
        acquire(lock, annotations, tmp_path / "other")


def test_checksum_failure_never_creates_final_video(tmp_path, monkeypatch):
    import io
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"qid": 1, "vid": "abcdefghijk_0.0_10.0"}) + "\n", encoding="utf-8")
    lock = tmp_path / "split.json"
    freeze_split([{"query_id": "qvh_1", "split": "test", "source_group": "abcdefghijk"}], lock,
                 metadata={"source_files_sha256": {"test": hashlib.sha256(annotations.read_bytes()).hexdigest()}})
    monkeypatch.setattr("scripts.acquire_qvh_quarantine.metadata", lambda url: ("0" * 64, 3))
    monkeypatch.setattr("scripts.acquire_qvh_quarantine.urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    output = tmp_path / "quarantine"
    with pytest.raises(ValueError, match="Checksum"):
        acquire(lock, annotations, output)
    assert not list(output.glob("*.mp4"))
    assert not list(output.glob("*.partial"))
