import hashlib
import io
import json

import pytest

from scripts import acquire_blender_pilot
from scripts.profile_text_encoder import synthetic_queries


def test_profile_queries_are_label_free_and_unique():
    queries = synthetic_queries()
    assert len(queries) == len(set(queries)) == 100


def test_blender_download_checks_hash_and_preserves_existing(tmp_path, monkeypatch):
    content = b"fixture"
    asset = {"video_id": "fixture", "name": "fixture.mov", "bytes": len(content),
             "sha256": hashlib.sha256(content).hexdigest(), "license": "CC-BY-3.0",
             "rights_source": "https://example.org", "attribution": "example"}
    monkeypatch.setattr(acquire_blender_pilot, "ASSETS", [asset])
    monkeypatch.setattr(acquire_blender_pilot, "urlopen", lambda *a, **kw: io.BytesIO(content))
    out = tmp_path / "pilot"
    acquire_blender_pilot.acquire(out)
    assert (out / "fixture.mov").read_bytes() == content
    assert json.loads((out / "source_receipts.jsonl").read_text())["labels_status"] == "not_collected"
    with pytest.raises(FileExistsError):
        acquire_blender_pilot.acquire(out)
