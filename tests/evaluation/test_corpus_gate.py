import hashlib
import json

import pytest

from videntra.evaluation.corpus_gate import evaluate_coverage, freeze_inventory, verify_inventory


def _record(video_id, sha, rights="unknown", receipt=None):
    return {"video_id": video_id, "sha256": sha, "rights_status": rights,
            "rights_receipt_path": str(receipt) if receipt else None,
            "rights_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt else None,
            "source_group": "source-" + video_id, "duplicate_group": "group-" + video_id}


def test_freeze_inventory_blocks_unknown_and_checks_receipts(tmp_path):
    receipt = tmp_path / "receipt.txt"
    receipt.write_text("review record", encoding="utf-8")
    blocked = _record("v1", "1" * 64)
    eligible = _record("v2", "2" * 64, "allowed", receipt)
    invalid = _record("v3", "3" * 64, "allowed", receipt)
    invalid["rights_receipt_sha256"] = "0" * 64
    path = tmp_path / "corpus.json"
    manifest = freeze_inventory([invalid, eligible, blocked], path, snapshot_id="s1")
    entries = {entry["video_id"]: entry for entry in manifest["entries"]}
    assert entries["v1"]["eligibility"] == "blocked"
    assert entries["v2"]["eligibility"] == "eligible_pending_independent_review"
    assert entries["v3"]["reasons"] == ["receipt_unverified"]
    assert manifest["independence_status"] == "unverified"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert verify_inventory(path)
    with pytest.raises(FileExistsError):
        freeze_inventory([blocked], path, snapshot_id="s1")
    manifest["entries"][0]["rights_status"] = "allowed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not verify_inventory(path)


def test_inventory_rejects_gold_and_duplicate_content(tmp_path):
    row = _record("v1", "1" * 64)
    with pytest.raises(ValueError, match="label-free"):
        freeze_inventory([{**row, "target_videos": ["v1"]}], tmp_path / "bad.json", snapshot_id="s")
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_inventory([row, _record("v2", "1" * 64)], tmp_path / "bad.json", snapshot_id="s")


def test_coverage_counts_missing_without_filtering_queries():
    result = evaluate_coverage([
        {"query_id": "q1", "dataset": "qvh", "target_videos": ["v1"]},
        {"query_id": "q2", "dataset": "qvh", "target_videos": ["v2", "v3"]},
        {"query_id": "q3", "dataset": "qvh", "target_videos": ["v4"]},
        {"query_id": "q4", "dataset": "qvh", "target_videos": []},
        {"query_id": "q5", "dataset": "avc", "target_videos": None},
    ], {"v1", "v2"})
    assert result["qvh"] == {"queries": 4, "labeled": 3, "unlabeled": 1, "covered": 2,
                             "not_covered": 1, "partially_covered": 1, "coverage": 2/3}
    assert result["avc"]["coverage"] is None
