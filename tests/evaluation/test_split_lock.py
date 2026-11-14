import json
import pytest
from videntra.evaluation.split_lock import freeze_split, verify_split_lock


def test_freeze_rejects_leakage_and_never_overwrites(tmp_path):
    path = tmp_path / "split.json"
    rows = [{"query_id": "q1", "split": "dev", "source_group": "source1"},
            {"query_id": "q2", "split": "test", "source_group": "source1"}]
    with pytest.raises(ValueError, match="multiple splits"):
        freeze_split(rows, path)
    rows[1]["source_group"] = "source2"
    result = freeze_split(rows, path)
    assert result["grouping_status"] == "caller_supplied_unverified"
    assert verify_split_lock(path)
    with pytest.raises(FileExistsError):
        freeze_split(rows, path)
    result["records"][0]["split"] = "changed"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert not verify_split_lock(path)


def test_freeze_requires_source_group(tmp_path):
    with pytest.raises(ValueError, match="source_group"):
        freeze_split([{"query_id": "q", "split": "dev"}], tmp_path / "s.json")


def test_qvh_freeze_excludes_sibling_clips(tmp_path):
    from scripts.freeze_qvh_source_split import build
    dev, test, output = (tmp_path / name for name in ("dev.jsonl", "test.jsonl", "out.json"))
    dev.write_text(json.dumps({"qid": 1, "vid": "abcdefghijk_0.0_10.0"}) + "\n", encoding="utf-8")
    test.write_text("\n".join(json.dumps(x) for x in [
        {"qid": 2, "vid": "abcdefghijk_10.0_20.0"},
        {"qid": 3, "vid": "lmnopqrstuv_0.0_10.0"}]) + "\n", encoding="utf-8")
    manifest = build(dev, test, output)
    assert manifest["metadata"]["test_excluded"] == 1
    assert [row["query_id"] for row in manifest["records"]] == ["qvh_1", "qvh_3"]
    assert verify_split_lock(output)
