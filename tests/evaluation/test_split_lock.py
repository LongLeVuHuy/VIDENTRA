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


def test_canonical_avc_pair_and_multitarget_connected_components():
    from scripts.train_ltr_ranker_cv import (
        assign_connected_source_groups,
        group_kfold_indices,
        resolve_canonical_source_group,
    )

    # Same-source AVC windows strip __s...__e... suffix
    g1, s1 = resolve_canonical_source_group("xvAvN5xn5oY__s140__e150", dataset="avc")
    g2, s2 = resolve_canonical_source_group("xvAvN5xn5oY__s560__e570", dataset="avc")
    assert g1 == g2 == "xvAvN5xn5oY"
    assert s1 == s2 == "avc_window_stripped"

    # QVH, MSR-VTT, manifest override, and unknown handling
    qg, qs = resolve_canonical_source_group("bP5KfdFJzC4_60.0_210.0", dataset="qvhighlights")
    assert qg == "bP5KfdFJzC4" and qs == "qvh_youtube_id"
    mg, ms = resolve_canonical_source_group("video7150", dataset="msrvtt")
    assert mg == "video7150" and ms == "msrvtt_video_id"
    ug, us = resolve_canonical_source_group("custom_clip_xyz")
    assert ug == "unknown:custom_clip_xyz" and us == "unknown_unverified"
    ov_g, ov_s = resolve_canonical_source_group(
        "custom_clip_xyz", asset_manifest={"custom_clip_xyz": "verified_src_1"}
    )
    assert ov_g == "verified_src_1" and ov_s == "manifest_verified"

    # Multi-target query + shared group merges via connected components into one fold
    query_targets = [
        ("q_avc_1", ["xvAvN5xn5oY__s140__e150"]),
        ("q_avc_2", ["xvAvN5xn5oY__s560__e570"]),
        ("q_multi", ["srcA__s0__e10", "srcB__s20__e30"]),
        ("q_bridge", ["srcB__s40__e50"]),
        ("q_other_1", ["video100"]),
        ("q_other_2", ["video101"]),
        ("q_other_3", ["video102"]),
        ("q_unknown", ["unrecognized_id_99"]),
    ]
    res = assign_connected_source_groups(query_targets)
    fg = res["fold_group_by_query"]
    assert fg["q_avc_1"] == fg["q_avc_2"] == "xvAvN5xn5oY"
    assert fg["q_multi"] == fg["q_bridge"] == "cc:srcA+srcB"
    assert res["group_stats"]["unknown_clips"] == ["unrecognized_id_99"]
    assert res["group_stats"]["multi_group_queries"] == ["q_multi"]
    assert len(res["manifest_hash"]) == 64

    groups = [fg[qid] for qid, _ in query_targets]
    splits = group_kfold_indices(groups, n_splits=5)
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        val_qids = {query_targets[i][0] for i in val_idx}
        if "q_avc_1" in val_qids or "q_avc_2" in val_qids:
            assert {"q_avc_1", "q_avc_2"}.issubset(val_qids), f"AVC pair split in fold {fold_idx}"
        if "q_multi" in val_qids or "q_bridge" in val_qids:
            assert {"q_multi", "q_bridge"}.issubset(val_qids), f"Connected component split in fold {fold_idx}"
        tr_grps = {groups[i] for i in tr_idx}
        val_grps = {groups[i] for i in val_idx}
        assert tr_grps.isdisjoint(val_grps)
