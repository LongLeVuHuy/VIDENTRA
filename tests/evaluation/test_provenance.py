from videntra.evaluation.provenance import hash_paths, verify_hashes


def test_hashes_detect_modified_and_new_inputs(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    first = source / "one.txt"
    first.write_text("one", encoding="utf-8")
    before = hash_paths([source, tmp_path / "missing"])
    assert verify_hashes(before)
    first.write_text("changed", encoding="utf-8")
    assert not verify_hashes(before)
    first.write_text("one", encoding="utf-8")
    (source / "added.txt").write_text("new", encoding="utf-8")
    assert hash_paths([source, tmp_path / "missing"]) != before
