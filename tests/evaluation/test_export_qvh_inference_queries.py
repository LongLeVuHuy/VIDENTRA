import json
import pytest

from scripts.export_qvh_inference_queries import export
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import freeze_split


def test_export_contains_only_query_and_id(tmp_path):
    annotations = tmp_path / "labels.jsonl"
    annotations.write_text(json.dumps({"qid": 4, "query": "original query", "vid": "abcdefghijk_0.0_10.0",
                                       "relevant_windows": [[1, 2]]}) + "\n")
    lock = tmp_path / "split.json"
    freeze_split([{"query_id": "qvh_4", "split": "test", "source_group": "abcdefghijk"}], lock,
                 metadata={"source_files_sha256": {"test": sha256_file(annotations)}})
    output = tmp_path / "queries.json"
    report = export(lock, annotations, "test", output)
    assert report["queries"] == 1
    assert json.loads(output.read_text()) == [{"query_id": "qvh_4", "text": "original query"}]
    with pytest.raises(FileExistsError):
        export(lock, annotations, "test", output)
