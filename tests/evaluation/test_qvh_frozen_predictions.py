import json

from scripts.evaluate_qvh_frozen_predictions import evaluate
from videntra.evaluation.corpus_gate import freeze_inventory
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import freeze_split


def test_uncovered_query_stays_in_denominator(tmp_path):
    receipt = tmp_path / "receipt.txt"
    receipt.write_text("fixture")
    video = "abcdefghijk_0.0_10.0"
    inventory = tmp_path / "inventory.json"
    freeze_inventory([{"video_id": video, "sha256": "a" * 64, "rights_status": "allowed",
        "rights_receipt_path": str(receipt.resolve()), "rights_receipt_sha256": sha256_file(receipt),
        "source_group": "abcdefghijk", "duplicate_group": "d1"}], inventory, snapshot_id="fixture")
    annotations = tmp_path / "annotations.jsonl"
    rows = [{"qid": 1, "vid": video, "relevant_windows": [[0, 4]]},
            {"qid": 2, "vid": "lmnopqrstuv_0.0_10.0", "relevant_windows": [[1, 5]]}]
    annotations.write_text("\n".join(map(json.dumps, rows)) + "\n")
    split = tmp_path / "split.json"
    freeze_split([{"query_id": "qvh_1", "split": "test", "source_group": "abcdefghijk"},
                  {"query_id": "qvh_2", "split": "test", "source_group": "lmnopqrstuv"}], split,
                 metadata={"source_files_sha256": {"test": sha256_file(annotations)}})
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"query_id": "qvh_1", "results": [{"video_id": video,
        "start_sec": 0, "end_sec": 4, "ranking_score": 0.8}]}) + "\n" +
        json.dumps({"query_id": "qvh_2", "results": []}) + "\n")
    result = evaluate(predictions=predictions, predictions_sha256=sha256_file(predictions),
                      inventory=inventory, split_lock=split, annotations=annotations, subset="test")
    assert result["n_queries"] == 2
    assert result["inventory_coverage"] == 0.5
    assert result["metrics"]["video_hit_at_1"] == 0.5
    assert result["metrics"]["moment_hit_at_1_iou05"] == 0.5
