import hashlib
import json
import pytest

from scripts.evaluate_momentbench_pilot import evaluate
from videntra.evaluation.corpus_gate import freeze_inventory
from videntra.evaluation.provenance import sha256_file


def test_pilot_scores_covered_queries_and_excludes_unknowns(tmp_path):
    receipt = tmp_path / "rights.txt"
    receipt.write_text("fixture", encoding="utf-8")
    records = [{"video_id": name, "sha256": digit * 64, "rights_status": "allowed",
                "rights_receipt_path": str(receipt), "rights_receipt_sha256": sha256_file(receipt),
                "source_group": name, "duplicate_group": name} for name, digit in (("a", "1"), ("b", "2"))]
    inventory = tmp_path / "inventory.json"
    freeze_inventory(records, inventory, snapshot_id="fixture")
    data = {"inventory_sha256": sha256_file(inventory), "sources": [
        {"video_id": "a", "split": "dev"}, {"video_id": "b", "split": "test"}]}
    data["sha256"] = hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps(data), encoding="utf-8")
    gold = lambda vid: [{"asset_id": vid, "t_in": 0.0, "t_out": 2.0}]
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"query_id": "q1", "gold_moments": gold("a")},
                                  {"query_id": "q2", "gold_moments": gold("outside")},
                                  {"query_id": "q3", "gold_moments": []}]), encoding="utf-8")
    label_receipt = tmp_path / "labels-receipt.json"
    label_receipt.write_text(json.dumps({"sha256": sha256_file(labels), "label_provenance": "model_generated_model_adjudicated_not_human_reviewed"}))
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text("\n".join(json.dumps({"query_id": q, "results": [{"video_id": "a", "start_sec": 0, "end_sec": 2, "ranking_score": 1}]}) for q in ("q1", "q2", "q3")), encoding="utf-8")
    result = evaluate(predictions, sha256_file(predictions), inventory, assignment, labels, label_receipt)
    assert result["counts"] == {"all_inside": 1, "mixed_inside_outside": 0, "all_outside": 1, "no_published_positive": 1}
    assert result["positive_coverage"] == 0.5
    assert result["metrics"]["dev"]["moment_hit_at_1_iou05"] == 1
    assert result["metrics"]["test"]["n"] == 0
    with pytest.raises(ValueError, match="Prediction hash"):
        evaluate(predictions, "0" * 64, inventory, assignment, labels, label_receipt)
