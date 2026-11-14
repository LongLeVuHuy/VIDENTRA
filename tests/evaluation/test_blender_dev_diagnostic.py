import hashlib
import json

from scripts.diagnose_blender_dev_localization import diagnose
from videntra.evaluation.corpus_gate import freeze_inventory
from videntra.evaluation.provenance import sha256_file


def test_diagnostic_uses_only_dev_labels(tmp_path):
    inv = tmp_path / "inventory.json"
    rows = [{"video_id": v, "sha256": str(i) * 64, "rights_status": "unknown", "rights_receipt_path": None,
             "rights_receipt_sha256": None, "source_group": v, "duplicate_group": v}
            for i, v in enumerate(("dev", "test"), 1)]
    freeze_inventory(rows, inv, snapshot_id="s")
    data = {"inventory_sha256": sha256_file(inv), "sources": [{"video_id": "dev", "split": "dev"},
                                                                {"video_id": "test", "split": "test"}]}
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["sha256"] = hashlib.sha256(canonical).hexdigest()
    assignment = tmp_path / "assignment.json"; assignment.write_text(json.dumps(data))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps({"query_id": q, "results": [{"video_id": v, "start_sec": 0, "end_sec": 10}]})
                                     for q, v in (("qd", "dev"), ("qt", "test"))) + "\n")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"query_id": q, "gold_moments": [{"asset_id": v, "t_in": 0, "t_out": 10}]}
                                  for q, v in (("qd", "dev"), ("qt", "test"))]))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sha256": sha256_file(labels),
        "label_provenance": "model_generated_model_adjudicated_not_human_reviewed"}))
    coords = tmp_path / "coords.json"; coords.write_text(json.dumps([{"video_id": "dev", "clip_seconds": 5}]))
    result = diagnose(predictions=predictions, expected_sha256=sha256_file(predictions), labels=labels,
                      receipt=receipt, inventory=inv, assignment=assignment, coordinates=coords)
    assert result["n"] == 1
    assert result["records"][0]["query_id"] == "qd"
    assert result["top1_iou05_count"] == result["oracle_iou05_count"] == 1
