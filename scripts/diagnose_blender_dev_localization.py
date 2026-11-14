"""Evaluator-only diagnosis on the fixed Blender dev predictions and model labels.

Oracle values are diagnostics only: they must never select inference candidates,
sampling, ranking, or boundaries. No test labels are evaluated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from videntra.evaluation.corpus_gate import verify_inventory
from videntra.evaluation.provenance import sha256_file


def _load_assignment(path: Path, inventory: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.pop("sha256", None)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if digest != hashlib.sha256(payload).hexdigest() or data.get("inventory_sha256") != sha256_file(inventory):
        raise ValueError("Assignment checksum or inventory binding mismatch")
    return data


def iou(a: float, b: float, c: float, d: float) -> float:
    intersection = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return intersection / union if union > 0 else 0.0


def diagnose(*, predictions: Path, expected_sha256: str, labels: Path, receipt: Path,
             inventory: Path, assignment: Path, coordinates: Path) -> dict:
    if not verify_inventory(inventory):
        raise ValueError("Inventory hash mismatch")
    split = _load_assignment(assignment, inventory)
    dev_videos = {r["video_id"] for r in split["sources"] if r["split"] == "dev"}
    if not dev_videos:
        raise ValueError("No fixed dev video")
    if sha256_file(predictions) != expected_sha256:
        raise ValueError("Prediction hash mismatch")
    preds = {r["query_id"]: r["results"] for r in map(json.loads, predictions.read_text(encoding="utf-8").splitlines())}
    source = json.loads(receipt.read_text(encoding="utf-8"))
    if source.get("sha256") != sha256_file(labels) or source.get("label_provenance") != "model_generated_model_adjudicated_not_human_reviewed":
        raise ValueError("Model label receipt mismatch")
    coords = json.loads(coordinates.read_text(encoding="utf-8"))
    timestamps = {v: sorted(float(c["clip_seconds"]) for c in coords if c["video_id"] == v) for v in dev_videos}
    if any(not values for values in timestamps.values()):
        raise ValueError("Dev video has no sampled anchors")
    result = []
    for q in json.loads(labels.read_text(encoding="utf-8")):
        positives = q.get("gold_moments", [])
        if not positives or any(g["asset_id"] not in dev_videos for g in positives):
            continue
        target = positives[0]["asset_id"]
        if any(g["asset_id"] != target for g in positives):
            continue
        candidates = preds.get(q["query_id"], [])
        top = candidates[0] if candidates else None
        actual = max((iou(top["start_sec"], top["end_sec"], g["t_in"], g["t_out"])
                      for g in positives), default=0.0) if top and top["video_id"] == target else 0.0
        oracle = max(iou(max(0.0, t - 5.0), t + 5.0, g["t_in"], g["t_out"])
                     for t in timestamps[target] for g in positives)
        anchor = (float(top["start_sec"]) + float(top["end_sec"])) / 2 if top and top["video_id"] == target else None
        anchor_inside = anchor is not None and any(g["t_in"] <= anchor <= g["t_out"] for g in positives)
        result.append({"query_id": q["query_id"], "top1_iou": actual, "oracle_iou": oracle,
                       "anchor_inside_labeled_window": anchor_inside})
    if not result:
        raise ValueError("No dev queries with labels")
    return {"status": "DEV_ONLY_MODEL_LABEL_DIAGNOSTIC", "n": len(result),
            "top1_iou05_count": sum(r["top1_iou"] >= 0.5 for r in result),
            "oracle_iou05_count": sum(r["oracle_iou"] >= 0.5 for r in result),
            "anchor_inside_count": sum(r["anchor_inside_labeled_window"] for r in result),
            "records": result,
            "limitations": ["Oracle uses labels only in evaluator, never in retrieval.",
                            "Oracle windows are not clipped to the last decoded timestamp; the result is only a diagnostic bound.",
                            "Labels are model-generated, public, and not human-reviewed.",
                            "Only dev source; tiny sample; no claim about test."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("predictions", "labels", "receipt", "inventory", "assignment", "coordinates", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    args = parser.parse_args()
    result = diagnose(predictions=args.predictions, expected_sha256=args.predictions_sha256,
        labels=args.labels, receipt=args.receipt, inventory=args.inventory, assignment=args.assignment,
        coordinates=args.coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
