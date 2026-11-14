"""Evaluator-only scoring of finalized predictions against model-produced labels.

The corpus and predictions must already exist. This tool never runs retrieval,
selects samples, or passes labels to inference. Checksums detect changes, but
are not proof of temporal order, independent review, or absence of exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from videntra.evaluation.corpus_gate import verify_inventory
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.evaluator import compute_binary_ndcg_at_k, compute_moment_ap_at_iou
from videntra.retrieval.visual import temporal_iou


def _load_assignment(path: Path, inventory: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.pop("sha256", None)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if digest != hashlib.sha256(payload).hexdigest() or data.get("inventory_sha256") != sha256_file(inventory):
        raise ValueError("Assignment checksum or inventory binding mismatch")
    return data


def evaluate(predictions: Path, expected_prediction_sha256: str, inventory: Path,
             assignment: Path, labels: Path, receipt: Path) -> dict:
    if not verify_inventory(inventory):
        raise ValueError("Invalid inventory")
    assign = _load_assignment(assignment, inventory)
    sources = assign.get("sources", [])
    split_by_video = {row["video_id"]: row["split"] for row in sources}
    inventory_data = json.loads(inventory.read_text(encoding="utf-8"))
    if len(split_by_video) != len(sources) or set(split_by_video) != {row["video_id"] for row in inventory_data["entries"]}:
        raise ValueError("Assignment and frozen inventory do not match")

    paths = [predictions, inventory, assignment, labels, receipt]
    hashes = {str(path): sha256_file(path) for path in paths}
    if hashes[str(predictions)] != expected_prediction_sha256:
        raise ValueError("Prediction hash does not match finalized artifact")
    # Parse predictions completely before opening labels.
    pred_by_query = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid, moments = row.get("query_id"), row.get("results")
        if not isinstance(qid, str) or not qid or qid in pred_by_query or not isinstance(moments, list):
            raise ValueError("Invalid or duplicate prediction query ID")
        previous = math.inf
        for m in moments:
            try:
                score, start, end = float(m["ranking_score"]), float(m["start_sec"]), float(m["end_sec"])
                video_id = m["video_id"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid predicted moment") from exc
            if video_id not in split_by_video or not all(map(math.isfinite, (score, start, end))) or start < 0 or end <= start or score > previous:
                raise ValueError("Invalid predicted moment or ranking")
            previous = score
        pred_by_query[qid] = moments

    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    if metadata.get("sha256") != hashes[str(labels)] or metadata.get("label_provenance") != "model_generated_model_adjudicated_not_human_reviewed":
        raise ValueError("Label receipt mismatch or unsupported provenance")
    qrels = json.loads(labels.read_text(encoding="utf-8"))
    ids = [q["query_id"] for q in qrels]
    if len(ids) != len(set(ids)) or set(ids) != set(pred_by_query):
        raise ValueError("Prediction and label query sets must match exactly")

    groups: dict[str, list[dict]] = {"all_inside": [], "dev": [], "test": []}
    counts = {"all_inside": 0, "mixed_inside_outside": 0, "all_outside": 0, "no_published_positive": 0}
    per_query = []
    for q in qrels:
        gold = [(x["asset_id"], float(x["t_in"]), float(x["t_out"])) for x in q.get("gold_moments", [])]
        if any(not math.isfinite(s) or not math.isfinite(e) or s < 0 or e <= s for _, s, e in gold):
            raise ValueError("Invalid labeled moment")
        targets = {vid for vid, _, _ in gold}
        inside = targets & split_by_video.keys()
        category = ("no_published_positive" if not targets else "all_inside" if inside == targets else
                    "mixed_inside_outside" if inside else "all_outside")
        counts[category] += 1
        record = {"query_id": q["query_id"], "category": category}
        if category == "all_inside":
            preds = pred_by_query[q["query_id"]]
            vids = list(dict.fromkeys(m["video_id"] for m in preds))
            rank = next((i for i, vid in enumerate(vids[:10], 1) if vid in targets), None)
            top = preds[0] if preds else None
            best_iou = max((temporal_iou(float(top["start_sec"]), float(top["end_sec"]), s, e)
                            for vid, s, e in gold if top and top["video_id"] == vid), default=0.0)
            record.update({"hit_at_1": int(bool(rank == 1)), "hit_at_5": int(rank is not None and rank <= 5),
                           "hit_at_10": int(rank is not None), "mrr_at_10": 1 / rank if rank else 0.0,
                           "ndcg_at_10": compute_binary_ndcg_at_k(vids, targets, 10),
                           "moment_hit_at_1_iou05": int(best_iou >= 0.5),
                           "moment_ap_at_05": compute_moment_ap_at_iou(preds, gold, 0.5)})
            groups["all_inside"].append(record)
            relevant_splits = {split_by_video[v] for v in targets}
            if len(relevant_splits) == 1:
                split = next(iter(relevant_splits))
                if split in groups:
                    groups[split].append(record)
        per_query.append(record)
    fields = ("hit_at_1", "hit_at_5", "hit_at_10", "mrr_at_10", "ndcg_at_10", "moment_hit_at_1_iou05", "moment_ap_at_05")
    metrics = {name: {"n": len(items), **{field: sum(item[field] for item in items) / len(items) if items else None for field in fields}}
               for name, items in groups.items()}
    positive_count = counts["all_inside"] + counts["all_outside"] + counts["mixed_inside_outside"]
    if hashes != {str(path): sha256_file(path) for path in paths}:
        raise RuntimeError("An input changed during evaluation")
    return {"status": "PROVISIONAL_MODEL_LABELS_NOT_HUMAN_TRUTH", "counts": counts,
            "positive_coverage": counts["all_inside"] / positive_count if positive_count else None,
            "all_query_coverage": counts["all_inside"] / len(qrels) if qrels else None,
            "metrics": metrics, "per_query": per_query, "input_sha256": hashes,
            "limitations": ["Model-generated and model-adjudicated labels are not human-reviewed ground truth.",
                            "Public queries and two films are not a representative held-out test.",
                            "Outside-corpus and no-published-positive queries are excluded, not scored as negatives.",
                            "Checksums alone do not prove retrieval isolation, corpus independence, or prediction finalization time."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("predictions", "inventory", "assignment", "labels", "receipt", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.predictions_sha256, args.inventory, args.assignment, args.labels, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "metrics": result["metrics"]}, indent=2))
