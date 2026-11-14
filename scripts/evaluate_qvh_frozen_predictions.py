"""Evaluate finalized QVHighlights predictions without filtering uncovered queries.

Inputs are evaluator-only. This script does not run retrieval, choose the corpus,
sample media, or provide qrels to an inference process.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from videntra.evaluation.corpus_gate import verify_inventory
from videntra.evaluation.evaluator import compute_moment_ap_at_iou
from videntra.evaluation.provenance import sha256_file
from videntra.evaluation.split_lock import qvh_source_id, verify_split_lock
from videntra.retrieval.visual import temporal_iou


def evaluate(*, predictions: Path, predictions_sha256: str, inventory: Path,
             split_lock: Path, annotations: Path, subset: str) -> dict:
    if subset not in {"dev", "test"} or not verify_inventory(inventory) or not verify_split_lock(split_lock):
        raise ValueError("Invalid inventory, split lock, or subset")
    paths = (predictions, inventory, split_lock, annotations)
    before = {str(p): sha256_file(p) for p in paths}
    if before[str(predictions)] != predictions_sha256:
        raise ValueError("Prediction checksum mismatch")
    inv = json.loads(inventory.read_text(encoding="utf-8"))
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    if before[str(annotations)] != lock.get("metadata", {}).get("source_files_sha256", {}).get(subset):
        raise ValueError("Annotation source checksum mismatch")
    corpus = {row["video_id"] for row in inv["entries"] if row.get("eligibility") == "eligible_pending_independent_review"}
    if not corpus:
        raise ValueError("No eligible corpus assets")
    selected = {r["query_id"]: r for r in lock["records"] if r["split"] == subset}
    if not selected:
        raise ValueError("No queries in requested split")

    # Parse and validate all predictions before reading ground truth.
    pred_by_id = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid, results = row.get("query_id"), row.get("results")
        if qid not in selected or qid in pred_by_id or not isinstance(results, list):
            raise ValueError("Invalid, duplicate, or out-of-split prediction query ID")
        previous = math.inf
        for item in results:
            try:
                video = item["video_id"]
                start, end, score = (float(item[k]) for k in ("start_sec", "end_sec", "ranking_score"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Malformed predicted moment") from exc
            if video not in corpus or not all(map(math.isfinite, (start, end, score))) or start < 0 or end <= start or score > previous:
                raise ValueError("Invalid predicted moment, video, or ranking order")
            previous = score
        pred_by_id[qid] = results
    if set(pred_by_id) != set(selected):
        raise ValueError("Predictions must include every query, including empty results")

    with annotations.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    labels = {f"qvh_{row['qid']}": row for row in rows}
    if len(labels) != len(rows):
        raise ValueError("Duplicate annotation query IDs")
    per_query = []
    for qid, split_row in selected.items():
        row = labels.get(qid)
        if row is None or qvh_source_id(row["vid"]) != split_row["source_group"]:
            raise ValueError("Annotation missing or source group mismatch")
        target = row["vid"]
        preds = pred_by_id[qid]
        videos = list(dict.fromkeys(p["video_id"] for p in preds))
        rank = next((i for i, vid in enumerate(videos[:10], 1) if vid == target), None)
        wins = row.get("relevant_windows", [])
        if not isinstance(wins, list):
            raise ValueError("Invalid relevant_windows")
        windows = []
        for window in wins:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("Invalid moment window")
            a, b = map(float, window)
            if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
                raise ValueError("Invalid moment boundaries")
            windows.append((target, a, b))
        top = preds[0] if preds else None
        best = max((temporal_iou(float(top["start_sec"]), float(top["end_sec"]), a, b)
                    for _, a, b in windows if top and top["video_id"] == target), default=0.0)
        per_query.append({"query_id": qid, "target_in_inventory": target in corpus,
                          "video_hit_at_1": int(rank == 1), "video_hit_at_5": int(rank is not None and rank <= 5),
                          "video_hit_at_10": int(rank is not None), "mrr_at_10": 1 / rank if rank else 0.0,
                          "moment_hit_at_1_iou05": int(best >= 0.5) if windows else None,
                          "moment_ap_at_05": compute_moment_ap_at_iou(preds, windows, 0.5) if windows else None})
    if before != {str(p): sha256_file(p) for p in paths}:
        raise RuntimeError("Evaluation inputs changed during run")
    n = len(per_query)
    moment_rows = [r for r in per_query if r["moment_hit_at_1_iou05"] is not None]
    return {"status": "PROVISIONAL_CORPUS_INDEPENDENCE_UNVERIFIED", "subset": subset,
            "n_queries": n, "n_target_in_inventory": sum(r["target_in_inventory"] for r in per_query),
            "inventory_coverage": sum(r["target_in_inventory"] for r in per_query) / n,
            "n_moment_labeled": len(moment_rows),
            "metrics": {**{key: sum(r[key] for r in per_query) / n for key in
                          ("video_hit_at_1", "video_hit_at_5", "video_hit_at_10", "mrr_at_10")},
                        "moment_hit_at_1_iou05": sum(r["moment_hit_at_1_iou05"] for r in moment_rows) / len(moment_rows) if moment_rows else None,
                        "moment_map_at_05": sum(r["moment_ap_at_05"] for r in moment_rows) / len(moment_rows) if moment_rows else None},
            "per_query": per_query, "input_sha256": before,
            "limitations": ["Corpus rights, source/near-duplicate grouping, timeline and independence require review.",
                            "Inventory membership is not proof that an asset is present in the actual retrieval index.",
                            "Checksums do not prove predictions were finalized before the labels were inspected.",
                            "Uncovered targets remain in denominators and count as misses."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("predictions", "inventory", "split-lock", "annotations", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--subset", required=True, choices=("dev", "test"))
    args = parser.parse_args()
    result = evaluate(predictions=args.predictions, predictions_sha256=args.predictions_sha256,
                      inventory=args.inventory, split_lock=args.split_lock, annotations=args.annotations, subset=args.subset)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("status", "subset", "n_queries", "n_target_in_inventory", "n_moment_labeled", "metrics")}, indent=2))
