"""Hash immutable run inputs, verify exported files, and enforce LTR leakage provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.is_file():
                    result[str(item)] = sha256_file(item)
        elif path.is_file():
            result[str(path)] = sha256_file(path)
        else:
            result[str(path)] = None
    return result


def verify_hashes(hashes: dict[str, str | None]) -> bool:
    return all((sha256_file(Path(path)) if Path(path).is_file() else None) == digest
               for path, digest in hashes.items())


def _extract_fold_query_map(fold_weights: Any) -> dict[str, dict[str, Any]]:
    """Build a query_id -> OOF fold payload mapping from fold_weights."""
    fold_query_map: dict[str, dict[str, Any]] = {}
    if isinstance(fold_weights, dict):
        items = list(fold_weights.values())
    elif isinstance(fold_weights, list):
        items = fold_weights
    else:
        return fold_query_map

    for entry in items:
        if not isinstance(entry, dict):
            continue
        val_qids = (
            entry.get("eval_query_ids")
            or entry.get("val_query_ids")
            or entry.get("heldout_query_ids")
            or entry.get("query_ids")
            or []
        )
        for qid in val_qids:
            fold_query_map[str(qid)] = entry
    return fold_query_map


def load_ltr_model_info(ltr_model_path: Path | str | None) -> dict[str, Any]:
    """Inspect an LTR model artifact for SHA-256, training split/IDs, and OOF fold weights."""
    if ltr_model_path is None:
        return {
            "path": None,
            "ltr_model_sha256": None,
            "dataset": None,
            "trained_query_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
        }
    path = Path(ltr_model_path)
    if not path.is_file():
        return {
            "path": str(path),
            "ltr_model_sha256": None,
            "dataset": None,
            "trained_query_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
        }

    model_sha256 = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    dataset = payload.get("trained_on_dataset") or metadata.get("dataset")
    fold_query_map = _extract_fold_query_map(payload.get("fold_weights"))

    raw_qids = payload.get("trained_query_ids") or metadata.get("trained_query_ids") or []
    trained_query_ids = [str(q) for q in raw_qids]
    if not trained_query_ids and fold_query_map:
        trained_query_ids = sorted(fold_query_map.keys())

    return {
        "path": str(path),
        "ltr_model_sha256": model_sha256,
        "dataset": str(dataset) if dataset is not None else None,
        "trained_query_ids": trained_query_ids,
        "has_fold_weights": bool(fold_query_map),
        "fold_query_map": fold_query_map,
    }


def compute_split_overlaps(
    ltr_info: dict[str, Any],
    split_query_ids: dict[str, Sequence[str]],
) -> dict[str, int]:
    """Compute intersecting query counts between LTR training set and each evaluated split."""
    trained_qids = {str(q) for q in ltr_info.get("trained_query_ids", [])}
    trained_dataset = ltr_info.get("dataset")
    overlaps: dict[str, int] = {}

    for split_name, qids in split_query_ids.items():
        eval_set = {str(q) for q in qids}
        if trained_qids:
            overlaps[split_name] = len(eval_set & trained_qids)
        elif trained_dataset == "dev_full_859" and split_name in {"dev_150", "dev_full_859", "ablation"}:
            # dev_150 is a strict subset of dev_full_859
            overlaps[split_name] = len(eval_set)
        elif trained_dataset and trained_dataset == split_name:
            overlaps[split_name] = len(eval_set)
        else:
            overlaps[split_name] = 0
    return overlaps


def evaluate_provenance_gate(
    ltr_info: dict[str, Any],
    split_query_ids: dict[str, Sequence[str]],
    used_oof_splits: Iterable[str] = (),
    allow_insample_eval: bool = False,
    degraded_channels: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate leakage-free provenance gate and return structured provenance metadata."""
    overlap_with_eval = compute_split_overlaps(ltr_info, split_query_ids)
    oof_set = set(used_oof_splits)
    fold_query_map = ltr_info.get("fold_query_map") or {}
    trained_qids = {str(q) for q in ltr_info.get("trained_query_ids", [])}

    eval_mode_by_split: dict[str, str] = {}
    for split_name, overlap_count in overlap_with_eval.items():
        if overlap_count == 0:
            eval_mode_by_split[split_name] = "clean_disjoint"
            continue
        if split_name in oof_set and fold_query_map:
            qids = [str(q) for q in split_query_ids.get(split_name, ())]
            overlapping_qids = [q for q in qids if not trained_qids or q in trained_qids]
            if overlapping_qids and all(q in fold_query_map for q in overlapping_qids):
                eval_mode_by_split[split_name] = "oof"
            else:
                eval_mode_by_split[split_name] = "fit_all_leakage_detected"
        else:
            eval_mode_by_split[split_name] = "fit_all_leakage_detected"

    modes = set(eval_mode_by_split.values())
    if "fit_all_leakage_detected" in modes:
        eval_mode = "fit_all_leakage_detected"
    elif "oof" in modes:
        eval_mode = "oof"
    else:
        eval_mode = "clean_disjoint"

    heldout_overlap = overlap_with_eval.get("heldout_test_201", 0)
    leakage_free = (heldout_overlap == 0) and ("fit_all_leakage_detected" not in modes)
    gate_passed = (heldout_overlap == 0) and (leakage_free or bool(allow_insample_eval))

    return {
        "ltr_trained_on": {
            "dataset": ltr_info.get("dataset"),
            "num_trained_queries": len(ltr_info.get("trained_query_ids", [])),
            "trained_query_ids": list(ltr_info.get("trained_query_ids", [])),
        },
        "ltr_model_sha256": ltr_info.get("ltr_model_sha256"),
        "overlap_with_eval": overlap_with_eval,
        "eval_mode": eval_mode,
        "eval_mode_by_split": eval_mode_by_split,
        "leakage_free": leakage_free,
        "allow_insample_eval": bool(allow_insample_eval),
        "gate_passed": gate_passed,
        "degraded_channels": sorted({str(c) for c in degraded_channels if c}),
    }
