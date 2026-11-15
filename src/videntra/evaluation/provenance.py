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


from videntra.ranking.features import LTR_FEATURE_NAMES


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


def _validate_oof_folds(
    fold_weights: Any,
    trained_query_ids: Sequence[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Validate OOF fold_weights for group overlap, duplicate queries, and missing queries."""
    errors: list[str] = []
    fold_summary: list[dict[str, Any]] = []
    if isinstance(fold_weights, dict):
        items = list(fold_weights.values())
    elif isinstance(fold_weights, list):
        items = fold_weights
    else:
        return errors, [], fold_summary

    seen_eval_qids: dict[str, Any] = {}
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            continue
        fold_id = entry.get("fold_id", entry.get("fold", idx))
        train_groups = {str(g) for g in (entry.get("train_group_ids") or [])}
        eval_groups = {str(g) for g in (entry.get("eval_group_ids") or [])}
        overlap_groups = sorted(train_groups & eval_groups)
        if overlap_groups:
            errors.append(f"train_eval_group_overlap:fold_{fold_id}:{overlap_groups[:5]}")

        val_qids = [
            str(q)
            for q in (
                entry.get("eval_query_ids")
                or entry.get("val_query_ids")
                or entry.get("heldout_query_ids")
                or entry.get("query_ids")
                or []
            )
        ]
        for qid in val_qids:
            if qid in seen_eval_qids:
                errors.append(f"duplicate_oof_query:{qid}:folds_{seen_eval_qids[qid]}_{fold_id}")
            else:
                seen_eval_qids[qid] = fold_id

        fold_summary.append({
            "fold_id": fold_id,
            "num_train_groups": len(train_groups),
            "num_eval_groups": len(eval_groups),
            "num_eval_queries": len(val_qids),
        })

    missing_qids = sorted(set(trained_query_ids) - set(seen_eval_qids.keys())) if items else []
    if items and missing_qids:
        errors.append(f"missing_oof_queries:{len(missing_qids)}:{missing_qids[:5]}")
    return errors, missing_qids, fold_summary


def load_ltr_model_info(ltr_model_path: Path | str | None) -> dict[str, Any]:
    """Inspect an LTR model artifact for SHA-256, feature schema, training split/IDs, and OOF fold weights."""
    if ltr_model_path is None:
        return {
            "path": None,
            "resolved_path": None,
            "exists": False,
            "error": None,
            "ltr_model_sha256": None,
            "dataset": None,
            "model_type": None,
            "feature_names": [],
            "n_features": 0,
            "trained_query_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
            "oof_validation_errors": [],
            "missing_oof_query_ids": [],
            "fold_summary": [],
            "manifest_hash": None,
            "source_group_rule": None,
        }
    path = Path(ltr_model_path)
    resolved_path = str(path.resolve())
    if not path.is_file():
        return {
            "path": str(path),
            "resolved_path": resolved_path,
            "exists": False,
            "error": "missing_artifact",
            "ltr_model_sha256": None,
            "dataset": None,
            "model_type": None,
            "feature_names": [],
            "n_features": 0,
            "trained_query_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
            "oof_validation_errors": [],
            "missing_oof_query_ids": [],
            "fold_summary": [],
            "manifest_hash": None,
            "source_group_rule": None,
        }

    model_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("LTR artifact JSON root must be an object")
    except Exception as exc:
        return {
            "path": str(path),
            "resolved_path": resolved_path,
            "exists": True,
            "error": f"corrupt_artifact:{exc}",
            "ltr_model_sha256": model_sha256,
            "dataset": None,
            "model_type": None,
            "feature_names": [],
            "n_features": 0,
            "trained_query_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
            "oof_validation_errors": [],
            "missing_oof_query_ids": [],
            "fold_summary": [],
            "manifest_hash": None,
            "source_group_rule": None,
        }

    metadata = payload.get("metadata") or {}
    dataset = payload.get("trained_on_dataset") or metadata.get("dataset")
    model_type = payload.get("model_type")
    raw_feature_names = payload.get("feature_names")
    feature_names = [str(f) for f in raw_feature_names] if isinstance(raw_feature_names, list) else list(LTR_FEATURE_NAMES)
    schema_error = None
    if len(feature_names) not in (18, 22, len(LTR_FEATURE_NAMES)) or tuple(feature_names) != LTR_FEATURE_NAMES[:len(feature_names)]:
        schema_error = f"invalid_feature_schema:expected_prefix_of_{len(LTR_FEATURE_NAMES)}_got_{len(feature_names)}"

    fold_weights = payload.get("fold_weights")
    fold_query_map = _extract_fold_query_map(fold_weights)

    raw_qids = payload.get("trained_query_ids") or metadata.get("trained_query_ids") or []
    trained_query_ids = [str(q) for q in raw_qids]
    if not trained_query_ids and fold_query_map:
        trained_query_ids = sorted(fold_query_map.keys())

    oof_errors, missing_oof_qids, fold_summary = _validate_oof_folds(fold_weights, trained_query_ids)
    manifest_hash = payload.get("manifest_hash") or metadata.get("manifest_hash")
    source_group_rule = payload.get("source_group_rule") or metadata.get("source_group_rule")
    if manifest_hash is not None and isinstance(fold_weights, (list, dict)):
        items = list(fold_weights.values()) if isinstance(fold_weights, dict) else fold_weights
        for idx, entry in enumerate(items):
            if isinstance(entry, dict) and entry.get("manifest_hash") is not None:
                if str(entry["manifest_hash"]) != str(manifest_hash):
                    fold_id = entry.get("fold_id", entry.get("fold", idx))
                    oof_errors.append(f"fold_manifest_hash_mismatch:fold_{fold_id}")

    return {
        "path": str(path),
        "resolved_path": resolved_path,
        "exists": True,
        "error": schema_error,
        "ltr_model_sha256": model_sha256,
        "dataset": str(dataset) if dataset is not None else None,
        "model_type": str(model_type) if model_type is not None else None,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "trained_query_ids": trained_query_ids,
        "has_fold_weights": bool(fold_query_map),
        "fold_query_map": fold_query_map,
        "oof_validation_errors": oof_errors,
        "missing_oof_query_ids": missing_oof_qids,
        "fold_summary": fold_summary,
        "manifest_hash": str(manifest_hash) if manifest_hash is not None else None,
        "source_group_rule": str(source_group_rule) if source_group_rule is not None else None,
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
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Evaluate leakage-free provenance gate and return structured provenance metadata."""
    overlap_with_eval = compute_split_overlaps(ltr_info, split_query_ids)
    oof_set = set(used_oof_splits)
    fold_query_map = ltr_info.get("fold_query_map") or {}
    trained_qids = {str(q) for q in ltr_info.get("trained_query_ids", [])}
    oof_errors = list(ltr_info.get("oof_validation_errors") or [])

    if expected_manifest_hash is not None:
        actual_manifest = ltr_info.get("manifest_hash")
        if actual_manifest != expected_manifest_hash:
            oof_errors.append(
                f"manifest_hash_mismatch:expected_{expected_manifest_hash}_got_{actual_manifest}"
            )

    eval_mode_by_split: dict[str, str] = {}
    for split_name, overlap_count in overlap_with_eval.items():
        if overlap_count == 0:
            eval_mode_by_split[split_name] = "clean_disjoint"
            continue
        if split_name in oof_set and fold_query_map and not oof_errors:
            qids = [str(q) for q in split_query_ids.get(split_name, ())]
            overlapping_qids = [q for q in qids if not trained_qids or q in trained_qids]
            if overlapping_qids and all(q in fold_query_map for q in overlapping_qids):
                eval_mode_by_split[split_name] = "oof"
            else:
                eval_mode_by_split[split_name] = "fit_all_leakage_detected"
        else:
            eval_mode_by_split[split_name] = "fit_all_leakage_detected"

    modes = set(eval_mode_by_split.values())
    if "fit_all_leakage_detected" in modes or oof_errors:
        eval_mode = "fit_all_leakage_detected"
    elif "oof" in modes:
        eval_mode = "oof"
    else:
        eval_mode = "clean_disjoint"

    heldout_overlap = overlap_with_eval.get("heldout_test_201", 0)
    leakage_free = (heldout_overlap == 0) and ("fit_all_leakage_detected" not in modes) and not oof_errors
    gate_passed = (heldout_overlap == 0) and not oof_errors and (leakage_free or bool(allow_insample_eval))

    return {
        "ltr_trained_on": {
            "dataset": ltr_info.get("dataset"),
            "num_trained_queries": len(ltr_info.get("trained_query_ids", [])),
            "trained_query_ids": list(ltr_info.get("trained_query_ids", [])),
        },
        "ltr_model_resolved_path": ltr_info.get("resolved_path"),
        "ltr_model_sha256": ltr_info.get("ltr_model_sha256"),
        "ltr_feature_schema": {
            "model_type": ltr_info.get("model_type"),
            "n_features": ltr_info.get("n_features", 0),
            "feature_names": list(ltr_info.get("feature_names", [])),
        },
        "manifest_hash": ltr_info.get("manifest_hash"),
        "source_group_rule": ltr_info.get("source_group_rule"),
        "oof_validation_errors": oof_errors,
        "fold_summary": list(ltr_info.get("fold_summary", [])),
        "overlap_with_eval": overlap_with_eval,
        "eval_mode": eval_mode,
        "eval_mode_by_split": eval_mode_by_split,
        "leakage_free": leakage_free,
        "allow_insample_eval": bool(allow_insample_eval),
        "gate_passed": gate_passed,
        "degraded_channels": sorted({str(c) for c in degraded_channels if c}),
    }
