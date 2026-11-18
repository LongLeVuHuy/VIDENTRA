"""Hash immutable run inputs, verify exported files, and enforce LTR leakage provenance."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from videntra.evaluation.split_lock import qvh_source_id
from videntra.ranking.features import LTR_FEATURE_NAMES, LTR_FEATURE_NAMES_V2

_AVC_WINDOW_RE = re.compile(r"^(.+?)__s\d+__e\d+$")
_MSRVTT_ID_RE = re.compile(r"^video\d+$")
SOURCE_GROUP_RULE = "canonical_avc_window_qvh_youtube_msrvtt_connected_components_v1"
_SYNTHETIC_DATASETS: frozenset[str] = frozenset({"synthetic", "mock", "test", "unit"})
_LINEAR_FOLD_TYPES: frozenset[str] = frozenset({"pairwise_logistic", "linear", "logistic", "fallback_linear"})
_TREE_FOLD_TYPES: frozenset[str] = frozenset({"tree_ensemble", "gbdt", "lightgbm"})


def resolve_canonical_source_group(
    clip_id: str,
    *,
    dataset: str | None = None,
    asset_manifest: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve a clip/video ID to its canonical ``(source_group, status)``.

    Strips AVC ``__s...__e...`` window suffixes, extracts QVHighlights YouTube
    source IDs, preserves MSR-VTT video IDs, and falls back to an explicit
    ``unknown:<clip_id>`` group with ``unknown_unverified`` status.
    """
    cid = str(clip_id).strip()
    if not cid:
        raise ValueError("clip_id must be a non-empty string")

    if asset_manifest and cid in asset_manifest:
        mapped = str(asset_manifest[cid]).strip()
        if mapped:
            return mapped, "manifest_verified"

    avc_match = _AVC_WINDOW_RE.fullmatch(cid)
    if avc_match:
        return avc_match.group(1), "avc_window_stripped"

    try:
        return qvh_source_id(cid), "qvh_youtube_id"
    except ValueError:
        pass

    if dataset == "msrvtt" or _MSRVTT_ID_RE.fullmatch(cid):
        return cid, "msrvtt_video_id"

    return f"unknown:{cid}", "unknown_unverified"


def compute_source_manifest_hash(manifest_records: Sequence[Mapping[str, Any]]) -> str:
    """Compute a deterministic SHA-256 hash over canonical query -> source_group records."""
    normalized = sorted(
        (dict(rec) for rec in manifest_records),
        key=lambda rec: str(rec.get("query_id", "")),
    )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_connected_source_groups(
    query_targets: Sequence[tuple[str, Sequence[str]]],
    *,
    dataset_by_query: Mapping[str, str] | None = None,
    asset_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve canonical source groups and merge multi-target groups via connected components.

    All target clips/groups of a query (and any queries sharing a source group)
    are assigned to the same connected-component fold group so no source group
    or multi-target query can split across CV folds.
    """
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    raw_groups_by_query: dict[str, list[str]] = {}
    statuses_by_query: dict[str, list[str]] = {}
    status_counts: dict[str, int] = defaultdict(int)
    unknown_clips: list[str] = []
    unique_clips: set[str] = set()
    unique_raw_groups: set[str] = set()

    for qid, targets in query_targets:
        q_str = str(qid)
        ds = dataset_by_query.get(q_str) if dataset_by_query else None
        q_groups: list[str] = []
        q_statuses: list[str] = []
        for tgt in targets:
            t_str = str(tgt).strip()
            if not t_str:
                continue
            unique_clips.add(t_str)
            grp, status = resolve_canonical_source_group(
                t_str, dataset=ds, asset_manifest=asset_manifest
            )
            unique_raw_groups.add(grp)
            status_counts[status] += 1
            if status == "unknown_unverified":
                unknown_clips.append(t_str)
            if grp not in parent:
                parent[grp] = grp
            q_groups.append(grp)
            q_statuses.append(status)

        if not q_groups:
            raise ValueError(f"Query {q_str} has no target clips for source_group resolution")

        first = q_groups[0]
        for g in q_groups[1:]:
            _union(first, g)

        raw_groups_by_query[q_str] = sorted(set(q_groups))
        statuses_by_query[q_str] = sorted(set(q_statuses))

    groups_by_root: dict[str, set[str]] = defaultdict(set)
    for grp in parent:
        groups_by_root[_find(grp)].add(grp)

    fold_group_by_query: dict[str, str] = {}
    manifest_records: list[dict[str, Any]] = []
    multi_group_queries: list[str] = []

    for qid, _ in query_targets:
        q_str = str(qid)
        q_groups = raw_groups_by_query[q_str]
        root = _find(q_groups[0])
        comp_groups = sorted(groups_by_root[root])
        if len(comp_groups) == 1:
            comp_id = comp_groups[0]
        else:
            comp_id = "cc:" + "+".join(comp_groups)
            if len(q_groups) > 1:
                multi_group_queries.append(q_str)
        fold_group_by_query[q_str] = comp_id
        manifest_records.append({
            "query_id": q_str,
            "source_group": comp_id,
            "raw_source_groups": q_groups,
            "resolution_statuses": statuses_by_query[q_str],
        })

    manifest_hash = compute_source_manifest_hash(manifest_records)
    return {
        "fold_group_by_query": fold_group_by_query,
        "raw_groups_by_query": raw_groups_by_query,
        "statuses_by_query": statuses_by_query,
        "manifest_records": manifest_records,
        "manifest_hash": manifest_hash,
        "group_stats": {
            "total_queries": len(query_targets),
            "unique_clips": len(unique_clips),
            "unique_source_groups": len(unique_raw_groups),
            "unique_fold_groups": len(set(fold_group_by_query.values())),
            "multi_group_queries": sorted(set(multi_group_queries)),
            "status_counts": dict(sorted(status_counts.items())),
            "unknown_clips": sorted(set(unknown_clips)),
        },
    }


def _normalize_fold_model_family(raw_type: Any) -> str | None:
    if raw_type is None:
        return None
    low = str(raw_type).strip().lower()
    if not low:
        return None
    if low in _TREE_FOLD_TYPES:
        return "tree_ensemble"
    if low in _LINEAR_FOLD_TYPES:
        return "pairwise_logistic"
    return low


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


def _iter_fold_entries(fold_weights: Any) -> list[dict[str, Any]]:
    if isinstance(fold_weights, dict):
        return [entry for entry in fold_weights.values() if isinstance(entry, dict)]
    if isinstance(fold_weights, list):
        return [entry for entry in fold_weights if isinstance(entry, dict)]
    return []


def _extract_fold_query_map(fold_weights: Any) -> dict[str, dict[str, Any]]:
    """Build a query_id -> OOF fold payload mapping from fold_weights."""
    fold_query_map: dict[str, dict[str, Any]] = {}
    for entry in _iter_fold_entries(fold_weights):
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
    *,
    root_model_type: str | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Validate OOF fold_weights for group overlap, model-type parity, duplicates, and missing queries."""
    errors: list[str] = []
    fold_summary: list[dict[str, Any]] = []
    items = _iter_fold_entries(fold_weights)
    if not items:
        return errors, [], fold_summary

    root_family = _normalize_fold_model_family(root_model_type)
    seen_eval_qids: dict[str, Any] = {}
    for idx, entry in enumerate(items):
        fold_id = entry.get("fold_id", entry.get("fold", idx))
        train_groups = {str(g) for g in (entry.get("train_group_ids") or [])}
        eval_groups = {str(g) for g in (entry.get("eval_group_ids") or [])}
        overlap_groups = sorted(train_groups & eval_groups)
        if overlap_groups:
            errors.append(f"train_eval_group_overlap:fold_{fold_id}:{overlap_groups[:5]}")

        explicit_fold_type = _normalize_fold_model_family(entry.get("model_type"))
        if explicit_fold_type is not None:
            if root_family is not None and explicit_fold_type != root_family:
                errors.append(
                    f"fold_model_type_mismatch:fold_{fold_id}:expected_{root_family}_got_{explicit_fold_type}"
                )
            if explicit_fold_type == "tree_ensemble" and not (
                isinstance(entry.get("trees"), list) and len(entry["trees"]) > 0
            ):
                errors.append(f"invalid_fold_trees:fold_{fold_id}")
            if explicit_fold_type == "pairwise_logistic" and not isinstance(entry.get("weights"), list):
                errors.append(f"invalid_fold_weights:fold_{fold_id}")
        elif "trees" in entry and not (isinstance(entry.get("trees"), list) and len(entry["trees"]) > 0):
            errors.append(f"invalid_fold_trees:fold_{fold_id}")
        elif "weights" not in entry and "trees" not in entry:
            errors.append(f"missing_fold_model_payload:fold_{fold_id}")

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
            "model_type": explicit_fold_type or ("tree_ensemble" if "trees" in entry else "pairwise_logistic"),
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
    def _empty(path_str: str | None, resolved_str: str | None, exists: bool, error: str | None, sha: str | None) -> dict[str, Any]:
        return {
            "path": path_str,
            "resolved_path": resolved_str,
            "exists": exists,
            "error": error,
            "ltr_model_sha256": sha,
            "dataset": None,
            "model_type": None,
            "feature_names": [],
            "n_features": 0,
            "trained_query_ids": [],
            "trained_group_ids": [],
            "has_fold_weights": False,
            "fold_query_map": {},
            "fold_group_map": {},
            "fold_by_id": {},
            "oof_validation_errors": [],
            "missing_oof_query_ids": [],
            "fold_summary": [],
            "manifest_hash": None,
            "source_group_rule": None,
        }

    if ltr_model_path is None:
        return _empty(None, None, False, None, None)
    path = Path(ltr_model_path)
    resolved_path = str(path.resolve())
    if not path.is_file():
        return _empty(str(path), resolved_path, False, "missing_artifact", None)

    model_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("LTR artifact JSON root must be an object")
    except Exception as exc:
        return _empty(str(path), resolved_path, True, f"corrupt_artifact:{exc}", model_sha256)

    metadata = payload.get("metadata") or {}
    dataset = payload.get("trained_on_dataset") or metadata.get("dataset")
    model_type = payload.get("model_type")
    raw_feature_names = payload.get("feature_names")
    feature_names = [str(f) for f in raw_feature_names] if isinstance(raw_feature_names, list) else list(LTR_FEATURE_NAMES)
    schema_error = None
    tup_names = tuple(feature_names)
    valid_prefixes = (
        LTR_FEATURE_NAMES[:18],
        LTR_FEATURE_NAMES[:22],
        LTR_FEATURE_NAMES,
        LTR_FEATURE_NAMES_V2,
    )
    if tup_names not in valid_prefixes:
        schema_error = f"invalid_feature_schema:expected_24d_or_32d_got_{len(feature_names)}"

    fold_weights = payload.get("fold_weights")
    fold_query_map = _extract_fold_query_map(fold_weights)
    fold_group_map: dict[str, dict[str, Any]] = {}
    fold_by_id: dict[str, dict[str, Any]] = {}
    trained_group_set: set[str] = {
        str(g) for g in (payload.get("trained_group_ids") or metadata.get("trained_group_ids") or []) if str(g)
    }
    for idx, entry in enumerate(_iter_fold_entries(fold_weights)):
        fid_key = str(entry.get("fold_id", entry.get("fold", idx)))
        fold_by_id[fid_key] = entry
        for grp in (entry.get("train_group_ids") or []):
            g_str = str(grp)
            if g_str:
                trained_group_set.add(g_str)
        for grp in (entry.get("eval_group_ids") or []):
            g_str = str(grp)
            if g_str:
                trained_group_set.add(g_str)
                fold_group_map[g_str] = entry

    raw_qids = payload.get("trained_query_ids") or metadata.get("trained_query_ids") or []
    trained_query_ids = [str(q) for q in raw_qids]
    if not trained_query_ids and fold_query_map:
        trained_query_ids = sorted(fold_query_map.keys())

    oof_errors, missing_oof_qids, fold_summary = _validate_oof_folds(
        fold_weights,
        trained_query_ids,
        root_model_type=str(model_type) if model_type is not None else None,
    )
    manifest_hash = payload.get("manifest_hash") or metadata.get("manifest_hash")
    source_group_rule = payload.get("source_group_rule") or metadata.get("source_group_rule")
    if manifest_hash is not None:
        for idx, entry in enumerate(_iter_fold_entries(fold_weights)):
            if entry.get("manifest_hash") is not None and str(entry["manifest_hash"]) != str(manifest_hash):
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
        "trained_group_ids": sorted(trained_group_set),
        "has_fold_weights": bool(fold_query_map),
        "fold_query_map": fold_query_map,
        "fold_group_map": fold_group_map,
        "fold_by_id": fold_by_id,
        "oof_validation_errors": oof_errors,
        "missing_oof_query_ids": missing_oof_qids,
        "fold_summary": fold_summary,
        "manifest_hash": str(manifest_hash) if manifest_hash is not None else None,
        "source_group_rule": str(source_group_rule) if source_group_rule is not None else None,
    }


def resolve_oof_fold_for_query(
    qid: str,
    query_source_groups: Sequence[str] | Iterable[str] | None,
    ltr_info: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Resolve an OOF fold payload by exact query_id first, then by source_group."""
    q_str = str(qid)
    fold_query_map = ltr_info.get("fold_query_map") or {}
    if q_str in fold_query_map:
        return fold_query_map[q_str], "oof_query_id"

    fold_group_map = ltr_info.get("fold_group_map") or {}
    trained_group_ids = {str(g) for g in (ltr_info.get("trained_group_ids") or ()) if str(g)}
    groups = [str(g) for g in (query_source_groups or ()) if str(g)]

    matched_folds: dict[str, dict[str, Any]] = {}
    for grp in groups:
        if grp in fold_group_map:
            entry = fold_group_map[grp]
            fid_key = str(entry.get("fold_id", entry.get("fold", id(entry))))
            matched_folds[fid_key] = entry

    if len(matched_folds) == 1:
        return next(iter(matched_folds.values())), "oof_source_group"
    if len(matched_folds) > 1:
        return None, "source_group_spans_multiple_folds"
    if any(grp in trained_group_ids for grp in groups):
        return None, "trained_group_without_oof_fold"
    return None, "disjoint_source_group"


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
    feature_flags: dict[str, Any] | None = None,
    selected_splits: Sequence[str] | None = None,
    split_queries: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    media_sha256_by_video: Mapping[str, str] | None = None,
    unhandled_shared_source_counts: Mapping[str, int] | None = None,
    strict_source_groups: bool = False,
) -> dict[str, Any]:
    """Evaluate leakage-free provenance gate and return structured provenance metadata."""
    overlap_with_eval = compute_split_overlaps(ltr_info, split_query_ids)
    oof_set = set(used_oof_splits)
    fold_query_map = ltr_info.get("fold_query_map") or {}
    trained_qids = {str(q) for q in ltr_info.get("trained_query_ids", [])}
    trained_group_ids = {str(g) for g in (ltr_info.get("trained_group_ids") or []) if str(g)}
    oof_errors = list(ltr_info.get("oof_validation_errors") or [])

    if expected_manifest_hash is not None:
        actual_manifest = ltr_info.get("manifest_hash")
        if actual_manifest != expected_manifest_hash:
            oof_errors.append(
                f"manifest_hash_mismatch:expected_{expected_manifest_hash}_got_{actual_manifest}"
            )

    source_group_overlap_with_eval: dict[str, int] = {}
    media_sha256_overlap_with_eval: dict[str, int] = {}
    split_source_groups_by_qid: dict[str, dict[str, list[str]]] = {}
    unverified_source_groups: dict[str, list[str]] = {}

    if split_queries:
        for split_name, q_list in split_queries.items():
            q_groups_map: dict[str, list[str]] = {}
            unverified_list: list[str] = []
            for q in q_list:
                qid_str = str(q.get("query_id", ""))
                ds_str = str(q.get("dataset", "")).strip().lower()
                raw_tgts = q.get("target_videos") or []
                q_groups: list[str] = []
                for vid in raw_tgts:
                    v_str = str(vid).strip()
                    if not v_str:
                        continue
                    grp, status = resolve_canonical_source_group(v_str, dataset=ds_str or None)
                    q_groups.append(grp)
                    if status == "unknown_unverified" and ds_str not in _SYNTHETIC_DATASETS:
                        unverified_list.append(f"{qid_str}:{v_str}")
                q_groups_map[qid_str] = sorted(set(q_groups))
            split_source_groups_by_qid[split_name] = q_groups_map
            if unverified_list:
                unverified_source_groups[split_name] = sorted(set(unverified_list))[:20]
                if strict_source_groups or (split_name == "heldout_test_201" and bool(ltr_info.get("exists"))):
                    oof_errors.append(
                        f"unverified_source_group:{split_name}:{len(unverified_list)}"
                    )

        eff_trained_groups = set(trained_group_ids)
        trained_media_hashes: set[str] = set()
        trained_ds = ltr_info.get("dataset")
        if trained_ds and trained_ds in split_queries:
            for q in split_queries[trained_ds]:
                qid_str = str(q.get("query_id", ""))
                if not trained_qids or qid_str in trained_qids:
                    eff_trained_groups.update(split_source_groups_by_qid[trained_ds].get(qid_str, ()))
                    if media_sha256_by_video:
                        for v in q.get("target_videos") or ():
                            h = media_sha256_by_video.get(str(v))
                            if h:
                                trained_media_hashes.add(h)

        for split_name, q_list in split_queries.items():
            q_gmap = split_source_groups_by_qid.get(split_name, {})
            src_overlap = 0
            media_overlap = 0
            for q in q_list:
                qid_str = str(q.get("query_id", ""))
                q_grps = set(q_gmap.get(qid_str, ()))
                if eff_trained_groups and (q_grps & eff_trained_groups):
                    src_overlap += 1
                if media_sha256_by_video and trained_media_hashes:
                    q_hashes = {
                        media_sha256_by_video.get(str(v))
                        for v in (q.get("target_videos") or ())
                        if media_sha256_by_video.get(str(v))
                    }
                    if q_hashes & trained_media_hashes:
                        media_overlap += 1
            source_group_overlap_with_eval[split_name] = src_overlap
            if media_sha256_by_video is not None:
                media_sha256_overlap_with_eval[split_name] = media_overlap

    eval_mode_by_split: dict[str, str] = {}
    for split_name, overlap_count in overlap_with_eval.items():
        q_gmap = split_source_groups_by_qid.get(split_name, {})
        src_overlap = source_group_overlap_with_eval.get(split_name, 0)
        media_overlap = media_sha256_overlap_with_eval.get(split_name, 0)
        unhandled_cnt = int((unhandled_shared_source_counts or {}).get(split_name, 0))

        if split_name == "heldout_test_201" and (overlap_count > 0 or src_overlap > 0 or media_overlap > 0):
            eval_mode_by_split[split_name] = "source_leakage_detected"
            continue

        if overlap_count == 0 and src_overlap == 0 and media_overlap == 0 and unhandled_cnt == 0:
            eval_mode_by_split[split_name] = "clean_disjoint"
            continue

        if split_name in oof_set and (fold_query_map or ltr_info.get("fold_group_map")) and not oof_errors and unhandled_cnt == 0:
            qids = [str(q) for q in split_query_ids.get(split_name, ())]
            oof_ok = True
            has_any_oof = False
            for qid in qids:
                q_grps = q_gmap.get(qid, ())
                in_trained_qid = (not trained_qids) or (qid in trained_qids)
                in_trained_grp = bool(trained_group_ids and set(q_grps) & trained_group_ids)
                if in_trained_qid or in_trained_grp:
                    payload_match, _status = resolve_oof_fold_for_query(qid, q_grps, ltr_info)
                    if payload_match is None:
                        oof_ok = False
                        break
                    has_any_oof = True
            eval_mode_by_split[split_name] = "oof" if (oof_ok and has_any_oof) else "fit_all_leakage_detected"
        else:
            eval_mode_by_split[split_name] = "fit_all_leakage_detected"

    modes = set(eval_mode_by_split.values())
    if "source_leakage_detected" in modes:
        eval_mode = "source_leakage_detected"
    elif "fit_all_leakage_detected" in modes or oof_errors:
        eval_mode = "fit_all_leakage_detected"
    elif "oof" in modes:
        eval_mode = "oof"
    else:
        eval_mode = "clean_disjoint"

    heldout_overlap = overlap_with_eval.get("heldout_test_201", 0)
    heldout_src_overlap = source_group_overlap_with_eval.get("heldout_test_201", 0)
    heldout_media_overlap = media_sha256_overlap_with_eval.get("heldout_test_201", 0)
    heldout_source_leakage = (heldout_overlap > 0) or (heldout_src_overlap > 0) or (heldout_media_overlap > 0)

    leakage_free = (
        not heldout_source_leakage
        and "source_leakage_detected" not in modes
        and "fit_all_leakage_detected" not in modes
        and not oof_errors
    )
    gate_passed = (not heldout_source_leakage) and not oof_errors and (leakage_free or bool(allow_insample_eval))

    return {
        "ltr_trained_on": {
            "dataset": ltr_info.get("dataset"),
            "num_trained_queries": len(ltr_info.get("trained_query_ids", [])),
            "trained_query_ids": list(ltr_info.get("trained_query_ids", [])),
            "num_trained_groups": len(ltr_info.get("trained_group_ids", [])),
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
        "source_group_overlap_with_eval": source_group_overlap_with_eval,
        "media_sha256_overlap_with_eval": media_sha256_overlap_with_eval,
        "unverified_source_groups": unverified_source_groups,
        "unhandled_shared_source_counts": dict(unhandled_shared_source_counts or {}),
        "heldout_source_leakage": heldout_source_leakage,
        "eval_mode": eval_mode,
        "eval_mode_by_split": eval_mode_by_split,
        "leakage_free": leakage_free,
        "allow_insample_eval": bool(allow_insample_eval),
        "gate_passed": gate_passed,
        "degraded_channels": sorted({str(c) for c in degraded_channels if c}),
        "feature_flags": dict(feature_flags) if feature_flags is not None else {},
        "selected_splits": [str(s) for s in selected_splits] if selected_splits is not None else sorted(split_query_ids.keys()),
    }


def resolve_a5_preset(
    a5_preset: str | None = None,
    *,
    enable_clip_consensus_boost: bool | None = None,
    ltr_blend_weight: float | None = None,
) -> tuple[bool, float]:
    """Resolve effective (enable_clip_consensus_boost, ltr_blend_weight) from preset + overrides."""
    preset_defaults: dict[str | None, tuple[bool, float]] = {
        None: (True, 0.015),
        "a5_full": (True, 0.015),
        "a5_lite": (False, 0.0),
        "a5_no_consensus": (False, 0.015),
        "a5_no_ltr": (True, 0.0),
    }
    if a5_preset not in preset_defaults:
        raise ValueError(
            f"Unsupported a5_preset {a5_preset!r}; expected one of a5_full, a5_lite, a5_no_consensus, a5_no_ltr"
        )
    cons_default, ltr_default = preset_defaults[a5_preset]
    resolved_cons = cons_default if enable_clip_consensus_boost is None else bool(enable_clip_consensus_boost)
    resolved_ltr = ltr_default if ltr_blend_weight is None else float(ltr_blend_weight)
    return resolved_cons, resolved_ltr


def resolve_query_source_groups(
    query: Mapping[str, Any],
) -> tuple[list[str], str, str]:
    """Resolve canonical source groups list, display group key, and resolution status for a query."""
    targets = [str(v).strip() for v in (query.get("target_videos") or ()) if str(v).strip()]
    if not targets:
        raise ValueError(f"Query {query.get('query_id')} has no target_videos")
    ds = query.get("dataset")
    resolved = [resolve_canonical_source_group(v, dataset=str(ds) if ds is not None else None) for v in targets]
    groups = sorted({g for g, _ in resolved})
    statuses = sorted({s for _, s in resolved})
    group_key = groups[0] if len(groups) == 1 else "|".join(groups)
    status_key = statuses[0] if len(statuses) == 1 else "+".join(statuses)
    return groups, group_key, status_key


def compute_effective_config_sha256(config: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over the effective runtime evaluation configuration."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_code_sha256(extra_paths: Sequence[Path] = ()) -> str:
    """Compute a deterministic SHA-256 across all source files in src/videntra and extra scripts."""
    paths: list[Path] = []
    src_root = Path("src/videntra")
    if src_root.is_dir():
        paths.extend(
            p for p in sorted(src_root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    for ep in extra_paths:
        p = Path(ep)
        if p.is_file():
            paths.append(p)
    digest = hashlib.sha256()
    seen: set[str] = set()
    for p in sorted(paths, key=lambda x: x.as_posix()):
        rel = p.as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        digest.update(rel.encode("utf-8"))
        digest.update(b":")
        digest.update(sha256_file(p).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def get_cpu_thread_settings() -> dict[str, Any]:
    """Capture active BLAS/PyTorch/FAISS thread settings for runtime provenance."""
    settings: dict[str, Any] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }
    try:
        import torch
        settings["torch_num_threads"] = int(torch.get_num_threads())
    except Exception:
        settings["torch_num_threads"] = None
    try:
        import faiss
        settings["faiss_omp_max_threads"] = int(faiss.omp_get_max_threads())
    except Exception:
        settings["faiss_omp_max_threads"] = None
    return settings
