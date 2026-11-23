"""5-Fold GroupKFold Cross-Validation trainer for VIDENTRA LearnedRankerLTR.

Trains an L2-regularized pairwise logistic ranker (or LightGBM LambdaMART if
installed and requested) on ``dev_full_859`` grouped strictly by
``source_video_id`` to prevent video-level leakage. Never touches
``heldout_test_201`` or ``test_queries/``.

Saves the compact JSON artifact to ``models/ranking/videntra_ltr.json``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from videntra.evaluation.provenance import (
    SOURCE_GROUP_RULE,
    assign_connected_source_groups,
    compute_source_manifest_hash,
    resolve_canonical_source_group,
)
from videntra.ranking.features import (
    LTR_FEATURE_NAMES,
    LTR_FEATURE_NAMES_V2,
    compute_query_z_scores,
    extract_ltr_features_24d,
    extract_ltr_features_32d,
)
from videntra.ranking.learned_ranker import LearnedRankerLTR

DEFAULT_DATASET_PATH = Path("work/ltr_dev859_features.jsonl")
DEFAULT_VAL_DIR = Path("validation")
DEFAULT_OUTPUT_PATH = Path("models/ranking/videntra_ltr.json")

_FORBIDDEN_PATH_TOKENS = ("heldout_test_201", "heldout", "test_queries")

POSITIVE_MONOTONIC_FEATURES: frozenset[str] = frozenset({
    "s_vis_blend_max",
    "z_vis_max",
    "s_vis_top3_mean",
    "s_vis_top5_mean",
    "s_vis_ctx_coherence",
    "s_late_interaction",
    "rrf_vis",
    "rrf_ocr",
    "rrf_asr",
    "bm25_log_score",
    "high_spec_token_coverage",
    "anchor_lexical_match",
    "number_consistency",
    "yolo_object_verification_boost",
    "graph_relation_score",
    "llm_modality_alignment",
    "vis_lexical_synergy",
    "subclause_weakest_link",
    "window_energy_concentration",
    "temporal_attention_coherence",
    "rrf_vis_caption",
})

NEGATIVE_MONOTONIC_FEATURES: frozenset[str] = frozenset({
    "delta_z_vis",
    "spike_drop_ratio",
    "unverified_lexical_hazard",
})

POSITIVE_MONOTONIC_FEATURES_V2: frozenset[str] = POSITIVE_MONOTONIC_FEATURES | frozenset({
    "vp_whole_cos",
    "vp_subclip_max",
    "vp_z",
    "vp_csls",
    "rrf_vp",
    "lex_reliability_score",
    "spatial_predicate_score",
    "action_order_score",
})

NEGATIVE_MONOTONIC_FEATURES_V2: frozenset[str] = NEGATIVE_MONOTONIC_FEATURES


def assert_no_heldout_leakage(path: str | Path) -> Path:
    """Fail closed if a training or validation path references held-out test data."""
    p = Path(path)
    norm = p.as_posix().lower()
    for token in _FORBIDDEN_PATH_TOKENS:
        if token in norm:
            raise ValueError(
                f"Refusing to access forbidden held-out/test path during LTR training: {p}"
            )
    return p


def group_kfold_indices(
    groups: Sequence[str],
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic GroupKFold split assigning all rows of a group to one fold.

    Uses greedy load-balancing by group size so folds are balanced and strictly
    disjoint by ``source_video_id``.
    """
    n_samples = len(groups)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        group_to_indices[str(g)].append(idx)

    if len(group_to_indices) < n_splits:
        raise ValueError(
            f"Cannot have n_splits={n_splits} > number of unique groups={len(group_to_indices)}"
        )

    # Sort largest groups first, tie-breaking deterministically by group name
    sorted_groups = sorted(
        group_to_indices.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    fold_buckets: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for _, idxs in sorted_groups:
        target_fold = min(range(n_splits), key=lambda f: (fold_sizes[f], f))
        fold_buckets[target_fold].extend(idxs)
        fold_sizes[target_fold] += len(idxs)

    all_indices = np.arange(n_samples, dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        val_idx = np.array(sorted(fold_buckets[f]), dtype=np.int64)
        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        train_idx = all_indices[mask]
        splits.append((train_idx, val_idx))
    return splits


def build_pairwise_differences(
    X: np.ndarray,
    y: np.ndarray,
    query_ids: Sequence[str],
    max_neg_per_pos: int = 20,
    base_scores: np.ndarray | None = None,
    return_sample_weights: bool = False,
) -> tuple[Any, ...]:
    """Construct hard-negative-weighted pairwise feature differences (x_pos - x_neg) per query."""
    by_query: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(query_ids):
        by_query[str(qid)].append(idx)

    diffs: list[np.ndarray] = []
    labels: list[float] = []
    base_diffs: list[float] = []
    pair_weights: list[float] = []

    for idxs in by_query.values():
        pos_idxs = [i for i in idxs if y[i] > 0]
        neg_idxs = [i for i in idxs if y[i] <= 0]
        if not pos_idxs or not neg_idxs:
            continue

        if base_scores is not None:
            neg_idxs_sorted = sorted(neg_idxs, key=lambda idx_n: float(base_scores[idx_n]), reverse=True)
        else:
            neg_idxs_sorted = neg_idxs
        sampled_negs = neg_idxs_sorted[:max_neg_per_pos]

        for p in pos_idxs:
            for rank_neg, n in enumerate(sampled_negs, start=1):
                if base_scores is not None:
                    margin = float(base_scores[p] - base_scores[n])
                    if rank_neg <= 5 or margin <= 0.05:
                        w_pair = 2.0
                    elif rank_neg <= 15:
                        w_pair = 1.25
                    else:
                        w_pair = 0.65
                else:
                    w_pair = 1.0

                delta = X[p] - X[n]
                diffs.append(delta)
                labels.append(1.0)
                pair_weights.append(w_pair)
                diffs.append(-delta)
                labels.append(0.0)
                pair_weights.append(w_pair)
                if base_scores is not None:
                    b_delta = float(base_scores[p] - base_scores[n])
                    base_diffs.append(b_delta)
                    base_diffs.append(-b_delta)

    if not diffs:
        empty_x = np.zeros((0, X.shape[1]), dtype=np.float64)
        empty_y = np.zeros(0, dtype=np.float64)
        empty_w = np.zeros(0, dtype=np.float64)
        if base_scores is not None and return_sample_weights:
            return empty_x, empty_y, np.zeros(0, dtype=np.float64), empty_w
        if base_scores is not None:
            return empty_x, empty_y, np.zeros(0, dtype=np.float64)
        if return_sample_weights:
            return empty_x, empty_y, empty_w
        return empty_x, empty_y

    X_pair = np.vstack(diffs).astype(np.float64)
    y_pair = np.asarray(labels, dtype=np.float64)
    w_arr = np.asarray(pair_weights, dtype=np.float64)
    if base_scores is not None and return_sample_weights:
        return X_pair, y_pair, np.asarray(base_diffs, dtype=np.float64), w_arr
    if base_scores is not None:
        return X_pair, y_pair, np.asarray(base_diffs, dtype=np.float64)
    if return_sample_weights:
        return X_pair, y_pair, w_arr
    return X_pair, y_pair


def _build_monotonic_masks(
    n_features: int,
    feature_names: Sequence[str] = LTR_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean masks for positive (>= 0) and negative (<= 0) monotonic constraints."""
    pos_set = POSITIVE_MONOTONIC_FEATURES_V2 if len(feature_names) >= len(LTR_FEATURE_NAMES_V2) else POSITIVE_MONOTONIC_FEATURES
    neg_set = NEGATIVE_MONOTONIC_FEATURES_V2 if len(feature_names) >= len(LTR_FEATURE_NAMES_V2) else NEGATIVE_MONOTONIC_FEATURES
    pos_mask = np.zeros(n_features, dtype=bool)
    neg_mask = np.zeros(n_features, dtype=bool)
    for idx, name in enumerate(feature_names[:n_features]):
        if name in pos_set:
            pos_mask[idx] = True
        elif name in neg_set:
            neg_mask[idx] = True
    return pos_mask, neg_mask


def train_pairwise_logistic_l2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_ids_train: Sequence[str],
    l2_reg: float = 250000.0,
    learning_rate: float = 0.08,
    max_iter: int = 400,
    monotonic_constraints: bool = True,
    feature_names: Sequence[str] = LTR_FEATURE_NAMES,
    base_scores_train: np.ndarray | None = None,
    blend_weight: float = 0.015,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Train hard-negative-weighted L2-regularized pairwise logistic regression in pure NumPy."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    feat_mean = np.mean(X, axis=0)
    raw_std = np.std(X, axis=0)
    feat_std = np.where(raw_std > 1e-8, raw_std, 1.0)
    X_norm = (X - feat_mean) / feat_std

    if base_scores_train is not None:
        base_arr = np.asarray(base_scores_train, dtype=np.float64)
        X_pair, y_pair, base_pair, pair_w = build_pairwise_differences(
            X_norm, y, query_ids_train, base_scores=base_arr, return_sample_weights=True
        )
    else:
        X_pair, y_pair, pair_w = build_pairwise_differences(
            X_norm, y, query_ids_train, return_sample_weights=True
        )
        base_pair = None

    n_features = X.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    if X_pair.shape[0] == 0:
        return weights, 0.0, feat_mean, feat_std

    pos_mask, neg_mask = _build_monotonic_masks(n_features, feature_names=feature_names)
    weight_sum = max(1.0, float(np.sum(pair_w)))
    lam = float(blend_weight)
    m = np.zeros(n_features, dtype=np.float64)
    v = np.zeros(n_features, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, max_iter + 1):
        if base_pair is not None:
            # Mirror bounded base + blend_weight * (x @ w) residual scoring
            logits = np.clip((base_pair + lam * (X_pair @ weights)) * 12.0, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = (12.0 * lam * (X_pair.T @ ((probs - y_pair) * pair_w))) / weight_sum + (l2_reg / weight_sum) * weights
        else:
            logits = np.clip(X_pair @ weights, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = (X_pair.T @ ((probs - y_pair) * pair_w)) / weight_sum + (l2_reg / weight_sum) * weights

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if monotonic_constraints:
            weights[pos_mask] = np.maximum(0.0, weights[pos_mask])
            weights[neg_mask] = np.minimum(0.0, weights[neg_mask])

    return weights, 0.0, feat_mean, feat_std


def evaluate_ranking_metrics(
    scores: np.ndarray,
    y: np.ndarray,
    query_ids: Sequence[str],
) -> dict[str, float]:
    """Compute Video Hit@1, MRR, and nDCG@10 across queries."""
    by_query: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(query_ids):
        by_query[str(qid)].append(idx)

    hits_at_1 = 0
    mrr_sum = 0.0
    ndcg10_sum = 0.0
    n_queries = 0

    for idxs in by_query.values():
        rel_count = sum(1 for i in idxs if y[i] > 0)
        if rel_count == 0:
            continue
        n_queries += 1
        ranked = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)

        if y[ranked[0]] > 0:
            hits_at_1 += 1

        rr = 0.0
        for rank_pos, i in enumerate(ranked, start=1):
            if y[i] > 0:
                rr = 1.0 / rank_pos
                break
        mrr_sum += rr

        dcg = 0.0
        for rank_pos, i in enumerate(ranked[:10], start=1):
            if y[i] > 0:
                dcg += 1.0 / math.log2(rank_pos + 1)
        ideal_k = min(rel_count, 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_k + 1))
        ndcg10_sum += (dcg / idcg) if idcg > 0 else 0.0

    if n_queries == 0:
        return {"hit_at_1": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0, "num_queries": 0}

    return {
        "hit_at_1": round(hits_at_1 / n_queries, 4),
        "mrr": round(mrr_sum / n_queries, 4),
        "ndcg_at_10": round(ndcg10_sum / n_queries, 4),
        "num_queries": n_queries,
    }


def _train_lightgbm_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_ids_train: Sequence[str],
    feature_names: Sequence[str] = LTR_FEATURE_NAMES,
) -> dict[str, Any] | None:
    """LightGBM LambdaMART training using native Dataset/train API (no sklearn dependency)."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    pos_mask, neg_mask = _build_monotonic_masks(X_train.shape[1], feature_names=feature_names)
    mono = [1 if bool(pos_mask[i]) else (-1 if bool(neg_mask[i]) else 0) for i in range(X_train.shape[1])]

    # Group rows contiguously by query_id for lambdarank
    order = sorted(range(len(query_ids_train)), key=lambda i: str(query_ids_train[i]))
    X_ord = X_train[order]
    y_ord = y_train[order]
    group_counts: list[int] = []
    last_q = None
    for i in order:
        q = str(query_ids_train[i])
        if q != last_q:
            group_counts.append(1)
            last_q = q
        else:
            group_counts[-1] += 1

    if hasattr(lgb, "Dataset") and hasattr(lgb, "train"):
        dtrain = lgb.Dataset(X_ord, label=y_ord, group=group_counts, free_raw_data=False)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.04,
            "min_data_in_leaf": 25,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
            "monotone_constraints": mono,
            "seed": 42,
            "verbosity": -1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=70)
        dump = booster.dump_model()
        ranker_obj: Any = booster
    else:
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=70,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.04,
            min_child_samples=25,
            random_state=42,
            verbose=-1,
        )
        ranker.fit(X_ord, y_ord, group=group_counts)
        dump = ranker.booster_.dump_model()
        ranker_obj = ranker

    def _convert_tree(node: dict[str, Any]) -> dict[str, Any]:
        if "leaf_value" in node:
            return {"leaf_value": float(node["leaf_value"])}
        return {
            "feature_idx": int(node["split_feature"]),
            "threshold": float(node["threshold"]),
            "left": _convert_tree(node["left_child"]),
            "right": _convert_tree(node["right_child"]),
        }

    trees = [_convert_tree(t["tree_structure"]) for t in dump.get("tree_info", [])]
    return {
        "model_type": "tree_ensemble",
        "trees": trees,
        "bias": 0.0,
        "learning_rate": 1.0,
        "ranker": ranker_obj,
    }


def run_group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    query_ids: Sequence[str],
    source_video_ids: Sequence[str],
    n_splits: int = 5,
    l2_reg: float = 250000.0,
    backend: str = "pairwise_logistic",
    base_scores: np.ndarray | None = None,
    blend_weight: float = 0.015,
    manifest_hash: str | None = None,
    source_group_rule: str = SOURCE_GROUP_RULE,
    coverage_stats: Mapping[str, Any] | None = None,
    group_stats: Mapping[str, Any] | None = None,
    raw_groups_by_query: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run 5-Fold GroupKFold CV grouped by canonical source group and build final artifact."""
    # Verify every query maps to a single consistent fold group across all its rows
    q_group_check: dict[str, set[str]] = defaultdict(set)
    for qid, grp in zip(query_ids, source_video_ids):
        q_group_check[str(qid)].add(str(grp))
    split_queries = {q: sorted(gs) for q, gs in q_group_check.items() if len(gs) > 1}
    if split_queries:
        raise ValueError(
            f"Queries span multiple unmerged source groups: {list(split_queries.items())[:3]}"
        )

    if manifest_hash is None:
        manifest_records = [
            {
                "query_id": qid,
                "source_group": next(iter(gs)),
                "raw_source_groups": (
                    list(raw_groups_by_query[qid])
                    if raw_groups_by_query and qid in raw_groups_by_query
                    else [next(iter(gs))]
                ),
            }
            for qid, gs in sorted(q_group_check.items())
        ]
        resolved_manifest_hash = compute_source_manifest_hash(manifest_records)
    else:
        resolved_manifest_hash = str(manifest_hash).strip()
        if not resolved_manifest_hash:
            raise ValueError("manifest_hash must be non-empty")

    splits = group_kfold_indices(source_video_ids, n_splits=n_splits)
    base_arr = np.asarray(base_scores, dtype=np.float64) if base_scores is not None else None
    active_feature_names = (
        LTR_FEATURE_NAMES_V2 if X.shape[1] == len(LTR_FEATURE_NAMES_V2) else LTR_FEATURE_NAMES
    )

    fold_metrics: list[dict[str, float]] = []
    fold_weights: list[dict[str, Any]] = []

    use_lgb = backend == "lightgbm"
    if backend == "auto":
        try:
            import lightgbm  # noqa: F401
            use_lgb = True
        except ImportError:
            use_lgb = False
    elif use_lgb:
        import lightgbm  # noqa: F401

    def _groups_for_indices(indices: np.ndarray) -> set[str]:
        grps: set[str] = set()
        for i in indices:
            qid = str(query_ids[i])
            if raw_groups_by_query and qid in raw_groups_by_query:
                grps.update(str(g) for g in raw_groups_by_query[qid])
            else:
                grps.add(str(source_video_ids[i]))
        return grps

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        q_tr = [query_ids[i] for i in tr_idx]
        q_val = [query_ids[i] for i in val_idx]
        base_tr = base_arr[tr_idx] if base_arr is not None else None

        # Verify zero source-group leakage between train and val folds
        tr_groups = _groups_for_indices(tr_idx)
        val_groups = _groups_for_indices(val_idx)
        if not tr_groups.isdisjoint(val_groups):
            overlap = sorted(tr_groups & val_groups)
            raise ValueError(f"GroupKFold source group leakage in fold {fold_idx}: {overlap[:5]}")

        fold_meta_base = {
            "fold_id": fold_idx,
            "fold": fold_idx,
            "train_group_ids": sorted(tr_groups),
            "eval_group_ids": sorted(val_groups),
            "eval_query_ids": sorted({str(q) for q in q_val}),
            "eval_source_video_ids": sorted({str(source_video_ids[i]) for i in val_idx}),
            "manifest_hash": resolved_manifest_hash,
        }
        fold_lgb_payload = _train_lightgbm_fold(X_tr, y_tr, q_tr, feature_names=active_feature_names) if use_lgb else None
        if use_lgb and fold_lgb_payload is None:
            raise RuntimeError("backend='lightgbm' requested but LightGBM training returned None")
        if fold_lgb_payload is not None:
            fold_entry = {
                **fold_meta_base,
                "model_type": "tree_ensemble",
                "feature_names": list(active_feature_names),
                "trees": fold_lgb_payload["trees"],
                "bias": float(fold_lgb_payload.get("bias", 0.0)),
                "learning_rate": float(fold_lgb_payload.get("learning_rate", 1.0)),
            }
        else:
            # Per-fold normalization: f_mean and f_std are fit strictly on X_tr
            w, bias, f_mean, f_std = train_pairwise_logistic_l2(
                X_tr,
                y_tr,
                q_tr,
                l2_reg=l2_reg,
                feature_names=active_feature_names,
                base_scores_train=base_tr,
                blend_weight=blend_weight,
            )
            fold_entry = {
                **fold_meta_base,
                "model_type": "pairwise_logistic",
                "feature_names": list(active_feature_names),
                "weights": [round(float(v), 6) for v in w],
                "bias": round(float(bias), 6),
                "feature_mean": [round(float(v), 6) for v in f_mean],
                "feature_std": [round(float(v), 6) for v in f_std],
            }
        fold_weights.append(fold_entry)

        fold_ranker = LearnedRankerLTR(model_path=None)
        fold_ranker.apply_fold_payload(fold_entry)
        ltr_val_scores = fold_ranker.score_matrix(X_val)
        if base_arr is not None:
            val_scores = base_arr[val_idx] + float(blend_weight) * np.tanh(ltr_val_scores)
        else:
            val_scores = ltr_val_scores
        metrics = evaluate_ranking_metrics(val_scores, y_val, q_val)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    # Validate OOF completeness & uniqueness across all folds
    seen_oof_qids: set[str] = set()
    for fw in fold_weights:
        qset = set(fw["eval_query_ids"])
        dup_qids = seen_oof_qids & qset
        if dup_qids:
            raise ValueError(f"Duplicate OOF query_ids across folds: {sorted(dup_qids)[:5]}")
        seen_oof_qids |= qset
    all_qids = {str(q) for q in query_ids}
    if seen_oof_qids != all_qids:
        missing_qids = sorted(all_qids - seen_oof_qids)
        raise ValueError(f"Missing OOF query_ids across folds: {missing_qids[:5]}")

    # Fit final model on full dev_full_859 dataset
    mean_cv_hit1 = float(np.mean([m["hit_at_1"] for m in fold_metrics]))
    mean_cv_mrr = float(np.mean([m["mrr"] for m in fold_metrics]))
    mean_cv_ndcg10 = float(np.mean([m["ndcg_at_10"] for m in fold_metrics]))

    lgb_payload = _train_lightgbm_fold(X, y, query_ids, feature_names=active_feature_names) if use_lgb else None
    if lgb_payload is not None:
        artifact: dict[str, Any] = {
            "model_type": "tree_ensemble",
            "feature_names": list(active_feature_names),
            "trees": lgb_payload["trees"],
            "bias": 0.0,
            "learning_rate": 1.0,
        }
    else:
        final_w, final_bias, final_mean, final_std = train_pairwise_logistic_l2(
            X,
            y,
            query_ids,
            l2_reg=l2_reg,
            feature_names=active_feature_names,
            base_scores_train=base_arr,
            blend_weight=blend_weight,
        )
        artifact = {
            "model_type": "pairwise_logistic",
            "feature_names": list(active_feature_names),
            "weights": [round(float(v), 6) for v in final_w],
            "bias": round(float(final_bias), 6),
            "feature_mean": [round(float(v), 6) for v in final_mean],
            "feature_std": [round(float(v), 6) for v in final_std],
        }

    artifact["manifest_hash"] = resolved_manifest_hash
    artifact["source_group_rule"] = source_group_rule
    artifact["fold_weights"] = fold_weights
    artifact["metadata"] = {
        "dataset": "dev_full_859",
        "trained_split": "dev_full_859",
        "trained_query_ids": sorted(all_qids),
        "cv_strategy": f"{n_splits}-fold GroupKFold",
        "grouped_by": "source_group",
        "source_group_rule": source_group_rule,
        "manifest_hash": resolved_manifest_hash,
        "l2_reg": float(l2_reg),
        "blend_weight": float(blend_weight),
        "residual_training": bool(base_arr is not None),
        "cv_hit_at_1": round(mean_cv_hit1, 4),
        "cv_mrr": round(mean_cv_mrr, 4),
        "cv_ndcg_at_10": round(mean_cv_ndcg10, 4),
        "fold_metrics": fold_metrics,
        "coverage_stats": dict(coverage_stats) if coverage_stats is not None else None,
        "group_stats": dict(group_stats) if group_stats is not None else None,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return artifact


def load_ltr_jsonl_dataset(
    dataset_path: Path,
    return_base_scores: bool = False,
) -> tuple[Any, ...]:
    """Load pre-extracted candidate rows from JSONL (no held-out/test paths)."""
    assert_no_heldout_leakage(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"LTR training dataset not found: {dataset_path}")

    rows: list[np.ndarray] = []
    labels: list[float] = []
    query_ids: list[str] = []
    source_video_ids: list[str] = []
    base_scores: list[float] = []
    has_all_base = True

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "features" in item and isinstance(item["features"], list) and len(item["features"]) in (len(LTR_FEATURE_NAMES), len(LTR_FEATURE_NAMES_V2)):
                vec = np.asarray(item["features"], dtype=np.float32)
            elif "features" in item and isinstance(item["features"], list) and len(item["features"]) == 22:
                vec = np.concatenate([
                    np.asarray(item["features"], dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                ])
            else:
                vec = extract_ltr_features_24d(item, return_dict=False)
            rows.append(vec)
            labels.append(float(item.get("label", item.get("is_relevant", 0))))
            query_ids.append(str(item["query_id"]))
            source_video_ids.append(str(item.get("source_video_id", item.get("video_id", ""))))
            if "base_video_score" in item and item["base_video_score"] is not None:
                base_scores.append(float(item["base_video_score"]))
            else:
                has_all_base = False

    if not rows:
        raise ValueError(f"Empty LTR dataset: {dataset_path}")

    X_mat = np.vstack(rows).astype(np.float64)
    y_arr = np.asarray(labels, dtype=np.float64)
    if return_base_scores:
        base_arr = np.asarray(base_scores, dtype=np.float64) if has_all_base else None
        return X_mat, y_arr, query_ids, source_video_ids, base_arr
    return X_mat, y_arr, query_ids, source_video_ids


def extract_dev859_ltr_dataset(
    val_dir: Path = DEFAULT_VAL_DIR,
    save_jsonl_path: Path | None = None,
    max_candidates_per_query: int = 60,
    return_base_scores: bool = False,
    return_metadata: bool = False,
    feature_dim: int = 24,
) -> tuple[Any, ...]:
    """Extract 24D or 32D LTR features from dev_full_859 (validation/ only, never held-out test)."""
    assert_no_heldout_leakage(val_dir)
    if save_jsonl_path is not None:
        assert_no_heldout_leakage(save_jsonl_path)

    import sys

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from scripts.run_validation_benchmark import load_validation_queries
    from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline
    import torch

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = UnifiedMultimodalSearchPipeline(
        db_path=Path("data/processed/catalog/corpus_metadata.db"),
        faiss_path=Path("data/processed/visual_keyframes/siglip2_keyframes_768d.faiss"),
        model_dir=Path("models/siglip2"),
        device=runtime_device,
        use_disk_cache=True,
        memory_cache_size=1024,
        ltr_blend_weight=0.0,
    )
    try:
        indexed_videos = set(pipeline.video_durations.keys()) if pipeline.video_durations else set()
        if not indexed_videos and pipeline.keyframe_map:
            indexed_videos = {k["video_id"] for k in pipeline.keyframe_map.values()}

        queries = load_validation_queries(val_dir, indexed_videos, max_msrvtt=None)
        if not queries:
            raise ValueError(f"No validation queries loaded from {val_dir}")

        group_assignment = assign_connected_source_groups(
            [(str(q["query_id"]), list(q["target_videos"])) for q in queries],
            dataset_by_query={str(q["query_id"]): str(q.get("dataset", "")) for q in queries},
        )
        fold_group_by_query = group_assignment["fold_group_by_query"]
        raw_groups_by_query = group_assignment["raw_groups_by_query"]
        manifest_hash = group_assignment["manifest_hash"]
        group_stats = group_assignment["group_stats"]

        rows: list[np.ndarray] = []
        labels: list[float] = []
        query_ids: list[str] = []
        source_video_ids: list[str] = []
        base_scores: list[float] = []
        jsonl_records: list[dict[str, Any]] = []

        queries_with_candidates = 0
        queries_with_positive = 0
        queries_with_pairable = 0
        missing_positive_query_ids: list[str] = []

        for idx, q in enumerate(queries, start=1):
            qid = str(q["query_id"])
            target_videos = list(q["target_videos"])
            target_set = set(target_videos)
            src_vid = fold_group_by_query[qid]

            resp = pipeline.search_with_metadata(
                snapshot_id="ltr_extract_dev859",
                query=str(q["query_text"]),
                top_k=30,
                mode="accurate",
                ranking_mode="diverse_video",
                ablation_config="A5",
            )
            diag = resp.get("diagnostics", {})
            all_cands = diag.get("all_candidate_moments", [])[:max_candidates_per_query]
            if not all_cands:
                missing_positive_query_ids.append(qid)
                continue
            queries_with_candidates += 1

            vis_cands = diag.get("visual_candidates", [])
            ocr_cands = diag.get("ocr_candidates", [])
            asr_cands = diag.get("asr_candidates", [])

            z_stats = compute_query_z_scores(
                [float(c["score"]) for c in vis_cands],
                top_k=120,
            )

            vis_scores_by_vid: dict[str, list[float]] = defaultdict(list)
            rrf_vis_by_vid: dict[str, float] = {}
            for rank, c in enumerate(vis_cands, start=1):
                vid = str(c["video_id"])
                vis_scores_by_vid[vid].append(float(c["score"]))
                if vid not in rrf_vis_by_vid:
                    rrf_vis_by_vid[vid] = 1.0 / (60.0 + rank)

            rrf_ocr_by_vid: dict[str, float] = {}
            fts_ocr_by_vid: dict[str, float] = {}
            for rank, c in enumerate(ocr_cands, start=1):
                vid = str(c["video_id"])
                if vid not in rrf_ocr_by_vid:
                    rrf_ocr_by_vid[vid] = 1.0 / (60.0 + rank)
                fts_ocr_by_vid[vid] = max(fts_ocr_by_vid.get(vid, 0.0), float(c.get("score", 0.0)))

            rrf_asr_by_vid: dict[str, float] = {}
            fts_asr_by_vid: dict[str, float] = {}
            for rank, c in enumerate(asr_cands, start=1):
                vid = str(c["video_id"])
                if vid not in rrf_asr_by_vid:
                    rrf_asr_by_vid[vid] = 1.0 / (60.0 + rank)
                fts_asr_by_vid[vid] = max(fts_asr_by_vid.get(vid, 0.0), float(c.get("score", 0.0)))

            has_pos = any(m.video_id in target_set for m in all_cands)
            has_neg = any(m.video_id not in target_set for m in all_cands)
            if has_pos:
                queries_with_positive += 1
            else:
                missing_positive_query_ids.append(qid)
            if not (has_pos and has_neg):
                continue
            queries_with_pairable += 1

            cand_ltr_map = diag.get("candidate_ltr_features", {})
            z_stats = diag.get("z_stats")
            if z_stats is None:
                z_stats = compute_query_z_scores(
                    [float(c["score"]) for c in vis_cands],
                    top_k=120,
                )

            for m in all_cands:
                vid = m.video_id
                if vid in cand_ltr_map:
                    cand_dict = cand_ltr_map[vid]
                else:
                    v_scores = sorted(vis_scores_by_vid.get(vid, ()), reverse=True)
                    s_max = v_scores[0] if v_scores else 0.0
                    top3 = sum(v_scores[:3]) / len(v_scores[:3]) if v_scores else s_max
                    top5 = sum(v_scores[:5]) / len(v_scores[:5]) if v_scores else s_max
                    spike_drop = (v_scores[0] - v_scores[1]) if len(v_scores) >= 2 else 0.0
                    has_lex = ("ocr" in m.match_status) or ("asr" in m.match_status)
                    cand_dict = {
                        "s_vis_blend_max": s_max,
                        "s_vis_top3_mean": top3,
                        "s_vis_top5_mean": top5,
                        "s_vis_ctx_coherence": s_max,
                        "s_late_interaction": 0.0,
                        "spike_drop_ratio": spike_drop,
                        "rrf_vis": rrf_vis_by_vid.get(vid, 0.0),
                        "rrf_ocr": rrf_ocr_by_vid.get(vid, 0.0),
                        "rrf_asr": rrf_asr_by_vid.get(vid, 0.0),
                        "fts_score": max(fts_ocr_by_vid.get(vid, 0.0), fts_asr_by_vid.get(vid, 0.0)),
                        "m_high": 1.0 if has_lex else 0.0,
                        "k_high": 1.0,
                        "anchor_lexical_match": 1.0 if has_lex else 0.0,
                        "number_consistency": 1.0,
                        "yolo_object_verification_boost": 0.0,
                        "graph_relation_score": 0.0,
                        "llm_modality_alignment": 1.0 if has_lex else 0.0,
                    }
                if int(feature_dim) == len(LTR_FEATURE_NAMES_V2):
                    vec = extract_ltr_features_32d(cand_dict, z_stats=z_stats, return_dict=False)
                else:
                    vec = extract_ltr_features_24d(cand_dict, z_stats=z_stats, return_dict=False)
                lbl = 1.0 if vid in target_set else 0.0
                base_sc = float(m.ranking_score)

                rows.append(vec)
                labels.append(lbl)
                query_ids.append(qid)
                source_video_ids.append(src_vid)
                base_scores.append(base_sc)
                if save_jsonl_path is not None:
                    jsonl_records.append({
                        "query_id": qid,
                        "source_video_id": src_vid,
                        "video_id": vid,
                        "label": lbl,
                        "base_video_score": base_sc,
                        "features": [float(v) for v in vec],
                    })

            if idx % 50 == 0 or idx == len(queries):
                print(
                    f"[LTR Extract] Processed {idx}/{len(queries)} dev_full_859 queries ({len(rows)} candidate rows)",
                    flush=True,
                )
    finally:
        pipeline.close()

    if not rows:
        raise ValueError("Extracted 0 candidate rows from dev_full_859")

    if save_jsonl_path is not None:
        save_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with save_jsonl_path.open("w", encoding="utf-8") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec) + "\n")

    coverage_stats = {
        "total_queries": len(queries),
        "queries_with_candidates": queries_with_candidates,
        "queries_with_positive_in_candidates": queries_with_positive,
        "queries_with_pairable_candidates": queries_with_pairable,
        "missing_positive_count": len(missing_positive_query_ids),
        "missing_positive_query_ids": sorted(missing_positive_query_ids),
        "candidate_recall": round(queries_with_positive / len(queries), 4),
    }
    extraction_meta = {
        "manifest_hash": manifest_hash,
        "source_group_rule": SOURCE_GROUP_RULE,
        "raw_groups_by_query": raw_groups_by_query,
        "group_stats": group_stats,
        "coverage_stats": coverage_stats,
    }
    print(
        f"[LTR Extract] Coverage: {queries_with_positive}/{len(queries)} positive in top-{max_candidates_per_query} "
        f"(recall={coverage_stats['candidate_recall']:.4f}, pairable={queries_with_pairable}, "
        f"missing_pos={len(missing_positive_query_ids)}) | "
        f"Groups: {group_stats['unique_source_groups']} source groups from {group_stats['unique_clips']} clips "
        f"(unknown={len(group_stats['unknown_clips'])})",
        flush=True,
    )

    X_mat = np.vstack(rows).astype(np.float64)
    y_arr = np.asarray(labels, dtype=np.float64)
    result: tuple[Any, ...] = (X_mat, y_arr, query_ids, source_video_ids)
    if return_base_scores:
        result = result + (np.asarray(base_scores, dtype=np.float64),)
    if return_metadata:
        result = result + (extraction_meta,)
    return result


def save_ltr_model_json(artifact: dict[str, Any], output_path: Path) -> Path:
    """Persist trained LTR artifact to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VIDENTRA LTR ranker with 5-Fold GroupKFold CV")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to dev_full_859 LTR JSONL")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR, help="Validation directory for dev_full_859 extraction")
    parser.add_argument("--extract-from-val", action="store_true", help="Extract 24D features directly from validation/ (dev_full_859)")
    parser.add_argument("--save-dataset", action="store_true", help="Persist extracted JSONL to --dataset when extracting from validation/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON weights path")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--l2-reg", type=float, default=250000.0, help="L2 regularization strength")
    parser.add_argument("--blend-weight", type=float, default=0.015, help="Residual blend weight matching pipeline inference")
    parser.add_argument("--feature-dim", type=int, choices=(24, 32), default=32, help="Feature schema dimensionality when extracting")
    parser.add_argument("--backend", choices=("auto", "pairwise_logistic", "lightgbm"), default="pairwise_logistic")
    args = parser.parse_args()

    extraction_meta: dict[str, Any] = {}
    if args.extract_from_val or (args.dataset == DEFAULT_DATASET_PATH and not args.dataset.is_file()):
        save_path = args.dataset if args.save_dataset else None
        X, y, query_ids, source_video_ids, base_scores, extraction_meta = extract_dev859_ltr_dataset(
            val_dir=args.val_dir,
            save_jsonl_path=save_path,
            return_base_scores=True,
            return_metadata=True,
            feature_dim=args.feature_dim,
        )
    else:
        X, y, query_ids, source_video_ids, base_scores = load_ltr_jsonl_dataset(
            args.dataset,
            return_base_scores=True,
        )
    artifact = run_group_kfold_cv(
        X=X,
        y=y,
        query_ids=query_ids,
        source_video_ids=source_video_ids,
        n_splits=args.n_splits,
        l2_reg=args.l2_reg,
        backend=args.backend,
        base_scores=base_scores,
        blend_weight=args.blend_weight,
        manifest_hash=extraction_meta.get("manifest_hash"),
        source_group_rule=extraction_meta.get("source_group_rule", SOURCE_GROUP_RULE),
        coverage_stats=extraction_meta.get("coverage_stats"),
        group_stats=extraction_meta.get("group_stats"),
        raw_groups_by_query=extraction_meta.get("raw_groups_by_query"),
    )
    out = save_ltr_model_json(artifact, args.output)
    meta = artifact["metadata"]
    print(
        f"Saved {artifact['model_type']} to {out} | "
        f"5-Fold CV Hit@1={meta['cv_hit_at_1']:.4f}, MRR={meta['cv_mrr']:.4f}, nDCG@10={meta['cv_ndcg_at_10']:.4f}"
    )


if __name__ == "__main__":
    main()
