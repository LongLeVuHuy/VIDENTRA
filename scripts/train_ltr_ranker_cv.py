"""5-Fold GroupKFold Cross-Validation trainer for VIDENTRA LearnedRankerLTR.

Trains an L2-regularized pairwise logistic ranker (or LightGBM LambdaMART if
installed and requested) on ``dev_full_859`` grouped strictly by
``source_video_id`` to prevent video-level leakage. Never touches
``heldout_test_201`` or ``test_queries/``.

Saves the compact JSON artifact to ``models/ranking/videntra_ltr_v1.json``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from videntra.ranking.features import (
    LTR_FEATURE_NAMES,
    compute_query_z_scores,
    extract_ltr_features_22d,
)

DEFAULT_DATASET_PATH = Path("work/ltr_dev859_features.jsonl")
DEFAULT_VAL_DIR = Path("validation")
DEFAULT_OUTPUT_PATH = Path("models/ranking/videntra_ltr_v1.json")

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
})

NEGATIVE_MONOTONIC_FEATURES: frozenset[str] = frozenset({
    "delta_z_vis",
    "spike_drop_ratio",
    "unverified_lexical_hazard",
})


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
) -> tuple[np.ndarray, np.ndarray]:
    """Construct balanced pairwise feature differences (x_pos - x_neg) per query."""
    by_query: dict[str, list[int]] = defaultdict(list)
    for idx, qid in enumerate(query_ids):
        by_query[str(qid)].append(idx)

    diffs: list[np.ndarray] = []
    labels: list[float] = []

    for idxs in by_query.values():
        pos_idxs = [i for i in idxs if y[i] > 0]
        neg_idxs = [i for i in idxs if y[i] <= 0]
        if not pos_idxs or not neg_idxs:
            continue

        sampled_negs = neg_idxs[:max_neg_per_pos]
        for p in pos_idxs:
            for n in sampled_negs:
                delta = X[p] - X[n]
                diffs.append(delta)
                labels.append(1.0)
                diffs.append(-delta)
                labels.append(0.0)

    if not diffs:
        return np.zeros((0, X.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.vstack(diffs).astype(np.float64), np.asarray(labels, dtype=np.float64)


def _build_monotonic_masks(
    n_features: int,
    feature_names: Sequence[str] = LTR_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean masks for positive (>= 0) and negative (<= 0) monotonic constraints."""
    pos_mask = np.zeros(n_features, dtype=bool)
    neg_mask = np.zeros(n_features, dtype=bool)
    for idx, name in enumerate(feature_names[:n_features]):
        if name in POSITIVE_MONOTONIC_FEATURES:
            pos_mask[idx] = True
        elif name in NEGATIVE_MONOTONIC_FEATURES:
            neg_mask[idx] = True
    return pos_mask, neg_mask


def train_pairwise_logistic_l2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_ids_train: Sequence[str],
    l2_reg: float = 1.0,
    learning_rate: float = 0.08,
    max_iter: int = 400,
    monotonic_constraints: bool = True,
    feature_names: Sequence[str] = LTR_FEATURE_NAMES,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Train L2-regularized pairwise logistic regression in pure NumPy.

    Applies Monotonic Sign Projection after each Adam step when
    ``monotonic_constraints=True`` so physical feature signs cannot invert.

    Returns:
        (weights, bias, feature_mean, feature_std)
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    feat_mean = np.mean(X, axis=0)
    raw_std = np.std(X, axis=0)
    feat_std = np.where(raw_std > 1e-8, raw_std, 1.0)
    X_norm = (X - feat_mean) / feat_std

    X_pair, y_pair = build_pairwise_differences(X_norm, y, query_ids_train)
    n_features = X.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    if X_pair.shape[0] == 0:
        return weights, 0.0, feat_mean, feat_std

    pos_mask, neg_mask = _build_monotonic_masks(n_features, feature_names=feature_names)
    n_pairs = float(X_pair.shape[0])
    # Adam optimizer for fast, deterministic convergence without SciPy
    m = np.zeros(n_features, dtype=np.float64)
    v = np.zeros(n_features, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, max_iter + 1):
        logits = np.clip(X_pair @ weights, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (X_pair.T @ (probs - y_pair)) / n_pairs + (l2_reg / n_pairs) * weights

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
) -> dict[str, Any] | None:
    """Optional LightGBM LambdaMART training if lightgbm is installed."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    # Group rows contiguously by query_id for LGBMRanker
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

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=40,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.05,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    ranker.fit(X_ord, y_ord, group=group_counts)

    def _convert_tree(node: dict[str, Any]) -> dict[str, Any]:
        if "leaf_value" in node:
            return {"leaf_value": float(node["leaf_value"])}
        return {
            "feature_idx": int(node["split_feature"]),
            "threshold": float(node["threshold"]),
            "left": _convert_tree(node["left_child"]),
            "right": _convert_tree(node["right_child"]),
        }

    dump = ranker.booster_.dump_model()
    trees = [_convert_tree(t["tree_structure"]) for t in dump.get("tree_info", [])]
    return {
        "model_type": "tree_ensemble",
        "trees": trees,
        "bias": 0.0,
        "learning_rate": 1.0,
        "ranker": ranker,
    }


def run_group_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    query_ids: Sequence[str],
    source_video_ids: Sequence[str],
    n_splits: int = 5,
    l2_reg: float = 1.0,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run 5-Fold GroupKFold CV grouped by source_video_id and build final artifact."""
    splits = group_kfold_indices(source_video_ids, n_splits=n_splits)

    fold_metrics: list[dict[str, float]] = []
    fold_weights: list[np.ndarray] = []

    use_lgb = backend == "lightgbm"
    if backend == "auto":
        try:
            import lightgbm  # noqa: F401
            use_lgb = True
        except ImportError:
            use_lgb = False

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        q_tr = [query_ids[i] for i in tr_idx]
        q_val = [query_ids[i] for i in val_idx]

        # Verify zero video leakage between train and val folds
        tr_vids = {source_video_ids[i] for i in tr_idx}
        val_vids = {source_video_ids[i] for i in val_idx}
        assert tr_vids.isdisjoint(val_vids), f"GroupKFold video leakage in fold {fold_idx}"

        w, bias, f_mean, f_std = train_pairwise_logistic_l2(
            X_tr, y_tr, q_tr, l2_reg=l2_reg
        )
        fold_weights.append(w)

        val_scores = ((X_val - f_mean) / f_std) @ w + bias
        metrics = evaluate_ranking_metrics(val_scores, y_val, q_val)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    # Fit final model on full dev_full_859 dataset
    mean_cv_hit1 = float(np.mean([m["hit_at_1"] for m in fold_metrics]))
    mean_cv_mrr = float(np.mean([m["mrr"] for m in fold_metrics]))
    mean_cv_ndcg10 = float(np.mean([m["ndcg_at_10"] for m in fold_metrics]))

    lgb_payload = _train_lightgbm_fold(X, y, query_ids) if use_lgb else None
    if lgb_payload is not None:
        artifact: dict[str, Any] = {
            "model_type": "tree_ensemble",
            "feature_names": list(LTR_FEATURE_NAMES),
            "trees": lgb_payload["trees"],
            "bias": 0.0,
            "learning_rate": 1.0,
        }
    else:
        final_w, final_bias, final_mean, final_std = train_pairwise_logistic_l2(
            X, y, query_ids, l2_reg=l2_reg
        )
        artifact = {
            "model_type": "pairwise_logistic",
            "feature_names": list(LTR_FEATURE_NAMES),
            "weights": [round(float(v), 6) for v in final_w],
            "bias": round(float(final_bias), 6),
            "feature_mean": [round(float(v), 6) for v in final_mean],
            "feature_std": [round(float(v), 6) for v in final_std],
        }

    artifact["metadata"] = {
        "dataset": "dev_full_859",
        "cv_strategy": f"{n_splits}-fold GroupKFold",
        "grouped_by": "source_video_id",
        "l2_reg": float(l2_reg),
        "cv_hit_at_1": round(mean_cv_hit1, 4),
        "cv_mrr": round(mean_cv_mrr, 4),
        "cv_ndcg_at_10": round(mean_cv_ndcg10, 4),
        "fold_metrics": fold_metrics,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return artifact


def load_ltr_jsonl_dataset(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load pre-extracted candidate rows from JSONL (no held-out/test paths)."""
    assert_no_heldout_leakage(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"LTR training dataset not found: {dataset_path}")

    rows: list[np.ndarray] = []
    labels: list[float] = []
    query_ids: list[str] = []
    source_video_ids: list[str] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "features" in item and isinstance(item["features"], list) and len(item["features"]) == len(LTR_FEATURE_NAMES):
                vec = np.asarray(item["features"], dtype=np.float32)
            else:
                vec = extract_ltr_features_22d(item, return_dict=False)
            rows.append(vec)
            labels.append(float(item.get("label", item.get("is_relevant", 0))))
            query_ids.append(str(item["query_id"]))
            source_video_ids.append(str(item.get("source_video_id", item.get("video_id", ""))))

    if not rows:
        raise ValueError(f"Empty LTR dataset: {dataset_path}")

    return (
        np.vstack(rows).astype(np.float64),
        np.asarray(labels, dtype=np.float64),
        query_ids,
        source_video_ids,
    )


def extract_dev859_ltr_dataset(
    val_dir: Path = DEFAULT_VAL_DIR,
    save_jsonl_path: Path | None = None,
    max_candidates_per_query: int = 60,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract 18D LTR features from dev_full_859 (validation/ only, never held-out test)."""
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
        db_path=Path("data/processed/metadata.db"),
        faiss_path=Path("data/processed/faiss_index.bin"),
        model_dir=Path("models/siglip2"),
        device=runtime_device,
        use_disk_cache=True,
        memory_cache_size=1024,
    )
    try:
        indexed_videos = set(pipeline.video_durations.keys()) if pipeline.video_durations else set()
        if not indexed_videos and pipeline.keyframe_map:
            indexed_videos = {k["video_id"] for k in pipeline.keyframe_map.values()}

        queries = load_validation_queries(val_dir, indexed_videos, max_msrvtt=None)
        if not queries:
            raise ValueError(f"No validation queries loaded from {val_dir}")

        rows: list[np.ndarray] = []
        labels: list[float] = []
        query_ids: list[str] = []
        source_video_ids: list[str] = []
        jsonl_records: list[dict[str, Any]] = []

        for idx, q in enumerate(queries, start=1):
            qid = str(q["query_id"])
            target_videos = list(q["target_videos"])
            target_set = set(target_videos)
            src_vid = str(target_videos[0])

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
                continue

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
            if not (has_pos and has_neg):
                continue

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
                vec = extract_ltr_features_22d(cand_dict, z_stats=z_stats, return_dict=False)
                lbl = 1.0 if vid in target_set else 0.0

                rows.append(vec)
                labels.append(lbl)
                query_ids.append(qid)
                source_video_ids.append(src_vid)
                if save_jsonl_path is not None:
                    jsonl_records.append({
                        "query_id": qid,
                        "source_video_id": src_vid,
                        "video_id": vid,
                        "label": lbl,
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

    return (
        np.vstack(rows).astype(np.float64),
        np.asarray(labels, dtype=np.float64),
        query_ids,
        source_video_ids,
    )


def save_ltr_model_json(artifact: dict[str, Any], output_path: Path) -> Path:
    """Persist trained LTR artifact to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VIDENTRA LTR ranker with 5-Fold GroupKFold CV")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to dev_full_859 LTR JSONL")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR, help="Validation directory for dev_full_859 extraction")
    parser.add_argument("--extract-from-val", action="store_true", help="Extract 18D features directly from validation/ (dev_full_859)")
    parser.add_argument("--save-dataset", action="store_true", help="Persist extracted JSONL to --dataset when extracting from validation/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON weights path")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--l2-reg", type=float, default=3000.0, help="L2 regularization strength")
    parser.add_argument("--backend", choices=("auto", "pairwise_logistic", "lightgbm"), default="auto")
    args = parser.parse_args()

    if args.extract_from_val or (args.dataset == DEFAULT_DATASET_PATH and not args.dataset.is_file()):
        save_path = args.dataset if args.save_dataset else None
        X, y, query_ids, source_video_ids = extract_dev859_ltr_dataset(
            val_dir=args.val_dir,
            save_jsonl_path=save_path,
        )
    else:
        X, y, query_ids, source_video_ids = load_ltr_jsonl_dataset(args.dataset)
    artifact = run_group_kfold_cv(
        X=X,
        y=y,
        query_ids=query_ids,
        source_video_ids=source_video_ids,
        n_splits=args.n_splits,
        l2_reg=args.l2_reg,
        backend=args.backend,
    )
    out = save_ltr_model_json(artifact, args.output)
    meta = artifact["metadata"]
    print(
        f"Saved {artifact['model_type']} to {out} | "
        f"5-Fold CV Hit@1={meta['cv_hit_at_1']:.4f}, MRR={meta['cv_mrr']:.4f}, nDCG@10={meta['cv_ndcg_at_10']:.4f}"
    )


if __name__ == "__main__":
    main()
