"""Lightweight pure-NumPy Learning-to-Rank (LTR) scorer for VIDENTRA Stage 6.

Supports:
- L2-regularized pairwise logistic / linear weights with optional feature standardization.
- Optional compact GBDT / LightGBM tree-ensemble JSON representation.
- Graceful fallback to calibrated default weights when no trained model artifact exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from videntra.ranking.features import (
    LTR_FEATURE_NAMES,
    LTR_FEATURE_NAMES_V2,
    QueryZScoreResult,
    extract_ltr_features_batch,
    extract_ltr_features_batch_32d,
)

DEFAULT_LTR_MODEL_PATH = Path("models/ranking/videntra_ltr.json")
_SUPPORTED_FEATURE_DIMS = frozenset({18, 22, 24, 32})

# Sensible prior weights aligned with Plan §4.2 when trained artifact is absent.
_FALLBACK_LINEAR_WEIGHTS: dict[str, float] = {
    "s_vis_blend_max": 1.00,
    "z_vis_max": 0.35,
    "delta_z_vis": -0.45,
    "s_vis_top3_mean": 0.30,
    "s_vis_top5_mean": 0.25,
    "s_vis_ctx_coherence": 0.20,
    "s_late_interaction": 0.40,
    "spike_drop_ratio": -0.25,
    "rrf_vis": 0.45,
    "rrf_ocr": 0.30,
    "rrf_asr": 0.25,
    "bm25_log_score": 0.15,
    "high_spec_token_coverage": 0.35,
    "anchor_lexical_match": 0.40,
    "number_consistency": 0.35,
    "yolo_object_verification_boost": 0.25,
    "graph_relation_score": 0.30,
    "llm_modality_alignment": 0.20,
    "vis_lexical_synergy": 0.35,
    "unverified_lexical_hazard": -0.45,
    "subclause_weakest_link": 0.30,
    "window_energy_concentration": 0.20,
    "temporal_attention_coherence": 0.25,
    "rrf_vis_caption": 0.25,
    "vp_whole_cos": 0.50,
    "vp_subclip_max": 0.40,
    "vp_z": 0.45,
    "vp_csls": 0.30,
    "rrf_vp": 0.45,
    "lex_reliability_score": 0.25,
    "spatial_predicate_score": 0.20,
    "action_order_score": 0.25,
}


def _compile_tree_ensemble(
    trees: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Flatten nested dict decision trees into contiguous 1D NumPy tables for fast traversal."""
    n_trees = len(trees)
    if n_trees == 0:
        return None

    def _tree_shape(root: Mapping[str, Any]) -> tuple[int, int]:
        max_d = 0
        count = 0
        stack: list[tuple[Mapping[str, Any], int]] = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > max_d:
                max_d = depth
            if "leaf_value" not in node:
                stack.append((node["right"], depth + 1))
                stack.append((node["left"], depth + 1))
        return count, max_d

    shapes = [_tree_shape(t) for t in trees]
    max_nodes = max(c for c, _ in shapes)
    max_depth = max(1, max(d for _, d in shapes))
    total_nodes = n_trees * max_nodes

    feat_flat = np.full(total_nodes, -1, dtype=np.int32)
    thresh_flat = np.zeros(total_nodes, dtype=np.float64)
    left_flat = np.arange(total_nodes, dtype=np.int32)
    right_flat = np.arange(total_nodes, dtype=np.int32)
    leaf_flat = np.zeros(total_nodes, dtype=np.float64)

    for ti, root in enumerate(trees):
        base = ti * max_nodes
        next_id = 1
        stack_fill: list[tuple[Mapping[str, Any], int]] = [(root, 0)]
        while stack_fill:
            node, nid = stack_fill.pop()
            flat_nid = base + nid
            if "leaf_value" in node:
                feat_flat[flat_nid] = -1
                left_flat[flat_nid] = flat_nid
                right_flat[flat_nid] = flat_nid
                leaf_flat[flat_nid] = float(node["leaf_value"])
            else:
                l_id = next_id
                r_id = next_id + 1
                next_id += 2
                feat_flat[flat_nid] = int(node["feature_idx"])
                thresh_flat[flat_nid] = float(node["threshold"])
                left_flat[flat_nid] = base + l_id
                right_flat[flat_nid] = base + r_id
                stack_fill.append((node["right"], r_id))
                stack_fill.append((node["left"], l_id))

    tree_offsets = (np.arange(n_trees, dtype=np.int32) * max_nodes)[None, :]
    return feat_flat, thresh_flat, left_flat, right_flat, leaf_flat, tree_offsets, max_depth


class LearnedRankerLTR:
    """Pure-NumPy LTR ranker over the 24/32 continuous multimodal features."""

    def __init__(self, model_path: str | Path | None = DEFAULT_LTR_MODEL_PATH) -> None:
        self.model_path: Path | None = Path(model_path) if model_path is not None else None
        self.is_loaded: bool = False
        self.model_type: str = "fallback_linear"
        self.feature_names: tuple[str, ...] = LTR_FEATURE_NAMES
        self.n_features: int = len(LTR_FEATURE_NAMES)
        self.feature_mean: np.ndarray = np.zeros(self.n_features, dtype=np.float64)
        self.feature_std: np.ndarray = np.ones(self.n_features, dtype=np.float64)
        self.weights: np.ndarray = np.array(
            [_FALLBACK_LINEAR_WEIGHTS[name] for name in LTR_FEATURE_NAMES],
            dtype=np.float64,
        )
        self.bias: float = 0.0
        self.learning_rate: float = 1.0
        self.trees: list[dict[str, Any]] = []
        self._compiled_trees: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
        self.fold_weights: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

        if self.model_path is not None and self.model_path.is_file():
            self.load(self.model_path)

    def _apply_payload(self, payload: Mapping[str, Any], *, src_path: Path | None = None, update_metadata: bool = True) -> None:
        """Validate and apply a linear/logistic or tree-ensemble payload onto this ranker."""
        names = tuple(payload.get("feature_names", self.feature_names or LTR_FEATURE_NAMES))
        if len(names) not in _SUPPORTED_FEATURE_DIMS:
            raise ValueError(
                f"Expected 18, 22, 24, or 32 features in LTR model, got {len(names)}"
            )
        if len(names) == len(LTR_FEATURE_NAMES_V2):
            self.n_features = len(LTR_FEATURE_NAMES_V2)
            self.feature_names = names
            legacy_prefix_len = 0
        else:
            self.n_features = len(LTR_FEATURE_NAMES)
            legacy_prefix_len = len(names) if len(names) < self.n_features else 0
            self.feature_names = LTR_FEATURE_NAMES if legacy_prefix_len else names

        default_type = "tree_ensemble" if "trees" in payload else "pairwise_logistic"
        model_type = str(payload.get("model_type", default_type)).lower()
        self.model_type = model_type
        if update_metadata:
            self.fold_weights = list(payload.get("fold_weights", []))
            self.metadata = dict(payload.get("metadata", {}))

        mean_arr = payload.get("feature_mean")
        std_arr = payload.get("feature_std")
        if mean_arr is not None:
            raw_mean = np.asarray(mean_arr, dtype=np.float64)
            if raw_mean.shape != (len(names),):
                raise ValueError(
                    f"Expected {len(names)} feature_mean entries, got {raw_mean.shape}"
                )
            if legacy_prefix_len:
                padded_mean = np.zeros(self.n_features, dtype=np.float64)
                padded_mean[:legacy_prefix_len] = np.where(np.isfinite(raw_mean), raw_mean, 0.0)
                self.feature_mean = padded_mean
            else:
                self.feature_mean = np.where(np.isfinite(raw_mean), raw_mean, 0.0)
        else:
            self.feature_mean = np.zeros(self.n_features, dtype=np.float64)
        if std_arr is not None:
            raw_std = np.asarray(std_arr, dtype=np.float64)
            if raw_std.shape != (len(names),):
                raise ValueError(
                    f"Expected {len(names)} feature_std entries, got {raw_std.shape}"
                )
            if legacy_prefix_len:
                padded_std = np.ones(self.n_features, dtype=np.float64)
                padded_std[:legacy_prefix_len] = np.where(np.isfinite(raw_std) & (raw_std > 1e-8), raw_std, 1.0)
                self.feature_std = padded_std
            else:
                self.feature_std = np.where(np.isfinite(raw_std) & (raw_std > 1e-8), raw_std, 1.0)
        else:
            self.feature_std = np.ones(self.n_features, dtype=np.float64)

        self.bias = float(payload.get("bias", payload.get("base_score", 0.0)))
        self.learning_rate = float(payload.get("learning_rate", 1.0))

        if model_type in {"tree_ensemble", "gbdt", "lightgbm"}:
            trees = payload.get("trees")
            if not isinstance(trees, list) or not trees:
                raise ValueError("tree_ensemble model JSON must contain a non-empty 'trees' list")
            self.trees = list(trees)
            self._compiled_trees = _compile_tree_ensemble(self.trees)
            self.weights = np.zeros(self.n_features, dtype=np.float64)
        else:
            raw_weights = payload.get("weights")
            if raw_weights is None or len(raw_weights) != len(names):
                raise ValueError(
                    f"Linear/logistic LTR model must provide {len(names)} weights"
                )
            if legacy_prefix_len:
                padded_w = np.zeros(self.n_features, dtype=np.float64)
                padded_w[:legacy_prefix_len] = np.asarray(raw_weights, dtype=np.float64)
                self.weights = padded_w
            else:
                self.weights = np.asarray(raw_weights, dtype=np.float64)
            self.trees = []
            self._compiled_trees = None

        if src_path is not None:
            self.model_path = src_path
        self.is_loaded = True

    def load(self, path: str | Path) -> None:
        """Load linear/logistic or tree-ensemble LTR weights from a JSON file."""
        src = Path(path)
        payload = json.loads(src.read_text(encoding="utf-8"))
        self._apply_payload(payload, src_path=src, update_metadata=True)

    def snapshot_state(self) -> dict[str, Any]:
        """Capture a deep snapshot of mutable ranker state for temporary OOF fold swaps."""
        return {
            "is_loaded": self.is_loaded,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "weights": self.weights.copy(),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "feature_mean": self.feature_mean.copy(),
            "feature_std": self.feature_std.copy(),
            "trees": list(self.trees),
            "_compiled_trees": self._compiled_trees,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a previously captured snapshot created by snapshot_state()."""
        self.is_loaded = bool(state["is_loaded"])
        self.model_type = str(state["model_type"])
        if "feature_names" in state:
            self.feature_names = tuple(state["feature_names"])
        self.n_features = int(state.get("n_features", len(self.feature_names)))
        self.weights = np.asarray(state["weights"], dtype=np.float64).copy()
        self.bias = float(state["bias"])
        self.learning_rate = float(state.get("learning_rate", 1.0))
        self.feature_mean = np.asarray(state["feature_mean"], dtype=np.float64).copy()
        self.feature_std = np.asarray(state["feature_std"], dtype=np.float64).copy()
        self.trees = list(state["trees"])
        self._compiled_trees = (
            state["_compiled_trees"]
            if "_compiled_trees" in state
            else _compile_tree_ensemble(self.trees)
        )

    def apply_fold_payload(self, fold_payload: Mapping[str, Any]) -> None:
        """Validate and temporarily apply an OOF fold payload while keeping artifact metadata intact."""
        payload = dict(fold_payload)
        payload.setdefault("feature_names", list(self.feature_names))
        self._apply_payload(payload, src_path=None, update_metadata=False)

    def _predict_tree_batch(self, tree: dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Evaluate one compact decision tree over all rows in X."""
        n_rows = X.shape[0]
        preds = np.empty(n_rows, dtype=np.float64)

        def _eval(node: dict[str, Any], mask: np.ndarray) -> None:
            if not np.any(mask):
                return
            if "leaf_value" in node:
                preds[mask] = float(node["leaf_value"])
                return
            f_idx = int(node["feature_idx"])
            thresh = float(node["threshold"])
            go_left = mask & (X[:, f_idx] <= thresh)
            go_right = mask & (~go_left)
            if np.any(go_left):
                _eval(node["left"], go_left)
            if np.any(go_right):
                _eval(node["right"], go_right)

        _eval(tree, np.ones(n_rows, dtype=bool))
        return preds

    def score_matrix(self, features: np.ndarray) -> np.ndarray:
        """Score an (N, 18) or (18,) feature matrix in pure NumPy."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            if X.size == 0:
                return np.zeros(0, dtype=np.float64)
            X = X.reshape(1, -1)
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        k_cols = X.shape[1]
        if k_cols not in _SUPPORTED_FEATURE_DIMS or k_cols > self.n_features:
            raise ValueError(f"Expected {self.n_features} (or compatible legacy) feature columns, got {k_cols}")

        X = np.where(np.isfinite(X), X, 0.0)

        if self.model_type in {"tree_ensemble", "gbdt", "lightgbm"} and self.trees:
            if k_cols < self.n_features:
                X_eval = np.zeros((X.shape[0], self.n_features), dtype=np.float64)
                X_eval[:, :k_cols] = X
            else:
                X_eval = X
            if self._compiled_trees is not None and len(self._compiled_trees[5][0]) == len(self.trees):
                feat_flat, thresh_flat, left_flat, right_flat, leaf_flat, tree_offsets, max_depth = self._compiled_trees
                n_rows = X_eval.shape[0]
                n_trees = tree_offsets.shape[1]
                cur = np.repeat(tree_offsets, n_rows, axis=0)
                x_flat = np.ascontiguousarray(X_eval, dtype=np.float64).ravel()
                row_offsets = (np.arange(n_rows, dtype=np.int32) * self.n_features)[:, None]
                for _ in range(max_depth):
                    f = feat_flat.take(cur)
                    vals = x_flat.take(row_offsets + np.maximum(0, f))
                    cur = np.where(vals <= thresh_flat.take(cur), left_flat.take(cur), right_flat.take(cur))
                return self.bias + self.learning_rate * np.sum(leaf_flat.take(cur), axis=1)
            scores = np.full(X_eval.shape[0], self.bias, dtype=np.float64)
            for tree in self.trees:
                scores += self.learning_rate * self._predict_tree_batch(tree, X_eval)
            return scores

        X_norm = (X - self.feature_mean[:k_cols]) / self.feature_std[:k_cols]
        return (X_norm @ self.weights[:k_cols]) + self.bias

    def score_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Extract features matching ranker schema and return LTR scores."""
        if not candidates:
            return np.zeros(0, dtype=np.float64)
        if self.n_features == len(LTR_FEATURE_NAMES_V2):
            X = extract_ltr_features_batch_32d(candidates, z_stats=z_stats)
        else:
            X = extract_ltr_features_batch(candidates, z_stats=z_stats)
        return self.score_matrix(X)

    def score_one(
        self,
        candidate: np.ndarray | Mapping[str, Any],
        z_stats: QueryZScoreResult | Mapping[str, float] | None = None,
    ) -> float:
        """Score a single candidate feature vector or signal mapping."""
        if isinstance(candidate, Mapping):
            scores = self.score_candidates([candidate], z_stats=z_stats)
        else:
            scores = self.score_matrix(np.asarray(candidate, dtype=np.float64))
        return float(scores[0]) if scores.size > 0 else 0.0
