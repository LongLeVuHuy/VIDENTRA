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
    QueryZScoreResult,
    extract_ltr_features_batch,
)

DEFAULT_LTR_MODEL_PATH = Path("models/ranking/videntra_ltr_v1.json")

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
}


class LearnedRankerLTR:
    """Pure-NumPy LTR ranker over the 24 continuous multimodal features."""

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
        self.fold_weights: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

        if self.model_path is not None and self.model_path.is_file():
            self.load(self.model_path)

    def _apply_payload(self, payload: Mapping[str, Any], *, src_path: Path | None = None, update_metadata: bool = True) -> None:
        """Validate and apply a linear/logistic or tree-ensemble payload onto this ranker."""
        names = tuple(payload.get("feature_names", self.feature_names or LTR_FEATURE_NAMES))
        if len(names) not in (18, 22, self.n_features):
            raise ValueError(
                f"Expected {self.n_features} (or legacy 18/22) features in LTR model, got {len(names)}"
            )
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
            "weights": self.weights.copy(),
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "feature_mean": self.feature_mean.copy(),
            "feature_std": self.feature_std.copy(),
            "trees": list(self.trees),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a previously captured snapshot created by snapshot_state()."""
        self.is_loaded = bool(state["is_loaded"])
        self.model_type = str(state["model_type"])
        if "feature_names" in state:
            self.feature_names = tuple(state["feature_names"])
        self.weights = np.asarray(state["weights"], dtype=np.float64).copy()
        self.bias = float(state["bias"])
        self.learning_rate = float(state.get("learning_rate", 1.0))
        self.feature_mean = np.asarray(state["feature_mean"], dtype=np.float64).copy()
        self.feature_std = np.asarray(state["feature_std"], dtype=np.float64).copy()
        self.trees = list(state["trees"])

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
        if k_cols not in (18, 22, self.n_features) or k_cols > self.n_features:
            raise ValueError(f"Expected {self.n_features} (or legacy 18/22) feature columns, got {k_cols}")

        X = np.where(np.isfinite(X), X, 0.0)

        if self.model_type in {"tree_ensemble", "gbdt", "lightgbm"} and self.trees:
            if k_cols < self.n_features:
                X_eval = np.zeros((X.shape[0], self.n_features), dtype=np.float64)
                X_eval[:, :k_cols] = X
            else:
                X_eval = X
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
        """Extract 18D features for candidates and return their LTR scores."""
        if not candidates:
            return np.zeros(0, dtype=np.float64)
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
