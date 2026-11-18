"""Offline VL Reranker (Qwen3-VL-Reranker-2B) cache contract & Top-M cascade reranking."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

VL_CACHE_REQUIRED_FIELDS: tuple[str, ...] = (
    "query_text_sha256",
    "video_id",
    "media_sha256",
    "snapshot_sha256",
    "frame_ordinals",
    "frame_pts_time",
    "sampling_recipe",
    "score",
    "status",
)

FORBIDDEN_VL_FIELDS: frozenset[str] = frozenset({
    "target_videos",
    "target_windows",
    "relevant_windows",
    "source_group",
    "source_group_status",
    "known_target_window_id",
    "a5_rank",
    "base_rank",
    "rank_in_pool",
})

_SCORED_STATUSES = frozenset({"ok", "scored", "success"})


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _validate_vl_row_contract(
    row: dict[str, Any],
    *,
    row_idx: int,
    strict_contract: bool,
    expected_provenance: Mapping[str, Any] | None = None,
) -> None:
    forbidden = FORBIDDEN_VL_FIELDS.intersection(row.keys())
    if forbidden:
        raise ValueError(
            f"VL reranker cache row {row_idx} contains forbidden ground-truth/rank-prior field(s): "
            f"{sorted(forbidden)}"
        )
    if not strict_contract:
        return
    missing = [f for f in VL_CACHE_REQUIRED_FIELDS if f not in row]
    if missing:
        raise ValueError(f"VL reranker cache row {row_idx} missing required contract field(s): {missing}")
    has_rev = bool(row.get("model_revision") or (row.get("model") and row.get("processor_revision")))
    if not has_rev:
        raise ValueError(
            f"VL reranker cache row {row_idx} missing model/processor revision metadata"
        )
    has_code_hash = bool(row.get("code_sha256") or row.get("prompt_sha256"))
    if not has_code_hash:
        raise ValueError(
            f"VL reranker cache row {row_idx} missing code_sha256 or prompt_sha256"
        )

    ordinals_raw = row.get("frame_ordinals")
    pts_raw = row.get("frame_pts_time")
    if not isinstance(ordinals_raw, list) or not isinstance(pts_raw, list) or not ordinals_raw:
        raise ValueError(
            f"VL reranker cache row {row_idx} frame_ordinals and frame_pts_time must be non-empty lists"
        )
    if len(ordinals_raw) != len(pts_raw):
        raise ValueError(
            f"VL reranker cache row {row_idx} frame_ordinals/frame_pts_time length mismatch"
        )
    prev_ord = -1
    prev_pts = -math.inf
    for o_raw, p_raw in zip(ordinals_raw, pts_raw):
        if isinstance(o_raw, bool) or not isinstance(o_raw, int) or o_raw < 0:
            raise ValueError(f"VL reranker cache row {row_idx} has invalid frame_ordinal: {o_raw!r}")
        if o_raw <= prev_ord:
            raise ValueError(
                f"VL reranker cache row {row_idx} frame_ordinals must be strictly increasing: {ordinals_raw}"
            )
        prev_ord = o_raw
        p_val = float(p_raw)
        if not math.isfinite(p_val) or p_val < 0.0 or p_val < prev_pts:
            raise ValueError(
                f"VL reranker cache row {row_idx} frame_pts_time must be finite and non-decreasing: {pts_raw}"
            )
        prev_pts = p_val

    if "query_text" in row and row["query_text"] is not None:
        raw_q = str(row["query_text"])
        q_sha = str(row.get("query_text_sha256") or "")
        sha_utf8 = hashlib.sha256(raw_q.encode("utf-8")).hexdigest()
        sha_nfc = hashlib.sha256(unicodedata.normalize("NFC", raw_q.strip()).encode("utf-8")).hexdigest()
        if q_sha not in (sha_utf8, sha_nfc):
            raise ValueError(
                f"VL reranker cache row {row_idx} query_text_sha256 does not match query_text"
            )

    if expected_provenance is not None:
        for key in ("snapshot_sha256", "model_revision", "sampling_recipe", "prompt_sha256", "code_sha256"):
            expected_val = expected_provenance.get(key)
            if expected_val is not None and str(expected_val).strip():
                actual_val = str(row.get(key) or "")
                if actual_val != str(expected_val).strip():
                    raise ValueError(
                        f"VL reranker cache row {row_idx} {key} mismatch: "
                        f"expected {expected_val!r}, got {actual_val!r}"
                    )
        media_map = expected_provenance.get("media_sha256_by_video")
        if isinstance(media_map, Mapping):
            vid = str(row.get("video_id") or "")
            if vid in media_map and str(media_map[vid]).strip():
                actual_msha = str(row.get("media_sha256") or "")
                if actual_msha != str(media_map[vid]).strip():
                    raise ValueError(
                        f"VL reranker cache row {row_idx} media_sha256 mismatch for video_id={vid}: "
                        f"expected {media_map[vid]!r}, got {actual_msha!r}"
                    )


def load_vl_reranker_cache(
    cache_path: Path,
    *,
    strict_contract: bool = True,
    strict: bool | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate offline VL reranker JSONL cache keyed strictly by query_text_sha256 and video_id."""
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"VL reranker cache not found: {path}")

    effective_strict = strict_contract if strict is None else bool(strict)
    by_qsha: dict[str, dict[str, dict[str, Any]]] = {}
    raw_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"VL reranker cache line {line_no} must be a JSON object")
            _validate_vl_row_contract(
                obj,
                row_idx=line_no,
                strict_contract=effective_strict,
                expected_provenance=expected_provenance,
            )
            q_sha = str(obj.get("query_text_sha256") or "")
            vid = str(obj.get("video_id") or "")
            if not q_sha or not vid:
                raise ValueError(f"VL reranker cache line {line_no} requires non-empty query_text_sha256 and video_id")
            q_bucket = by_qsha.setdefault(q_sha, {})
            if effective_strict and vid in q_bucket:
                raise ValueError(
                    f"Duplicate VL reranker cache pair (query_text_sha256={q_sha}, video_id={vid}) at line {line_no}"
                )
            raw_score = obj.get("score")
            parsed_score = float(raw_score) if raw_score is not None and math.isfinite(float(raw_score)) else None
            entry = {
                **obj,
                "score": parsed_score,
                "status": str(obj.get("status") or ""),
            }
            q_bucket[vid] = entry
            raw_rows.append(entry)

    return {
        "cache_path": str(path),
        "cache_sha256": sha256_file(path),
        "row_count": len(raw_rows),
        "by_qsha": by_qsha,
        "raw_rows": raw_rows,
    }


def verify_order_invariance(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify per-pair VL scores are invariant to candidate ordering / batching."""
    map_a = {
        (str(r["query_text_sha256"]), str(r["video_id"])): float(r["score"])
        for r in rows_a
        if r.get("score") is not None and str(r.get("status", "")).lower() in _SCORED_STATUSES
    }
    map_b = {
        (str(r["query_text_sha256"]), str(r["video_id"])): float(r["score"])
        for r in rows_b
        if r.get("score") is not None and str(r.get("status", "")).lower() in _SCORED_STATUSES
    }
    if set(map_a.keys()) != set(map_b.keys()):
        raise ValueError("Order-invariance pair set mismatch between VL runs")
    max_diff = 0.0
    for key, s_a in map_a.items():
        diff = abs(s_a - map_b[key])
        if diff > max_diff:
            max_diff = diff
        if diff > atol:
            raise ValueError(
                f"VL score order-invariance violation for {key}: |{s_a} - {map_b[key]}| = {diff} > {atol}"
            )
    return {
        "pair_count": len(map_a),
        "max_abs_diff": round(max_diff, 8),
        "invariant": True,
    }


def _extract_valid_vl_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    if isinstance(raw, dict):
        status = str(raw.get("status") or "ok").lower()
        if status not in _SCORED_STATUSES:
            return None
        sc = raw.get("score")
        if sc is None:
            return None
        val = float(sc)
        return val if math.isfinite(val) else None
    return None


def rerank_top_m_with_vl(
    base_ids: Sequence[str],
    vl_entries: dict[str, Any] | None,
    *,
    top_m: int = 5,
    mode: str = "pure",
    beta: float = 0.5,
) -> tuple[list[str], dict[str, Any]]:
    """Rerank up to top_m leading candidates using cached VL scores, leaving tail and unscored slots intact."""
    if mode not in ("pure", "rrf"):
        raise ValueError(f"Unsupported vl mode '{mode}'; expected 'pure' or 'rrf'")
    if top_m < 1:
        raise ValueError(f"top_m must be >= 1, got {top_m}")
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    pool = [str(v) for v in base_ids]
    if not pool:
        return [], {
            "applied": False,
            "fallback_reason": "empty_pool",
            "scored_count": 0,
            "top_m": top_m,
            "mode": mode,
            "beta": float(beta),
            "scores": {},
        }

    head = pool[:top_m]
    tail = pool[top_m:]
    entries = vl_entries or {}

    scored_by_pos: dict[int, float] = {}
    for idx, vid in enumerate(head):
        sc = _extract_valid_vl_score(entries.get(vid))
        if sc is not None:
            scored_by_pos[idx] = sc

    if not entries:
        return pool, {
            "applied": False,
            "fallback_reason": "missing_vl_scores",
            "scored_count": 0,
            "top_m": len(head),
            "mode": mode,
            "beta": float(beta),
            "scores": {},
        }

    top1_scored = 0 in scored_by_pos
    if not top1_scored:
        # Baseline top-1 MUST remain at slot 0 when unscored; record explicit reason
        reranked_head = list(head)
        eligible_tail_positions = sorted(i for i in scored_by_pos if i > 0)
        if len(eligible_tail_positions) >= 2:
            vl_sorted = sorted(
                eligible_tail_positions,
                key=lambda pos: (-scored_by_pos[pos], pos),
            )
            vl_rank_map = {pos: r for r, pos in enumerate(vl_sorted, 1)}
            if mode == "pure":
                ordered_positions = vl_sorted
            else:
                ordered_positions = sorted(
                    eligible_tail_positions,
                    key=lambda pos: (
                        -(
                            (1.0 - float(beta)) / (60.0 + (pos + 1))
                            + float(beta) / (60.0 + vl_rank_map[pos])
                        ),
                        pos,
                    ),
                )
            for dst_pos, src_pos in zip(eligible_tail_positions, ordered_positions):
                reranked_head[dst_pos] = head[src_pos]
        return reranked_head + tail, {
            "applied": False,
            "fallback_reason": "top1_unscored",
            "scored_count": len(scored_by_pos),
            "top_m": len(head),
            "mode": mode,
            "beta": float(beta),
            "scores": {head[pos]: round(sc, 8) for pos, sc in scored_by_pos.items()},
        }

    if len(scored_by_pos) < 2:
        return pool, {
            "applied": False,
            "fallback_reason": "insufficient_vl_scores",
            "scored_count": len(scored_by_pos),
            "top_m": len(head),
            "mode": mode,
            "beta": float(beta),
            "scores": {head[pos]: round(sc, 8) for pos, sc in scored_by_pos.items()},
        }

    scored_positions = sorted(scored_by_pos.keys())
    vl_sorted = sorted(
        scored_positions,
        key=lambda pos: (-scored_by_pos[pos], pos),
    )
    vl_rank_map = {pos: r for r, pos in enumerate(vl_sorted, 1)}

    combined_scores: dict[str, float] = {}
    if mode == "pure":
        ordered_positions = vl_sorted
        for pos in scored_positions:
            combined_scores[head[pos]] = float(scored_by_pos[pos])
    else:
        pos_comb: dict[int, float] = {}
        for pos in scored_positions:
            comb = (
                (1.0 - float(beta)) / (60.0 + (pos + 1))
                + float(beta) / (60.0 + vl_rank_map[pos])
            )
            pos_comb[pos] = comb
            combined_scores[head[pos]] = float(comb)
        ordered_positions = sorted(
            scored_positions,
            key=lambda pos: (-pos_comb[pos], pos),
        )

    reranked_head = list(head)
    for dst_pos, src_pos in zip(scored_positions, ordered_positions):
        reranked_head[dst_pos] = head[src_pos]

    return reranked_head + tail, {
        "applied": True,
        "fallback_reason": None,
        "scored_count": len(scored_by_pos),
        "top_m": len(head),
        "mode": mode,
        "beta": float(beta),
        "scores": {vid: round(sc, 8) for vid, sc in combined_scores.items()},
    }
