"""Fast zero-LLM spatial, object cardinality, and chronological action verifier (P4)."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np

from videntra.indexing.augmented_vector import _extract_obj_bbox, deduplicate_and_filter_frame_objects
from videntra.query.rules import COCO_SYNONYM_MAP, extract_object_requirements
from videntra.query.schemas import QueryGraph
from videntra.relations.predicates import _check_2d_spatial_geometry

_SPATIAL_PATTERN_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("left_of", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:on\s+the\s+left\s+of|to\s+the\s+left\s+of|left\s+of)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
    ("right_of", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:on\s+the\s+right\s+of|to\s+the\s+right\s+of|right\s+of)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
    ("above", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:above|on\s+top\s+of)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
    ("below", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:below|under|underneath|beneath)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
    ("inside", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:inside)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
    ("near", re.compile(r"\b(?:a|an|the)?\s*([a-z]+)\s+(?:next\s+to|beside|near|close\s+to)\s+(?:a|an|the)?\s*([a-z]+)\b", re.I)),
)

_REVERSE_SPATIAL_OP: dict[str, str] = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
}

_ORDER_CUES_RE = re.compile(r"\b(?:then|and\s+then|before|after|followed\s+by|first|finally)\b", re.I)
_MOTION_CUES_RE = re.compile(
    r"\b(?:running|walking|jumping|falling|approaching|leaving|driving|chasing|playing|dancing|riding|flying|swimming)\b",
    re.I,
)


@dataclass(slots=True)
class SpatialActionAssessment:
    """Output of fast P4 verification for one candidate video/moment."""

    score: float = 0.0
    cardinality_score: float = 0.0
    spatial_predicate_score: float = 0.0
    action_order_score: float = 0.0
    status: str = "unknown"


def _canonical_coco_class(term: str) -> str | None:
    t = term.lower().strip()
    if t in COCO_SYNONYM_MAP:
        return COCO_SYNONYM_MAP[t]
    if t.endswith("s") and len(t) > 3 and t[:-1] in COCO_SYNONYM_MAP:
        return COCO_SYNONYM_MAP[t[:-1]]
    return None


def extract_spatial_relations_from_query(
    query: str,
    query_graph: QueryGraph | None = None,
) -> list[tuple[str, str, str]]:
    """Return list of (canonical_op, subj_coco_class, obj_coco_class) tuples."""
    relations: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    if query_graph is not None:
        ent_map = {e.id: e for e in query_graph.entities}
        for pred in query_graph.predicates:
            op = str(getattr(pred, "operator", "") or "").lower().strip()
            if op in ("left_of", "right_of", "above", "below", "inside", "near") and getattr(pred, "polarity", "positive") != "negative":
                roles = getattr(pred, "roles", {}) or {}
                role_vals = list(roles.values()) if isinstance(roles, dict) else []
                s_id = roles.get("subject") or roles.get("agent") or (role_vals[0] if len(role_vals) >= 1 else None)
                o_id = roles.get("object") or roles.get("reference") or roles.get("target") or (role_vals[1] if len(role_vals) >= 2 else None)
                s_ent = ent_map.get(str(s_id)) if s_id else None
                o_ent = ent_map.get(str(o_id)) if o_id else None
                if s_ent is not None and o_ent is not None:
                    sc = _canonical_coco_class(s_ent.mention) or _canonical_coco_class(s_ent.type)
                    oc = _canonical_coco_class(o_ent.mention) or _canonical_coco_class(o_ent.type)
                    if sc and oc:
                        item = (op, sc, oc)
                        if item not in seen:
                            seen.add(item)
                            relations.append(item)

    for op, pat in _SPATIAL_PATTERN_SPECS:
        for m in pat.finditer(query):
            sc = _canonical_coco_class(m.group(1))
            oc = _canonical_coco_class(m.group(2))
            if sc and oc:
                item = (op, sc, oc)
                if item not in seen:
                    seen.add(item)
                    relations.append(item)

    return relations


def _monotonic_chain_dp(clause_curve: np.ndarray, *, reverse: bool = False) -> float:
    """Best average similarity along non-decreasing temporal indices (forward or reversed clauses)."""
    mat = clause_curve[::-1] if reverse else clause_curve
    c_len, t_len = mat.shape
    if c_len == 0 or t_len == 0:
        return 0.0
    dp = np.full(t_len, -1e9, dtype=np.float32)
    dp[:] = mat[0]
    for ci in range(1, c_len):
        pref_max = np.maximum.accumulate(dp)
        dp = pref_max + mat[ci]
    return float(np.max(dp) / float(c_len))


class SpatialActionVerifier:
    """Zero-LLM P4 verifier combining per-frame cardinality, 2D predicates, and chronological DP."""

    def __init__(self, *, window_pad_sec: float = 2.0) -> None:
        self.window_pad_sec = float(window_pad_sec)

    def prepare_query_context(
        self,
        query: str,
        clauses: Sequence[str],
        query_graph: QueryGraph | None = None,
    ) -> dict[str, Any]:
        """Parse query requirements once per search request."""
        raw_reqs = extract_object_requirements(query)
        obj_reqs: dict[str, int] = {}
        for req in raw_reqs:
            if isinstance(req, dict) and req.get("class_name"):
                cname = str(req["class_name"]).lower().strip()
                cnt = int(req.get("cardinality", req.get("count", 1)) or 1)
                obj_reqs[cname] = max(obj_reqs.get(cname, 0), max(1, cnt))
        spatial_rels = extract_spatial_relations_from_query(query, query_graph=query_graph)
        has_after = bool(re.search(r"\bafter\b", query, re.I))
        has_order_cue = bool(_ORDER_CUES_RE.search(query))
        has_motion_cue = bool(_MOTION_CUES_RE.search(query))
        return {
            "obj_reqs": obj_reqs,
            "spatial_rels": spatial_rels,
            "clauses": list(clauses),
            "has_order_cue": has_order_cue,
            "has_after": has_after,
            "has_motion_cue": has_motion_cue,
        }

    def verify_candidate(
        self,
        *,
        query_ctx: dict[str, Any],
        video_id: str,
        window_start: float,
        window_end: float,
        video_objects: Sequence[dict[str, Any]],
        video_obj_pts: Sequence[float] | None,
        detector_ran: bool,
        clause_curve: np.ndarray | None = None,
        vp_clause_order: float | None = None,
        already_deduplicated: bool = False,
    ) -> SpatialActionAssessment:
        """Compute P4 assessment in O(window objects + C*T) with missing evidence = unknown."""
        obj_reqs = query_ctx.get("obj_reqs") or {}
        spatial_rels = query_ctx.get("spatial_rels") or []
        has_motion = bool(query_ctx.get("has_motion_cue"))
        need_obj_stats = bool(detector_ran and (obj_reqs or spatial_rels))
        need_objs = bool(need_obj_stats or has_motion)
        need_order = bool(query_ctx.get("has_order_cue") or len(query_ctx.get("clauses") or ()) >= 2)
        if not need_objs and not need_order:
            return SpatialActionAssessment()

        win_objs: Sequence[dict[str, Any]] = ()
        if need_objs and video_objects:
            lo_t = float(window_start) - self.window_pad_sec
            hi_t = float(window_end) + self.window_pad_sec
            if video_obj_pts is not None and len(video_obj_pts) == len(video_objects):
                i_l = bisect_left(video_obj_pts, lo_t)
                i_r = bisect_right(video_obj_pts, hi_t)
                raw_win = video_objects[i_l:i_r]
            else:
                raw_win = [o for o in video_objects if lo_t <= float(o.get("pts_sec", 0.0)) <= hi_t]
            source_win = raw_win if raw_win else video_objects
            win_objs = source_win if already_deduplicated else deduplicate_and_filter_frame_objects(source_win)

        # Group deduplicated objects by frame key only when object counts or spatial relations apply
        objs_by_frame: dict[Any, list[dict[str, Any]]] = {}
        cls_frame_counts: dict[str, dict[Any, int]] = {}
        if need_obj_stats and win_objs:
            for o in win_objs:
                cname = str(o.get("canonical_class") or o.get("class_name") or "").lower().strip()
                if not cname:
                    continue
                fk = o.get("keyframe_id")
                if fk is None:
                    fk = round(float(o.get("pts_sec", 0.0)), 2)
                objs_by_frame.setdefault(fk, []).append(o)
                cdict = cls_frame_counts.setdefault(cname, {})
                cdict[fk] = cdict.get(fk, 0) + 1

        # 1. Per-Frame Cardinality Verification (never guess when detector didn't cover video)
        card_score = 0.0
        has_supported = False
        has_contradicted = False
        if obj_reqs and detector_ran:
            term_scores: list[float] = []
            for cname, req_n in obj_reqs.items():
                max_frame_cnt = max(cls_frame_counts.get(cname, {}).values(), default=0)
                req_int = max(1, int(req_n))
                if max_frame_cnt >= req_int:
                    term_scores.append(0.85 if req_int >= 2 else 0.35)
                    has_supported = True
                elif max_frame_cnt > 0 and req_int >= 2:
                    # Strictly fewer instances per frame than requested count (e.g. 1 vs 3)
                    term_scores.append(-0.65)
                    has_contradicted = True
                else:
                    # Absence may be missed detection; mildly negative only for multi-frame covered vids
                    term_scores.append(-0.10 if req_int >= 2 else 0.0)
            if term_scores:
                card_score = float(np.clip(sum(term_scores) / len(term_scores), -1.0, 1.0))

        # 2. 2D Spatial Relation Verification via Co-occurring FrameObjects
        spatial_score = 0.0
        if spatial_rels and detector_ran and objs_by_frame:
            rel_scores: list[float] = []
            for op, subj_c, obj_c in spatial_rels:
                matched_any = False
                opposite_any = False
                rev_op = _REVERSE_SPATIAL_OP.get(op)
                for frame_list in objs_by_frame.values():
                    subj_boxes = [
                        _extract_obj_bbox(o)
                        for o in frame_list
                        if str(o.get("canonical_class") or o.get("class_name") or "").lower().strip() == subj_c
                    ]
                    obj_boxes = [
                        _extract_obj_bbox(o)
                        for o in frame_list
                        if str(o.get("canonical_class") or o.get("class_name") or "").lower().strip() == obj_c
                    ]
                    subj_valid = [b for b in subj_boxes if b is not None]
                    obj_valid = [b for b in obj_boxes if b is not None]
                    if not subj_valid or not obj_valid:
                        continue
                    for sb in subj_valid:
                        for ob in obj_valid:
                            if subj_c == obj_c and sb == ob:
                                continue
                            if _check_2d_spatial_geometry(op, sb, ob):
                                matched_any = True
                            elif rev_op is not None and _check_2d_spatial_geometry(rev_op, sb, ob):
                                opposite_any = True
                if matched_any:
                    rel_scores.append(0.80)
                    has_supported = True
                elif opposite_any:
                    rel_scores.append(-0.55)
                    has_contradicted = True
                else:
                    rel_scores.append(0.0)
            if rel_scores:
                spatial_score = float(np.clip(sum(rel_scores) / len(rel_scores), -1.0, 1.0))

        # 3. Chronological Action Ordering (Forward vs Reverse DP + Motion/Area shift)
        order_score = 0.0
        if clause_curve is not None and clause_curve.ndim == 2 and clause_curve.shape[0] >= 2 and clause_curve.shape[1] >= 2:
            if query_ctx.get("has_after"):
                # "X after Y" expects Y -> X chronological ordering, handled cleanly unless already ordered
                fwd = _monotonic_chain_dp(clause_curve, reverse=False)
                rev = _monotonic_chain_dp(clause_curve, reverse=True)
                margin = max(fwd - rev, 0.5 * (rev - fwd))
            else:
                fwd = _monotonic_chain_dp(clause_curve, reverse=False)
                rev = _monotonic_chain_dp(clause_curve, reverse=True)
                margin = fwd - rev
            dp_sc = float(math.tanh(margin * 18.0))
            if query_ctx.get("has_order_cue") or len(query_ctx.get("clauses") or ()) >= 2:
                order_score = dp_sc
                if dp_sc > 0.15:
                    has_supported = True
                elif dp_sc < -0.20:
                    has_contradicted = True

        if vp_clause_order is not None and (query_ctx.get("has_order_cue") or len(query_ctx.get("clauses") or ()) >= 2):
            vp_centered = 2.0 * float(vp_clause_order) - 1.0
            order_score = 0.6 * order_score + 0.4 * vp_centered if abs(order_score) > 1e-6 else 0.7 * vp_centered

        # Dynamic motion displacement check if query has motion verb & multiple timestamped boxes
        if query_ctx.get("has_motion_cue") and win_objs:
            centers_by_t: list[tuple[float, float, float]] = []
            for o in win_objs:
                box = _extract_obj_bbox(o)
                if box is not None:
                    centers_by_t.append((float(o.get("pts_sec", 0.0)), 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])))
            if len(centers_by_t) >= 3:
                centers_by_t.sort(key=lambda x: x[0])
                xs = np.array([c[1] for c in centers_by_t], dtype=np.float32)
                ys = np.array([c[2] for c in centers_by_t], dtype=np.float32)
                norm_scale = max(float(np.max(xs)), 1.0) if float(np.max(xs)) > 2.0 else 1.0
                disp = float(math.hypot(float(np.std(xs)) / norm_scale, float(np.std(ys)) / norm_scale))
                if disp >= 0.08:
                    order_score = float(np.clip(order_score + 0.15, -1.0, 1.0))

        combined = float(np.clip(0.40 * card_score + 0.30 * spatial_score + 0.30 * order_score, -1.0, 1.0))
        if has_contradicted and not has_supported:
            status = "contradicted"
        elif has_supported and not has_contradicted:
            status = "supported"
        elif has_supported and has_contradicted:
            status = "partial"
        else:
            status = "unknown"

        return SpatialActionAssessment(
            score=combined,
            cardinality_score=card_score,
            spatial_predicate_score=spatial_score,
            action_order_score=order_score,
            status=status,
        )
