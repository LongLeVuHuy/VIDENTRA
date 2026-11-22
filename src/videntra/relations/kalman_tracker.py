"""SORT / ByteTrack Kalman object tracker across presentation-order video shots (R6).

Separates observed detector boxes from Kalman-predicted boxes strictly,
resets tracks at shot cuts or PTS gaps exceeding ``max_gap_sec``, and
computes measurable track signals (object consistency, motion direction/speed,
pair intersection, and short-occlusion continuity) without mutating raw detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from videntra.indexing.augmented_vector import _bbox_iou, _extract_obj_bbox
from videntra.query.rules import extract_object_requirements

_MOTION_QUERY_RE = re.compile(
    r"\b(?:walk(?:ing|s)?|run(?:ning|s)?|driv(?:ing|e|es)|rid(?:ing|e|es)|"
    r"fly(?:ing|s)?|jump(?:ing|s)?|danc(?:ing|e|es)|swim(?:ming|s)?|"
    r"chas(?:ing|e|es)|throw(?:ing|s)?|catch(?:ing|es)?|fall(?:ing|s)?|"
    r"climb(?:ing|s)?|rac(?:ing|e|es)|mov(?:ing|e|es)|approach(?:ing|es)?|"
    r"leav(?:ing|e|es)|enter(?:ing|s)?|exit(?:ing|s)?|cycl(?:ing|e)|"
    r"skat(?:ing|e)|ski(?:ing)?|surf(?:ing)?|sail(?:ing)?|kick(?:ing|s)?|"
    r"hit(?:ting|s)?|roll(?:ing|s)?|slid(?:ing|e)|spin(?:ning|s)?)\b",
    re.I,
)

_STATIC_QUERY_RE = re.compile(
    r"\b(?:sitting|standing\s+still|lying|sleeping|resting|parked|stationary)\b",
    re.I,
)

_RELATION_QUERY_RE = re.compile(
    r"\b(?:with|next\s+to|beside|near|close\s+to|behind|in\s+front\s+of|"
    r"on\s+top\s+of|under|between|chasing|following|talking\s+to|"
    r"playing\s+with|hugging|kissing|fighting|passing|giving|handing|"
    r"helping|holding|carrying)\b",
    re.I,
)


def classify_motion_relation_query(query_text: str) -> dict[str, bool]:
    """Classify query intent into motion/relation slice prior to scoring."""
    q = str(query_text or "")
    has_motion = bool(_MOTION_QUERY_RE.search(q))
    has_static = bool(_STATIC_QUERY_RE.search(q))
    has_relation = bool(_RELATION_QUERY_RE.search(q))
    return {
        "has_motion": has_motion,
        "has_static": has_static,
        "has_relation": has_relation,
        "is_motion_or_relation": bool(has_motion or has_relation),
    }


def _xyxy_to_z(box: tuple[float, float, float, float]) -> np.ndarray:
    """Convert normalized (x1, y1, x2, y2) to [cx, cy, area, aspect_ratio]."""
    x1, y1, x2, y2 = box
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    s = w * h
    r = w / h
    return np.array([cx, cy, s, r], dtype=np.float64)


def _x_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert Kalman state [cx, cy, s, r, ...] to clamped (x1, y1, x2, y2)."""
    cx, cy, s, r = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    s = max(1e-6, s)
    r = max(0.05, min(20.0, r))
    w = math.sqrt(s * r)
    h = s / max(1e-6, w)
    x1 = max(0.0, min(1.0, cx - 0.5 * w))
    y1 = max(0.0, min(1.0, cy - 0.5 * h))
    x2 = max(x1 + 1e-4, min(1.0, cx + 0.5 * w))
    y2 = max(y1 + 1e-4, min(1.0, cy + 0.5 * h))
    return (x1, y1, min(1.0, x2), min(1.0, y2))


@dataclass
class TrackStep:
    """One frame step inside a track, keeping observed vs predicted boxes distinct."""

    frame_ordinal: int
    pts_sec: float
    shot_id: int
    source_kind: str  # "observed" or "predicted"
    class_name: str
    confidence: float
    observed_box: tuple[float, float, float, float] | None
    predicted_box: tuple[float, float, float, float]
    velocity_xy: tuple[float, float]


@dataclass
class TrackRecord:
    """Completed or active single-object Kalman track within a continuous shot."""

    track_id: int
    shot_id: int
    class_name: str
    steps: list[TrackStep] = field(default_factory=list)
    observed_count: int = 0
    predicted_only_count: int = 0
    recovered_after_occlusion: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def duration_sec(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        return max(0.0, self.steps[-1].pts_sec - self.steps[0].pts_sec)

    @property
    def mean_speed(self) -> float:
        obs_steps = [s for s in self.steps if s.source_kind == "observed"]
        if len(obs_steps) < 2:
            return 0.0
        speeds = [math.hypot(s.velocity_xy[0], s.velocity_xy[1]) for s in obs_steps[1:]]
        return float(sum(speeds) / len(speeds)) if speeds else 0.0

    @property
    def net_displacement(self) -> float:
        obs_boxes = [s.observed_box for s in self.steps if s.source_kind == "observed" and s.observed_box is not None]
        if len(obs_boxes) < 2:
            return 0.0
        b0, b1 = obs_boxes[0], obs_boxes[-1]
        c0x, c0y = 0.5 * (b0[0] + b0[2]), 0.5 * (b0[1] + b0[3])
        c1x, c1y = 0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])
        return float(math.hypot(c1x - c0x, c1y - c0y))


class _KalmanBoxState:
    """Constant-velocity 7-state Kalman filter [cx, cy, s, r, vx, vy, vs] with dt scaling."""

    def __init__(self, initial_box: tuple[float, float, float, float], track_id: int, shot_id: int, class_name: str) -> None:
        self.track_id = track_id
        self.shot_id = shot_id
        self.class_name = class_name
        z = _xyxy_to_z(initial_box)
        self.x = np.zeros(7, dtype=np.float64)
        self.x[:4] = z

        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1]).astype(np.float64)
        self.Q_base = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 5e-3, 5e-3, 5e-3]).astype(np.float64)
        self.R = np.diag([2e-3, 2e-3, 4e-3, 4e-3]).astype(np.float64)
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)

        self.time_since_update = 0
        self.hits = 1
        self.was_occluded = False

    def predict(self, dt: float) -> tuple[float, float, float, float]:
        dt_safe = max(0.05, min(2.0, float(dt)))
        F = np.eye(7, dtype=np.float64)
        F[0, 4] = dt_safe
        F[1, 5] = dt_safe
        F[2, 6] = dt_safe
        if self.x[2] + self.x[6] * dt_safe <= 1e-5:
            self.x[6] = 0.0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * dt_safe
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.was_occluded = True
        return _x_to_xyxy(self.x)

    def update(self, box: tuple[float, float, float, float]) -> bool:
        """Update filter with an observed box; returns True if recovered from occlusion."""
        recovered = self.was_occluded
        z = _xyxy_to_z(box)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I7 = np.eye(7, dtype=np.float64)
        self.P = (I7 - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.was_occluded = False
        return recovered


class ShotAwareKalmanTracker:
    """SORT / ByteTrack multi-object tracker obeying shot cuts and PTS gaps."""

    def __init__(
        self,
        *,
        mode: str = "bytetrack",
        iou_threshold: float = 0.25,
        low_iou_threshold: float = 0.15,
        high_conf_thresh: float = 0.35,
        low_conf_thresh: float = 0.20,
        max_age_frames: int = 3,
        max_gap_sec: float = 2.5,
        min_hits: int = 2,
    ) -> None:
        if mode not in ("sort", "bytetrack"):
            raise ValueError(f"Unsupported tracker mode '{mode}', expected 'sort' or 'bytetrack'")
        self.mode = mode
        self.iou_threshold = float(iou_threshold)
        self.low_iou_threshold = float(low_iou_threshold)
        self.high_conf_thresh = float(high_conf_thresh)
        self.low_conf_thresh = float(low_conf_thresh)
        self.max_age_frames = int(max_age_frames)
        self.max_gap_sec = float(max_gap_sec)
        self.min_hits = int(min_hits)

    @staticmethod
    def _associate(
        detections: Sequence[dict[str, Any]],
        track_states: Sequence[tuple[_KalmanBoxState, tuple[float, float, float, float]]],
        *,
        iou_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not detections or not track_states:
            return [], list(range(len(detections))), list(range(len(track_states)))

        cost_mat = np.full((len(detections), len(track_states)), 1e6, dtype=np.float64)
        iou_mat = np.zeros((len(detections), len(track_states)), dtype=np.float64)
        for d_idx, det in enumerate(detections):
            d_box = det["box"]
            d_cls = det["class_name"]
            for t_idx, (trk, pred_box) in enumerate(track_states):
                if trk.class_name != d_cls:
                    continue
                iou = _bbox_iou(d_box, pred_box)
                iou_mat[d_idx, t_idx] = iou
                if iou >= iou_thresh:
                    cost_mat[d_idx, t_idx] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matches: list[tuple[int, int]] = []
        matched_d: set[int] = set()
        matched_t: set[int] = set()
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if iou_mat[r, c] >= iou_thresh and track_states[c][0].class_name == detections[r]["class_name"]:
                matches.append((r, c))
                matched_d.add(r)
                matched_t.add(c)
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
        unmatched_t = [i for i in range(len(track_states)) if i not in matched_t]
        return matches, unmatched_d, unmatched_t

    def track_video_frames(
        self,
        frame_observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Track detections across presentation-ordered frames.

        Each item in ``frame_observations`` should provide:
        - ``ordinal`` (int), ``pts_sec`` (float), ``is_shot_cut`` (bool, optional)
        - ``detections``: list of YOLO dict objects (with bbox + class_name + confidence)
        """
        ordered = sorted(
            frame_observations,
            key=lambda f: (int(f.get("ordinal", 0)), float(f.get("pts_sec", 0.0))),
        )
        next_track_id = 1
        current_shot_id = 0
        prev_pts: float | None = None
        active: list[_KalmanBoxState] = []
        records_by_id: dict[int, TrackRecord] = {}
        reset_events: list[dict[str, Any]] = []

        def _flush_active() -> None:
            active.clear()

        for f_obs in ordered:
            ordinal = int(f_obs.get("ordinal", 0))
            pts = float(f_obs.get("pts_sec", 0.0))
            is_cut = bool(f_obs.get("is_shot_cut", False))
            gap_exceeded = prev_pts is not None and (pts - prev_pts) > self.max_gap_sec
            if prev_pts is not None and (is_cut or gap_exceeded or pts < prev_pts):
                _flush_active()
                current_shot_id += 1
                reset_events.append({
                    "ordinal": ordinal,
                    "pts_sec": pts,
                    "reason": "shot_cut" if is_cut else ("gap_exceeded" if gap_exceeded else "pts_backward"),
                    "shot_id": current_shot_id,
                })

            dt = (pts - prev_pts) if (prev_pts is not None and pts >= prev_pts) else 0.5
            prev_pts = pts

            # Parse raw detections
            parsed_dets: list[dict[str, Any]] = []
            for raw_obj in f_obs.get("detections") or ():
                box = _extract_obj_bbox(dict(raw_obj))
                cname = str(raw_obj.get("canonical_class") or raw_obj.get("class_name") or "").lower().strip()
                conf = float(raw_obj.get("confidence", 0.0) or 0.0)
                if box is None or not cname or conf < self.low_conf_thresh:
                    continue
                parsed_dets.append({"box": box, "class_name": cname, "confidence": conf})

            high_dets = [d for d in parsed_dets if d["confidence"] >= self.high_conf_thresh]
            low_dets = [d for d in parsed_dets if d["confidence"] < self.high_conf_thresh]

            # 1. Predict all active tracks
            predicted_pairs: list[tuple[_KalmanBoxState, tuple[float, float, float, float]]] = []
            for trk in active:
                pred_box = trk.predict(dt)
                predicted_pairs.append((trk, pred_box))

            # 2. First association with high-confidence detections (SORT + ByteTrack)
            matches_hi, unmatched_hi_d, unmatched_t = self._associate(
                high_dets, predicted_pairs, iou_thresh=self.iou_threshold
            )
            matched_track_indices: set[int] = set()
            for d_idx, t_idx in matches_hi:
                trk, pred_box = predicted_pairs[t_idx]
                det = high_dets[d_idx]
                recovered = trk.update(det["box"])
                rec = records_by_id[trk.track_id]
                if recovered:
                    rec.recovered_after_occlusion += 1
                rec.observed_count += 1
                rec.steps.append(TrackStep(
                    frame_ordinal=ordinal,
                    pts_sec=pts,
                    shot_id=current_shot_id,
                    source_kind="observed",
                    class_name=trk.class_name,
                    confidence=det["confidence"],
                    observed_box=det["box"],
                    predicted_box=pred_box,
                    velocity_xy=(float(trk.x[4]), float(trk.x[5])),
                ))
                matched_track_indices.add(t_idx)

            # 3. ByteTrack second association with low-confidence detections for unmatched tracks
            still_unmatched_t = unmatched_t
            if self.mode == "bytetrack" and low_dets and unmatched_t:
                rem_pairs = [predicted_pairs[i] for i in unmatched_t]
                matches_lo, _, rem_local_t = self._associate(
                    low_dets, rem_pairs, iou_thresh=self.low_iou_threshold
                )
                for d_idx, local_t_idx in matches_lo:
                    t_idx = unmatched_t[local_t_idx]
                    trk, pred_box = predicted_pairs[t_idx]
                    det = low_dets[d_idx]
                    recovered = trk.update(det["box"])
                    rec = records_by_id[trk.track_id]
                    if recovered:
                        rec.recovered_after_occlusion += 1
                    rec.observed_count += 1
                    rec.steps.append(TrackStep(
                        frame_ordinal=ordinal,
                        pts_sec=pts,
                        shot_id=current_shot_id,
                        source_kind="observed",
                        class_name=trk.class_name,
                        confidence=det["confidence"],
                        observed_box=det["box"],
                        predicted_box=pred_box,
                        velocity_xy=(float(trk.x[4]), float(trk.x[5])),
                    ))
                    matched_track_indices.add(t_idx)
                still_unmatched_t = [unmatched_t[i] for i in rem_local_t]

            # 4. Record predicted-only steps for surviving unmatched tracks; expire old ones
            next_active: list[_KalmanBoxState] = []
            for t_idx, (trk, pred_box) in enumerate(predicted_pairs):
                if t_idx in matched_track_indices:
                    next_active.append(trk)
                elif trk.time_since_update <= self.max_age_frames:
                    rec = records_by_id[trk.track_id]
                    rec.predicted_only_count += 1
                    rec.steps.append(TrackStep(
                        frame_ordinal=ordinal,
                        pts_sec=pts,
                        shot_id=current_shot_id,
                        source_kind="predicted",
                        class_name=trk.class_name,
                        confidence=0.0,
                        observed_box=None,
                        predicted_box=pred_box,
                        velocity_xy=(float(trk.x[4]), float(trk.x[5])),
                    ))
                    next_active.append(trk)

            # 5. Spawn new tracks from unmatched high-confidence detections
            for d_idx in unmatched_hi_d:
                det = high_dets[d_idx]
                trk = _KalmanBoxState(det["box"], next_track_id, current_shot_id, det["class_name"])
                rec = TrackRecord(
                    track_id=next_track_id,
                    shot_id=current_shot_id,
                    class_name=det["class_name"],
                    observed_count=1,
                )
                rec.steps.append(TrackStep(
                    frame_ordinal=ordinal,
                    pts_sec=pts,
                    shot_id=current_shot_id,
                    source_kind="observed",
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    observed_box=det["box"],
                    predicted_box=det["box"],
                    velocity_xy=(0.0, 0.0),
                ))
                records_by_id[next_track_id] = rec
                next_active.append(trk)
                next_track_id += 1

            active = next_active

        valid_tracks = [
            rec for rec in records_by_id.values() if rec.observed_count >= self.min_hits
        ]
        all_tracks = list(records_by_id.values())
        return {
            "mode": self.mode,
            "num_frames": len(ordered),
            "num_shots": current_shot_id + 1,
            "reset_events": reset_events,
            "tracks": valid_tracks,
            "all_tracks": all_tracks,
        }


def compute_track_query_signals(
    query_text: str,
    tracking_output: Mapping[str, Any],
    *,
    frame_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute measurable crop-presence and track-level signals for R6 reranking."""
    intent = classify_motion_relation_query(query_text)
    reqs = extract_object_requirements(query_text)
    req_classes = {
        str(r["class_name"]).lower().strip()
        for r in reqs
        if isinstance(r, dict) and r.get("class_name")
    }

    tracks: Sequence[TrackRecord] = tracking_output.get("tracks") or ()
    n_frames = max(1, int(tracking_output.get("num_frames", 1)))

    # 1. Crop-only object score (independent of tracking, for global+crop ablation)
    crop_match_score = 0.0
    if frame_observations is not None and req_classes:
        matched_frames = 0
        conf_sum = 0.0
        for f_obs in frame_observations:
            best_c = 0.0
            for det in f_obs.get("detections") or ():
                cname = str(det.get("canonical_class") or det.get("class_name") or "").lower().strip()
                if cname in req_classes:
                    best_c = max(best_c, float(det.get("confidence", 0.0) or 0.0))
            if best_c > 0.0:
                matched_frames += 1
                conf_sum += best_c
        crop_match_score = min(1.0, (matched_frames / float(n_frames)) * (conf_sum / max(1, matched_frames)))
    elif frame_observations is not None:
        # Generic foreground presence confidence
        fg_frames = sum(1 for f_obs in frame_observations if (f_obs.get("detections") or ()))
        crop_match_score = 0.25 * min(1.0, fg_frames / float(n_frames))

    if not tracks:
        return {
            "crop_match_score": float(crop_match_score),
            "object_consistency": 0.0,
            "motion_alignment": 0.0,
            "track_intersection": 0.0,
            "occlusion_continuity": 0.0,
            "track_score": 0.0,
        }

    relevant_tracks = [t for t in tracks if not req_classes or t.class_name in req_classes]
    if not relevant_tracks:
        relevant_tracks = list(tracks)

    # 2. Object track consistency: longest/cleanest observed persistence ratio per class
    best_persistence = max(t.observed_count / float(n_frames) for t in relevant_tracks)
    class_cov = (
        len({t.class_name for t in relevant_tracks if t.class_name in req_classes}) / float(len(req_classes))
        if req_classes
        else min(1.0, len(relevant_tracks) / 3.0)
    )
    object_consistency = float(min(1.0, 0.65 * best_persistence + 0.35 * class_cov))

    # 3. Motion alignment from Kalman velocities & net displacement
    avg_disp = max((t.net_displacement for t in relevant_tracks), default=0.0)
    avg_speed = max((t.mean_speed for t in relevant_tracks), default=0.0)
    motion_mag = min(1.0, 2.0 * avg_disp + 1.5 * avg_speed)
    if intent["has_motion"]:
        motion_alignment = float(motion_mag)
    elif intent["has_static"]:
        motion_alignment = float(max(0.0, 1.0 - motion_mag))
    else:
        motion_alignment = float(0.25 * motion_mag)

    # 4. Pairwise track spatial proximity / intersection within the same shot
    track_intersection = 0.0
    if len(tracks) >= 2:
        by_shot: dict[int, list[TrackRecord]] = {}
        for t in tracks:
            by_shot.setdefault(t.shot_id, []).append(t)
        best_inter = 0.0
        for s_tracks in by_shot.values():
            if len(s_tracks) < 2:
                continue
            step_maps = [
                {
                    st.frame_ordinal: st.observed_box
                    for st in t.steps
                    if st.source_kind == "observed" and st.observed_box is not None
                }
                for t in s_tracks
            ]
            for i in range(len(s_tracks)):
                for j in range(i + 1, len(s_tracks)):
                    common_ords = set(step_maps[i].keys()) & set(step_maps[j].keys())
                    if not common_ords:
                        continue
                    for o in common_ords:
                        bi, bj = step_maps[i][o], step_maps[j][o]
                        iou = _bbox_iou(bi, bj)
                        cix, ciy = 0.5 * (bi[0] + bi[2]), 0.5 * (bi[1] + bi[3])
                        cjx, cjy = 0.5 * (bj[0] + bj[2]), 0.5 * (bj[1] + bj[3])
                        dist = math.hypot(cix - cjx, ciy - cjy)
                        prox = max(iou, max(0.0, 1.0 - dist * 2.5) * 0.5)
                        if prox > best_inter:
                            best_inter = prox
        track_intersection = float(best_inter if intent["has_relation"] else 0.35 * best_inter)

    # 5. Short-occlusion continuity (only credits tracks that actually re-observed after occlusion)
    rec_total = sum(t.recovered_after_occlusion for t in relevant_tracks)
    occlusion_continuity = float(min(1.0, rec_total / 2.0))

    track_score = float(
        0.40 * object_consistency
        + 0.30 * motion_alignment
        + 0.20 * track_intersection
        + 0.10 * occlusion_continuity
    )
    return {
        "crop_match_score": float(crop_match_score),
        "object_consistency": object_consistency,
        "motion_alignment": motion_alignment,
        "track_intersection": track_intersection,
        "occlusion_continuity": occlusion_continuity,
        "track_score": track_score,
    }
