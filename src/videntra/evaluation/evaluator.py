"""Retrieval and localization evaluation metrics separated from inference runtime."""
from dataclasses import dataclass, asdict
import math
from typing import Any, Sequence
import numpy as np

from videntra.retrieval.visual import temporal_iou


def _extract_moment_tuple(m: Any) -> tuple[str, float, float]:
    """Helper to extract (video_id, start_sec, end_sec) from various representations."""
    if isinstance(m, (tuple, list)) and len(m) >= 3:
        return str(m[0]), float(m[1]), float(m[2])
    if isinstance(m, dict):
        vid = str(m.get("video_id") or m.get("vid", ""))
        start = float(m.get("moment_start_seconds", m.get("start_sec", m.get("start_seconds", m.get("start", 0.0)))))
        end = float(m.get("moment_end_seconds", m.get("end_sec", m.get("end_seconds", m.get("end", 0.0)))))
        return vid, start, end
    vid = str(getattr(m, "video_id", ""))
    start = float(getattr(m, "start_seconds", getattr(m, "start_sec", getattr(m, "start", getattr(m, "moment_start_seconds", 0.0)))))
    end = float(getattr(m, "end_seconds", getattr(m, "end_sec", getattr(m, "end", getattr(m, "moment_end_seconds", 0.0)))))
    return vid, start, end


def _normalize_gt_windows(gts: Sequence[Any]) -> list[tuple[str, float, float]]:
    """Flatten and normalize ground truth items into a list of (video_id, start_sec, end_sec)."""
    gt_list: list[tuple[str, float, float]] = []
    for g in gts:
        if isinstance(g, dict) and "relevant_windows" in g:
            vid = str(g.get("video_id") or g.get("vid", ""))
            for w in g["relevant_windows"]:
                gt_list.append((vid, float(w[0]), float(w[1])))
        elif isinstance(g, (tuple, list)) and len(g) >= 3:
            gt_list.append((str(g[0]), float(g[1]), float(g[2])))
        elif isinstance(g, dict):
            vid = str(g.get("video_id") or g.get("vid", ""))
            s = float(g.get("start_seconds", g.get("start_sec", g.get("start", 0.0))))
            e = float(g.get("end_seconds", g.get("end_sec", g.get("end", 0.0))))
            if e > s or ("start_seconds" in g or "start_sec" in g or "start" in g):
                gt_list.append((vid, s, e))
        else:
            t = _extract_moment_tuple(g)
            gt_list.append(t)
    return gt_list


def compute_binary_ndcg_at_k(
    predicted_video_ids: list[str],
    target_video_ids: set[str],
    k: int,
    indexed_video_ids: set[str] | None = None,
) -> float:
    """Computes Video nDCG@K with first-occurrence rank deduplication and multi-target support.
    
    Returns 0.0 if k <= 0 or IDCG == 0.
    """
    if k <= 0:
        return 0.0

    if indexed_video_ids is not None:
        effective_target_vids = set(target_video_ids) & set(indexed_video_ids)
    else:
        effective_target_vids = set(target_video_ids)

    if not effective_target_vids:
        return 0.0

    # Deduplicate predicted_video_ids preserving first occurrence
    seen: set[str] = set()
    dedup_preds: list[str] = []
    for vid in predicted_video_ids:
        if vid not in seen:
            seen.add(vid)
            dedup_preds.append(vid)

    # Calculate DCG@K
    dcg = 0.0
    for i, vid in enumerate(dedup_preds[:k]):
        if vid in effective_target_vids:
            dcg += 1.0 / math.log2(i + 2)

    # Calculate IDCG@K
    ideal_hits = min(k, len(effective_target_vids))
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_graded_moment_ndcg_at_k(
    predicted_moments: Sequence[Any],
    ground_truth_windows: Sequence[Any],
    k: int,
) -> float:
    """Computes Graded Moment nDCG@K with 4-level gains based on tIoU thresholds:
    - tIoU >= 0.70 -> grade 3 (gain 2^3 - 1 = 7.0)
    - 0.50 <= tIoU < 0.70 -> grade 2 (gain 2^2 - 1 = 3.0)
    - 0.30 <= tIoU < 0.50 -> grade 1 (gain 2^1 - 1 = 1.0)
    - tIoU < 0.30 or wrong video_id -> grade 0 (gain 0.0)

    Enforces greedy 1-to-1 matching to prevent duplicate GT window exploitation.
    """
    if k <= 0:
        return 0.0

    gt_windows = _normalize_gt_windows(ground_truth_windows)
    m_gt = len(gt_windows)
    if m_gt == 0:
        return 0.0

    # Parse predicted moments
    preds = [_extract_moment_tuple(p) for p in predicted_moments[:k]]

    used_gt_indices: set[int] = set()
    dcg = 0.0

    for i, (p_vid, p_s, p_e) in enumerate(preds):
        best_gt_idx = -1
        best_iou = -1.0

        for gt_idx, (gt_vid, gt_s, gt_e) in enumerate(gt_windows):
            if gt_idx in used_gt_indices:
                continue
            if gt_vid != p_vid:
                continue
            iou = temporal_iou(p_s, p_e, gt_s, gt_e)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Assign 4-level graded relevance
        if best_gt_idx >= 0 and best_iou >= 0.30:
            used_gt_indices.add(best_gt_idx)
            if best_iou >= 0.70:
                gain = 7.0  # 2^3 - 1
            elif best_iou >= 0.50:
                gain = 3.0  # 2^2 - 1
            else:
                gain = 1.0  # 2^1 - 1
        else:
            gain = 0.0

        if gain > 0.0:
            dcg += gain / math.log2(i + 2)

    # Calculate IDCG@K: ideal gain is 7.0 for up to min(k, m_gt)
    ideal_hits = min(k, m_gt)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 7.0 / math.log2(i + 2)

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_moment_ap_at_iou(
    predicted_moments: Sequence[Any],
    ground_truth_windows: Sequence[Any],
    iou_thresh: float = 0.5,
    *,
    iou_threshold: float | None = None,
) -> float:
    """E7: Computes Average Precision (AP) at a single tIoU threshold with 1-to-1 greedy matching."""
    if iou_threshold is not None:
        if iou_thresh != 0.5 and iou_thresh != iou_threshold:
            raise ValueError("Conflicting IoU thresholds")
        iou_thresh = iou_threshold
    gt_windows = _normalize_gt_windows(ground_truth_windows)
    m_gt = len(gt_windows)
    if m_gt == 0 or not predicted_moments:
        return 0.0

    preds = [_extract_moment_tuple(p) for p in predicted_moments]
    used_gt_indices: set[int] = set()
    tp_cumsum = 0
    precisions: list[float] = []

    for rank_idx, (p_vid, p_s, p_e) in enumerate(preds, start=1):
        best_gt_idx = -1
        best_iou = -1.0
        for gt_idx, (gt_vid, gt_s, gt_e) in enumerate(gt_windows):
            if gt_idx in used_gt_indices or gt_vid != p_vid:
                continue
            iou = temporal_iou(p_s, p_e, gt_s, gt_e)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            used_gt_indices.add(best_gt_idx)
            tp_cumsum += 1
            precisions.append(tp_cumsum / float(rank_idx))

    if not precisions:
        return 0.0
    return float(sum(precisions) / float(m_gt))


def compute_moment_map_range(
    predicted_moments: Sequence[Any],
    ground_truth_windows: Sequence[Any],
    iou_thresholds: Sequence[float] | None = None,
) -> float:
    """E7: Computes Moment mAP@[0.5:0.95:0.05] averaged over 10 tIoU thresholds."""
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    if not iou_thresholds:
        return 0.0
    aps = [compute_moment_ap_at_iou(predicted_moments, ground_truth_windows, float(th)) for th in iou_thresholds]
    return float(sum(aps) / len(aps))


def compute_bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """E4: Computes non-parametric bootstrap mean and (1-alpha) confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if n == 1 or n_resamples <= 0:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[indices], axis=1)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(boot_means, low_pct))
    ci_high = float(np.percentile(boot_means, high_pct))
    return mean_val, ci_low, ci_high


def compute_paired_sign_test_pvalue(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """E4: Computes two-sided paired permutation / sign test p-value between two per-query metric arrays."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    diffs = b - a
    obs_diff = abs(float(np.mean(diffs)))
    if obs_diff <= 1e-12:
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    perm_diffs = np.abs(np.mean(signs * diffs, axis=1))
    count_extreme = int(np.sum(perm_diffs >= (obs_diff - 1e-12)))
    return float((count_extreme + 1) / (n_permutations + 1))


def classify_query_failure(
    hit_at_1: int,
    target_videos: set[str],
    predicted_top1_video: str,
    predicted_top1_attribution: str,
    visual_top120_vids: set[str] | None = None,
    moment_hit_iou05: int | str | None = None,
) -> str:
    """E10: Automated Failure Taxonomy classifying every query outcome into PASS or F1..F5."""
    if hit_at_1 == 1:
        if moment_hit_iou05 == 0:
            return "F5_BOUNDARY_LOCALIZATION_DRIFT"
        return "PASS"

    # Check F3: Same YouTube source video ID (first 11 chars before timestamp suffix)
    pred_prefix = predicted_top1_video[:11] if len(predicted_top1_video) >= 11 else ""
    if pred_prefix and any(len(tv) >= 11 and tv[:11] == pred_prefix for tv in target_videos):
        return "F3_SAME_SOURCE_CLIP_CONFUSION"

    # Check F1: Target video not in Visual FAISS Top-120 candidate pool
    if visual_top120_vids is not None and not (target_videos & visual_top120_vids):
        return "F1_VISUAL_CANDIDATE_MISS"

    # Check F2: Lexical distractor overthrew visual candidate
    if "asr" in predicted_top1_attribution or "ocr" in predicted_top1_attribution:
        return "F2_LEXICAL_DISTRACTOR_OVERTHROW"

    # Otherwise F4: Fine-grained visual confusion within candidate pool
    return "F4_FINE_GRAINED_VISUAL_CONFUSION"


@dataclass(frozen=True)
class EvaluationReport:
    total_queries: int
    evaluated_queries: int
    video_hit_at_1: float
    video_hit_at_5: float
    video_hit_at_10: float
    moment_hit_at_1_iou03: float | None
    moment_hit_at_1_iou05: float | None
    moment_hit_at_1_iou07: float | None
    moment_hit_at_5_iou05: float | None
    mean_reciprocal_rank: float
    mean_boundary_mae_seconds: float | None
    video_ndcg_at_1: float = 0.0
    video_ndcg_at_5: float = 0.0
    video_ndcg_at_10: float = 0.0
    moment_ndcg_at_5: float | None = None
    moment_ndcg_at_10: float | None = None
    moment_labeled_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def evaluate_predictions(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    top_ks: Sequence[int] = (1, 5, 10),
) -> EvaluationReport:
    """Evaluate frozen prediction items against official ground truth annotations.
    
    predictions: list of dicts with keys: query_id, video_id, moment_start_seconds, moment_end_seconds, ranking_score
    ground_truth: list of dicts with keys: query_id, video_id, start_seconds, end_seconds (or relevant_windows)
    """
    # Group ground truth by query_id
    gt_by_query: dict[str, list[dict[str, Any]]] = {}
    for gt in ground_truth:
        qid = str(gt.get("query_id") or gt.get("qid") or "q1")
        gt_by_query.setdefault(qid, []).append(gt)

    # Group predictions by query_id
    preds_by_query: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        prov = p.get("provenance") if isinstance(p.get("provenance"), dict) else {}
        qid = str(p.get("query_id") or p.get("qid") or prov.get("query_id") or (list(gt_by_query.keys())[0] if len(gt_by_query) == 1 else "unknown"))
        preds_by_query.setdefault(qid, []).append(p)

    total_queries = len(gt_by_query)
    if total_queries == 0:
        return EvaluationReport(
            total_queries=0,
            evaluated_queries=0,
            video_hit_at_1=0.0,
            video_hit_at_5=0.0,
            video_hit_at_10=0.0,
            moment_hit_at_1_iou03=None,
            moment_hit_at_1_iou05=None,
            moment_hit_at_1_iou07=None,
            moment_hit_at_5_iou05=None,
            mean_reciprocal_rank=0.0,
            mean_boundary_mae_seconds=None,
        )

    vid_hit_1 = 0
    vid_hit_5 = 0
    vid_hit_10 = 0
    mom_hit_1_03 = 0
    mom_hit_1_05 = 0
    mom_hit_1_07 = 0
    mom_hit_5_05 = 0
    reciprocal_ranks = []
    boundary_errors = []
    v_ndcg_1_list: list[float] = []
    v_ndcg_5_list: list[float] = []
    v_ndcg_10_list: list[float] = []
    m_ndcg_5_list: list[float] = []
    m_ndcg_10_list: list[float] = []
    moment_labeled = 0

    for qid, gts in gt_by_query.items():
        q_preds = preds_by_query.get(qid, [])
        # Sort predictions by ranking_score descending
        q_preds = sorted(q_preds, key=lambda x: float(x.get("ranking_score", 0.0)), reverse=True)

        target_vids = set(g["video_id"] for g in gts)

        # Video Hit@K
        pred_vids = list(dict.fromkeys(p["video_id"] for p in q_preds))
        if any(v in target_vids for v in pred_vids[:1]):
            vid_hit_1 += 1
        if any(v in target_vids for v in pred_vids[:5]):
            vid_hit_5 += 1
        if any(v in target_vids for v in pred_vids[:10]):
            vid_hit_10 += 1

        # Video nDCG@K
        v_ndcg_1_list.append(compute_binary_ndcg_at_k(pred_vids, target_vids, k=1))
        v_ndcg_5_list.append(compute_binary_ndcg_at_k(pred_vids, target_vids, k=5))
        v_ndcg_10_list.append(compute_binary_ndcg_at_k(pred_vids, target_vids, k=10))

        # MRR calculation
        rr = 0.0
        for rank, vid in enumerate(pred_vids[:10], start=1):
            if vid in target_vids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Moment Hit@K at tIoU thresholds
        gt_wins = _normalize_gt_windows(gts)
        if not gt_wins:
            continue
        moment_labeled += 1

        def _check_moment_hit(pred_sublist, iou_thresh):
            for p in pred_sublist:
                p_vid = p["video_id"]
                p_s = float(p.get("moment_start_seconds", p.get("start", 0.0)))
                p_e = float(p.get("moment_end_seconds", p.get("end", 0.0)))
                best_iou, best_error = -1.0, None
                for gt_vid, gt_s, gt_e in gt_wins:
                    if gt_vid == p_vid:
                        iou = temporal_iou(p_s, p_e, gt_s, gt_e)
                        if iou > best_iou:
                            best_iou = iou
                            best_error = abs(p_s - gt_s) + abs(p_e - gt_e)
                if best_iou >= iou_thresh:
                    return True, best_error
            return False, None

        hit_1_03, err_1_03 = _check_moment_hit(q_preds[:1], 0.3)
        hit_1_05, err_1_05 = _check_moment_hit(q_preds[:1], 0.5)
        hit_1_07, _ = _check_moment_hit(q_preds[:1], 0.7)
        hit_5_05, _ = _check_moment_hit(q_preds[:5], 0.5)

        if hit_1_03:
            mom_hit_1_03 += 1
        if hit_1_05:
            mom_hit_1_05 += 1
            if err_1_05 is not None:
                boundary_errors.append(err_1_05 / 2.0)
        if hit_1_07:
            mom_hit_1_07 += 1
        if hit_5_05:
            mom_hit_5_05 += 1

        # Graded Moment nDCG@K
        m_ndcg_5_list.append(compute_graded_moment_ndcg_at_k(q_preds, gt_wins, k=5))
        m_ndcg_10_list.append(compute_graded_moment_ndcg_at_k(q_preds, gt_wins, k=10))

    n = total_queries
    mean_mae = float(sum(boundary_errors) / len(boundary_errors)) if boundary_errors else None

    return EvaluationReport(
        total_queries=n,
        evaluated_queries=len(gt_by_query.keys() & preds_by_query.keys()),
        video_hit_at_1=round(vid_hit_1 / n, 4),
        video_hit_at_5=round(vid_hit_5 / n, 4),
        video_hit_at_10=round(vid_hit_10 / n, 4),
        moment_hit_at_1_iou03=round(mom_hit_1_03 / moment_labeled, 4) if moment_labeled else None,
        moment_hit_at_1_iou05=round(mom_hit_1_05 / moment_labeled, 4) if moment_labeled else None,
        moment_hit_at_1_iou07=round(mom_hit_1_07 / moment_labeled, 4) if moment_labeled else None,
        moment_hit_at_5_iou05=round(mom_hit_5_05 / moment_labeled, 4) if moment_labeled else None,
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / n, 4),
        mean_boundary_mae_seconds=round(mean_mae, 4) if mean_mae is not None else None,
        video_ndcg_at_1=round(sum(v_ndcg_1_list) / n, 4),
        video_ndcg_at_5=round(sum(v_ndcg_5_list) / n, 4),
        video_ndcg_at_10=round(sum(v_ndcg_10_list) / n, 4),
        moment_ndcg_at_5=round(sum(m_ndcg_5_list) / len(m_ndcg_5_list), 4) if m_ndcg_5_list else None,
        moment_ndcg_at_10=round(sum(m_ndcg_10_list) / len(m_ndcg_10_list), 4) if m_ndcg_10_list else None,
        moment_labeled_queries=moment_labeled,
    )
