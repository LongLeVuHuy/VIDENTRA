"""Isolated evaluator for multi-event ActivityNet Captions full-chain retrieval and VideoComp pairs.

Never imported by retrieval, sampling, query, or ranking runtime modules.
"""
from __future__ import annotations

from collections import defaultdict
from typing import AbstractSet, Any, Mapping, Sequence

import numpy as np

VIDEOCOMP_DISRUPTION_TYPES: tuple[str, ...] = (
    "temporal_reordering",
    "action_replacement",
    "segment_mismatch",
)


def compute_tiou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute 1D temporal Intersection-over-Union over intervals `[start, end)`."""
    ps, pe = float(pred_start), float(pred_end)
    gs, ge = float(gt_start), float(gt_end)
    if pe <= ps or ge <= gs:
        return 0.0
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _extract_gt_strict_order_pairs(
    gt_events: Sequence[Mapping[str, Any]],
    explicit_pairs: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return `(earlier_cid, later_cid)` pairs representing strict non-overlapping chronological order."""
    pairs: set[tuple[str, str]] = set()
    if explicit_pairs:
        for a, b in explicit_pairs:
            pairs.add((str(a), str(b)))

    indexed = [
        (str(ev["clause_id"]), float(ev["start_sec"]), float(ev["end_sec"]))
        for ev in gt_events
    ]
    for i in range(len(indexed)):
        cid_a, st_a, et_a = indexed[i]
        for j in range(len(indexed)):
            if i == j:
                continue
            cid_b, st_b, et_b = indexed[j]
            # Strict non-overlapping GT order; concurrent overlapping GT events allow overlap
            if et_a <= st_b + 1e-6 and (st_a < st_b - 1e-6 or et_a < et_b - 1e-6):
                pairs.add((cid_a, cid_b))
    return sorted(pairs)


def _check_candidate_full_chain(
    gt_events: Sequence[Mapping[str, Any]],
    strict_order_pairs: Sequence[tuple[str, str]],
    pred_spans: Sequence[Mapping[str, Any]],
    *,
    tiou_threshold: float,
) -> tuple[bool, int, int]:
    """Check whether one candidate prediction matches every GT event and chronological constraint."""
    span_by_cid: dict[str, Mapping[str, Any]] = {}
    for sp in pred_spans:
        cid = str(sp.get("clause_id") or "")
        if cid:
            span_by_cid[cid] = sp

    matched_events = 0
    total_events = len(gt_events)
    if total_events == 0:
        return False, 0, 0

    for gt_ev in gt_events:
        cid = str(gt_ev["clause_id"])
        sp = span_by_cid.get(cid)
        if sp is None or str(sp.get("status", "supported")) == "missing":
            continue
        ps = float(sp.get("start_sec", 0.0))
        pe = float(sp.get("end_sec", 0.0))
        if compute_tiou(ps, pe, float(gt_ev["start_sec"]), float(gt_ev["end_sec"])) >= float(tiou_threshold):
            matched_events += 1

    if matched_events < total_events:
        return False, matched_events, total_events

    for cid_a, cid_b in strict_order_pairs:
        sp_a = span_by_cid.get(cid_a)
        sp_b = span_by_cid.get(cid_b)
        if sp_a is None or sp_b is None:
            return False, matched_events, total_events
        sa, ea = float(sp_a.get("start_sec", 0.0)), float(sp_a.get("end_sec", 0.0))
        sb, eb = float(sp_b.get("start_sec", 0.0)), float(sp_b.get("end_sec", 0.0))
        if sa > sb + 1e-6 or ea > eb + 1e-6:
            return False, matched_events, total_events

    return True, matched_events, total_events


def _length_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k == 3:
        return "3"
    return ">=4"


def _normalize_query_prediction(raw_pred: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize prediction payload to `(top_k_candidate_dicts, top20_video_ids)`."""
    if raw_pred is None:
        return [], []
    if isinstance(raw_pred, (list, tuple)):
        cands = [dict(c) for c in raw_pred if isinstance(c, Mapping)]
        vids = [str(c.get("video_id") or "") for c in cands if c.get("video_id")]
        return cands, list(dict.fromkeys(vids))
    if isinstance(raw_pred, Mapping):
        if "top_k_results" in raw_pred and isinstance(raw_pred["top_k_results"], (list, tuple)):
            cands = [dict(c) for c in raw_pred["top_k_results"] if isinstance(c, Mapping)]
        elif "video_id" in raw_pred and "event_spans" in raw_pred:
            cands = [dict(raw_pred)]
        else:
            cands = []
        raw_vids = raw_pred.get("ranked_video_ids") or [
            str(c.get("video_id") or "") for c in cands if c.get("video_id")
        ]
        vids = list(dict.fromkeys(str(v) for v in raw_vids if str(v)))
        return cands, vids
    return [], []


def evaluate_activitynet_full_chain(
    locked_queries: Sequence[Mapping[str, Any]],
    predictions_by_qid: Mapping[str, Any],
    *,
    indexed_video_ids: AbstractSet[str] | None = None,
    given_video_predictions_by_qid: Mapping[str, Any] | None = None,
    tiou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate locked multi-event ActivityNet queries with fixed denominator (zero GT filtering)."""
    n_total = len(locked_queries)
    if n_total == 0:
        return {
            "total_queries": 0,
            "tiou_threshold": float(tiou_threshold),
            "full_chain_hit_at_1": 0.0,
            "full_chain_hit_at_5": 0.0,
            "given_video_full_chain_hit_at_1": 0.0,
            "video_recall_at_20": 0.0,
            "per_event_tiou_05_rate": 0.0,
            "coverage": 0.0,
            "missing_predictions": 0,
            "missing_from_index": 0,
            "by_chain_length": {"2": {}, "3": {}, ">=4": {}},
            "per_query_hits": [],
        }

    hit1_list: list[int] = []
    hit5_list: list[int] = []
    given_vid_hits: list[int] = []
    rec20_list: list[int] = []
    total_events_all = 0
    matched_events_all = 0
    missing_preds = 0
    missing_index = 0
    covered = 0

    bucket_queries: dict[str, int] = defaultdict(int)
    bucket_chain_hits: dict[str, int] = defaultdict(int)
    bucket_ev_total: dict[str, int] = defaultdict(int)
    bucket_ev_hits: dict[str, int] = defaultdict(int)

    for q in locked_queries:
        qid = str(q["query_id"])
        gt_vid = str(q["video_id"])
        gt_events = list(q.get("events") or [])
        n_ev = len(gt_events)
        b_key = _length_bucket(n_ev)
        bucket_queries[b_key] += 1
        bucket_ev_total[b_key] += n_ev
        total_events_all += n_ev

        order_pairs = _extract_gt_strict_order_pairs(
            gt_events, explicit_pairs=q.get("order_pairs")
        )

        vid_indexed = (indexed_video_ids is None) or (gt_vid in indexed_video_ids)
        if not vid_indexed:
            missing_index += 1

        cands, top_vids = _normalize_query_prediction(predictions_by_qid.get(qid))
        if not cands and not top_vids:
            missing_preds += 1
        elif vid_indexed:
            covered += 1

        rec20 = 1 if (vid_indexed and gt_vid in top_vids[:20]) else 0
        rec20_list.append(rec20)

        # Top-1 full-chain evaluation
        h1 = 0
        top1_ev_hits = 0
        if vid_indexed and cands:
            c1 = cands[0]
            if str(c1.get("video_id") or "") == gt_vid:
                ok1, m1_cnt, _ = _check_candidate_full_chain(
                    gt_events,
                    order_pairs,
                    c1.get("event_spans") or (),
                    tiou_threshold=tiou_threshold,
                )
                top1_ev_hits = m1_cnt
                h1 = 1 if ok1 else 0

        # Top-5 independent evaluation: at least one of top-5 results has right video AND full chain
        h5 = 0
        if vid_indexed and cands:
            for c_i in cands[:5]:
                if str(c_i.get("video_id") or "") == gt_vid:
                    ok_i, _, _ = _check_candidate_full_chain(
                        gt_events,
                        order_pairs,
                        c_i.get("event_spans") or (),
                        tiou_threshold=tiou_threshold,
                    )
                    if ok_i:
                        h5 = 1
                        break

        # Given-video full-chain evaluation
        gv_h = 0
        if vid_indexed:
            gv_raw = (
                given_video_predictions_by_qid.get(qid)
                if given_video_predictions_by_qid is not None
                else None
            )
            gv_cands, _ = _normalize_query_prediction(gv_raw)
            target_cand = next(
                (c for c in gv_cands if str(c.get("video_id") or gt_vid) == gt_vid),
                None,
            )
            if target_cand is None and cands:
                target_cand = next(
                    (c for c in cands if str(c.get("video_id") or "") == gt_vid),
                    None,
                )
            if target_cand is not None:
                ok_gv, _, _ = _check_candidate_full_chain(
                    gt_events,
                    order_pairs,
                    target_cand.get("event_spans") or (),
                    tiou_threshold=tiou_threshold,
                )
                gv_h = 1 if ok_gv else 0

        hit1_list.append(h1)
        hit5_list.append(h5)
        given_vid_hits.append(gv_h)
        matched_events_all += top1_ev_hits
        bucket_chain_hits[b_key] += h1
        bucket_ev_hits[b_key] += top1_ev_hits

    by_len: dict[str, dict[str, Any]] = {}
    for bk in ("2", "3", ">=4"):
        qn = bucket_queries[bk]
        en = bucket_ev_total[bk]
        by_len[bk] = {
            "queries": qn,
            "full_chain_hit_at_1": round(bucket_chain_hits[bk] / max(1, qn), 6) if qn else 0.0,
            "events": en,
            "per_event_tiou_05_rate": round(bucket_ev_hits[bk] / max(1, en), 6) if en else 0.0,
        }

    return {
        "total_queries": n_total,
        "tiou_threshold": float(tiou_threshold),
        "full_chain_hit_at_1": round(sum(hit1_list) / float(n_total), 6),
        "full_chain_hit_at_1_count": int(sum(hit1_list)),
        "full_chain_hit_at_5": round(sum(hit5_list) / float(n_total), 6),
        "given_video_full_chain_hit_at_1": round(sum(given_vid_hits) / float(n_total), 6),
        "video_recall_at_20": round(sum(rec20_list) / float(n_total), 6),
        "per_event_tiou_05_rate": round(matched_events_all / max(1, total_events_all), 6),
        "coverage": round(covered / float(n_total), 6),
        "missing_predictions": int(missing_preds),
        "missing_from_index": int(missing_index),
        "by_chain_length": by_len,
        "per_query_hits": hit1_list,
    }


def paired_source_group_bootstrap_ci(
    hits_a: Sequence[int | float],
    hits_b: Sequence[int | float],
    source_group_ids: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Paired cluster bootstrap on `source_group_id` for metric deltas (`hits_a - hits_b`)."""
    n = len(hits_a)
    if n == 0 or len(hits_b) != n or len(source_group_ids) != n:
        raise ValueError("hits_a, hits_b, and source_group_ids must have equal non-zero length")

    a_arr = np.asarray(hits_a, dtype=np.float64)
    b_arr = np.asarray(hits_b, dtype=np.float64)
    delta_obs = float(np.mean(a_arr - b_arr))
    rescued = int(np.sum((a_arr > b_arr)))
    lost = int(np.sum((b_arr > a_arr)))

    by_group: dict[str, tuple[float, int]] = {}
    grp_sums: dict[str, float] = defaultdict(float)
    grp_cnts: dict[str, int] = defaultdict(int)
    for idx, gid in enumerate(source_group_ids):
        gs = str(gid)
        grp_sums[gs] += float(a_arr[idx] - b_arr[idx])
        grp_cnts[gs] += 1
    groups = sorted(grp_cnts.keys())
    g_sum_arr = np.asarray([grp_sums[g] for g in groups], dtype=np.float64)
    g_cnt_arr = np.asarray([grp_cnts[g] for g in groups], dtype=np.float64)
    by_group.clear()

    rng = np.random.default_rng(int(seed))
    n_groups = len(groups)
    boot_deltas = np.empty(int(n_resamples), dtype=np.float64)
    for b_idx in range(int(n_resamples)):
        sample_idx = rng.integers(0, n_groups, size=n_groups)
        tot_cnt = float(np.sum(g_cnt_arr[sample_idx]))
        boot_deltas[b_idx] = float(np.sum(g_sum_arr[sample_idx])) / max(1.0, tot_cnt)

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    return {
        "n_queries": n,
        "n_groups": n_groups,
        "delta": round(delta_obs, 6),
        "ci95_low": round(float(ci_low), 6),
        "ci95_high": round(float(ci_high), 6),
        "rescued": rescued,
        "lost": lost,
        "net": rescued - lost,
        "significant_gt_zero": bool(float(ci_low) > 0.0),
    }


def evaluate_videocomp(
    pairs: Sequence[Mapping[str, Any]],
    scores_by_pair_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate VideoComp pairs across domains and disruption types with explicit ties + `all` product."""
    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "ties": 0, "missing": 0}
    )
    overall_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "ties": 0, "missing": 0}
    )

    domains: set[str] = set()
    for pair in pairs:
        pid = str(pair.get("pair_id") or pair.get("key") or "")
        dom = str(pair.get("domain") or "unknown")
        ptype = str(pair.get("type") or "")
        domains.add(dom)
        b_dom = buckets[(dom, ptype)]
        b_all = overall_type[ptype]
        b_dom["total"] += 1
        b_all["total"] += 1

        sc = scores_by_pair_id.get(pid)
        if sc is None or "positive_score" not in sc or "negative_score" not in sc:
            b_dom["missing"] += 1
            b_all["missing"] += 1
            continue
        pos = float(sc["positive_score"])
        neg = float(sc["negative_score"])
        if pos > neg:
            b_dom["correct"] += 1
            b_all["correct"] += 1
        elif pos == neg:
            b_dom["ties"] += 1
            b_all["ties"] += 1

    def _summarize_group(counts_by_type: dict[str, dict[str, int]]) -> dict[str, Any]:
        out_t: dict[str, Any] = {}
        all_prod = 1.0
        for t in VIDEOCOMP_DISRUPTION_TYPES:
            c = counts_by_type.get(t, {"total": 0, "correct": 0, "ties": 0, "missing": 0})
            acc = round(c["correct"] / float(c["total"]), 6) if c["total"] > 0 else 0.0
            out_t[t] = {
                "total": c["total"],
                "correct": c["correct"],
                "ties": c["ties"],
                "missing": c["missing"],
                "accuracy": acc,
            }
            all_prod *= acc
        return {
            **out_t,
            "all": round(float(all_prod), 6),
        }

    by_domain: dict[str, Any] = {}
    for dom in sorted(domains):
        sub = {t: buckets[(dom, t)] for t in VIDEOCOMP_DISRUPTION_TYPES if (dom, t) in buckets}
        by_domain[dom] = _summarize_group(sub)

    return {
        "total_pairs": len(pairs),
        "by_domain": by_domain,
        "overall": _summarize_group(overall_type),
    }
