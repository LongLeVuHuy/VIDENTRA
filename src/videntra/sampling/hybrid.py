"""Hybrid adaptive frame sampling with multi-strategy selection, quality lookahead, and leak-safe signals."""
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, Sequence
import uuid

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from videntra.config.ingestion import (
    SamplingConfig,
    VALID_SAMPLING_STRATEGIES,
    normalize_sampling_strategy,
)
from videntra.ingestion.decode import DecodedFrame


# 32x32 DCT transform matrix for 64-bit pHash
_N = 32
_I = np.arange(_N)
_J = np.arange(_N)
_DCT_MATRIX = np.cos(np.pi / _N * (_I[:, None] + 0.5) * _J[None, :])
_DCT_MATRIX_T = np.ascontiguousarray(_DCT_MATRIX.T)


_GRID_CELL_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _rgb_to_luma(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8/float image to uint8 or float32 BT.601 luma plane using C++ SIMD when uint8."""
    if image_rgb.dtype == np.uint8 and image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        return cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY)
    return (
        0.299 * image_rgb[:, :, 0].astype(np.float32)
        + 0.587 * image_rgb[:, :, 1].astype(np.float32)
        + 0.114 * image_rgb[:, :, 2].astype(np.float32)
    )


def _compute_phash_64_from_luma(luma: np.ndarray) -> int:
    """Compute 64-bit DCT perceptual hash from 2D uint8/float32 luma plane using all 63 AC coefficients."""
    h, w = luma.shape[:2]
    if h != 32 or w != 32:
        gray_32 = cv2.resize(luma, (32, 32), interpolation=cv2.INTER_AREA)
    else:
        gray_32 = luma
    dct_2d = _DCT_MATRIX_T @ gray_32.astype(np.float64, copy=False) @ _DCT_MATRIX
    low_freq = dct_2d[:8, :8]
    ac_coeffs = low_freq.reshape(-1)[1:]  # all 63 AC coefficients (excluding DC (0,0))
    med = float(np.median(ac_coeffs))
    bits = (low_freq > med).reshape(-1)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def compute_phash_64(image_rgb: np.ndarray) -> int:
    """Compute 64-bit DCT perceptual hash from RGB uint8 image."""
    return _compute_phash_64_from_luma(_rgb_to_luma(image_rgb))


def phash_hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def resize_lowres(image_rgb: np.ndarray, max_dim: int = 256) -> np.ndarray:
    """Resize RGB image with area anti-aliasing so its maximum dimension does not exceed max_dim."""
    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0 or max(h, w) <= max_dim:
        return image_rgb
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_luma_difference_from_luma(y1: np.ndarray, y2: np.ndarray) -> float:
    """Mean absolute difference normalized in [0, 1] between two luma planes."""
    if y1.shape != y2.shape:
        y2 = cv2.resize(y2, (y1.shape[1], y1.shape[0]), interpolation=cv2.INTER_AREA)
    if y1.dtype == y2.dtype and y1.dtype in (np.uint8, np.float32):
        return float(np.mean(cv2.absdiff(y1, y2)) / 255.0)
    return float(np.mean(np.abs(y1.astype(np.float32) - y2.astype(np.float32))) / 255.0)


def compute_luma_difference(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """Mean absolute luma difference normalized in [0, 1]."""
    return compute_luma_difference_from_luma(_rgb_to_luma(img1_rgb), _rgb_to_luma(img2_rgb))


def compute_grid_hist_512(img_rgb: np.ndarray) -> np.ndarray:
    """Compute normalized 512-bin color histograms across a 4x4 spatial grid in one vectorized pass."""
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((16, 512), dtype=np.float32)

    cell_shifted = _GRID_CELL_CACHE.get((h, w))
    if cell_shifted is None:
        cell_h = max(1, h // 4)
        cell_w = max(1, w // 4)
        gy = np.minimum(np.arange(h, dtype=np.uint16) // cell_h, 3)
        gx = np.minimum(np.arange(w, dtype=np.uint16) // cell_w, 3)
        cell_shifted = ((gy[:, None] * 4 + gx[None, :]) << 9).reshape(-1)
        if len(_GRID_CELL_CACHE) < 64:
            _GRID_CELL_CACHE[(h, w)] = cell_shifted

    r_q = (img_rgb[:, :, 0] >> 5).astype(np.uint16).reshape(-1)
    g_q = (img_rgb[:, :, 1] >> 5).astype(np.uint16).reshape(-1)
    b_q = (img_rgb[:, :, 2] >> 5).astype(np.uint16).reshape(-1)
    color_idx = (r_q << 6) | (g_q << 3) | b_q

    combined = cell_shifted | color_idx
    counts = np.bincount(combined, minlength=16 * 512).astype(np.float32).reshape(16, 512)
    sums = counts.sum(axis=1, keepdims=True)
    np.divide(counts, sums, out=counts, where=sums > 0)
    return counts


def compute_hist_distance_from_hists(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute normalized 4x4 grid histogram intersection distance from precomputed histograms."""
    mean_inter = float(np.minimum(h1, h2).sum()) / float(h1.shape[0] if h1.ndim == 2 and h1.shape[0] > 0 else 16)
    return float(np.clip(1.0 - mean_inter, 0.0, 1.0))


def compute_hist_distance(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """Normalized 4x4 spatial grid color histogram intersection distance in [0, 1]."""
    return compute_hist_distance_from_hists(compute_grid_hist_512(img1_rgb), compute_grid_hist_512(img2_rgb))


def _compute_sharpness_from_luma(y: np.ndarray) -> float:
    """Variance of 2D discrete Laplacian operator computed on luma plane."""
    h, w = y.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    return float(cv2.Laplacian(y, cv2.CV_32F, ksize=1).var())


def compute_sharpness_score(img_rgb: np.ndarray) -> float:
    """Variance of 2D discrete Laplacian operator as sharpness metric."""
    return _compute_sharpness_from_luma(_rgb_to_luma(img_rgb))


def is_blank_or_fade_frame(img_rgb: np.ndarray, min_luma_std: float = 8.0) -> bool:
    """Returns True if the frame has low luma standard deviation (e.g. solid black or white)."""
    return float(np.std(_rgb_to_luma(img_rgb))) < min_luma_std


def _compute_compact_hist_64(img_rgb: np.ndarray) -> np.ndarray:
    """Compute lightweight 256-dim (2x2 spatial grid x 64 RGB bins) normalized descriptor for fast O(N^2) medoid scoring."""
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(256, dtype=np.float32)
    cy = max(1, h // 2)
    cx = max(1, w // 2)
    gy = np.minimum(np.arange(h, dtype=np.int32) // cy, 1)
    gx = np.minimum(np.arange(w, dtype=np.int32) // cx, 1)
    cell = (gy[:, None] * 2 + gx[None, :]).reshape(-1)
    r_q = img_rgb[:, :, 0].astype(np.int32) >> 6
    g_q = img_rgb[:, :, 1].astype(np.int32) >> 6
    b_q = img_rgb[:, :, 2].astype(np.int32) >> 6
    color_idx = ((r_q << 4) | (g_q << 2) | b_q).reshape(-1)
    counts = np.bincount((cell << 6) | color_idx, minlength=256).astype(np.float32)
    s = float(counts.sum())
    if s > 0:
        counts /= s
    return counts


@dataclass(slots=True)
class _FrameSignalEntry:
    """Bounded-memory per-frame visual summary independent of raw pixel lifetime."""

    ordinal: int
    pts: int | None
    timestamp_us: int | None
    timestamp_seconds: float | None
    epoch_id: str
    phash_int: int
    sharpness: float
    luma_std: float
    is_blank: bool
    lowres_luma: np.ndarray | None
    grid_hist: np.ndarray | None
    compact_hist: np.ndarray | None = None


@dataclass(frozen=True)
class SampleRecord:
    sampled_frame_id: str
    video_id: str
    timeline_revision: str
    sampler_revision: str
    original_decoded_frame_index: int
    timestamp_us: int | None
    timestamp_seconds: float | None
    reasons: tuple[str, ...]
    is_mandatory: bool
    phash: str
    scene_id: str
    epoch_id: str = "epoch_0"
    pts: int | None = None
    sharpness: float | None = None


@dataclass(frozen=True)
class SampleManifest:
    video_id: str
    timeline_revision: str
    sampler_revision: str
    samples: tuple[SampleRecord, ...]
    total_frames_examined: int
    mandatory_anchor_count: int
    soft_sample_count: int
    max_gap_violations: int
    overflow_count: int
    shot_cut_ordinals: tuple[int, ...] = ()

    def to_parquet(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("sampled_frame_id", pa.string()),
            ("video_id", pa.string()),
            ("timeline_revision", pa.string()),
            ("sampler_revision", pa.string()),
            ("original_decoded_frame_index", pa.int64()),
            ("timestamp_us", pa.int64()),
            ("timestamp_seconds", pa.float64()),
            ("reasons", pa.list_(pa.string())),
            ("is_mandatory", pa.bool_()),
            ("phash", pa.string()),
            ("scene_id", pa.string()),
            ("epoch_id", pa.string()),
            ("pts", pa.int64()),
            ("sharpness", pa.float64()),
        ])
        data = {name: [] for name in schema.names}
        for s in self.samples:
            data["sampled_frame_id"].append(s.sampled_frame_id)
            data["video_id"].append(s.video_id)
            data["timeline_revision"].append(s.timeline_revision)
            data["sampler_revision"].append(s.sampler_revision)
            data["original_decoded_frame_index"].append(s.original_decoded_frame_index)
            data["timestamp_us"].append(s.timestamp_us)
            data["timestamp_seconds"].append(s.timestamp_seconds)
            data["reasons"].append(list(s.reasons))
            data["is_mandatory"].append(s.is_mandatory)
            data["phash"].append(s.phash)
            data["scene_id"].append(s.scene_id)
            data["epoch_id"].append(s.epoch_id)
            data["pts"].append(s.pts)
            data["sharpness"].append(s.sharpness)
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, str(p), compression="zstd")


def make_sample_id(video_id: str, timeline_revision: str, ordinal: int, sampler_rev: str) -> str:
    key = f"{video_id}:{timeline_revision}:{ordinal}:{sampler_rev}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


class HybridAdaptiveSampler:
    """Multi-strategy keyframe sampler supporting legacy v1, sharpness lookahead, shot medoid, and v2 adaptive."""

    def __init__(
        self,
        config: SamplingConfig,
        video_id: str,
        timeline_revision: str,
        sampler_revision: str | None = None,
    ):
        config.validate()
        self.cfg = config
        self.video_id = video_id
        self.timeline_revision = timeline_revision
        norm_strategy = normalize_sampling_strategy(config.strategy)
        self.strategy = norm_strategy if norm_strategy in VALID_SAMPLING_STRATEGIES else "hybrid_adaptive"
        self.sampler_revision = sampler_revision or f"videntra_{self.strategy}"

        self.last_selected_ts_sec: float | None = None
        self.last_selected_phash: int | None = None
        self.last_selected_hist: np.ndarray | None = None
        self.prev_entry: _FrameSignalEntry | None = None
        self.pre_cut_hist: np.ndarray | None = None
        self.last_cut_ts_sec: float | None = None
        self.last_content_entry: _FrameSignalEntry | None = None

        self.scene_idx = 0
        self._samples_by_ordinal: dict[int, SampleRecord] = {}
        self.recent_phashes: deque[int] = deque(maxlen=5)
        self.pending_first_valid_anchor: bool = False
        self.pending_first_entry: _FrameSignalEntry | None = None

        self.d_history: deque[tuple[float, float]] = deque()
        self.ring_buffer: deque[_FrameSignalEntry] = deque()
        self.pending_burst_targets: list[float] = []
        self.soft_sample_timestamps: deque[float] = deque()
        self.last_burst_trigger_ts: float = -999.0
        self.last_soft_commit_ts: float = -999.0
        self.overflow_count = 0
        self.total_frames = 0
        self.shot_cut_ordinals: set[int] = set()
        self.current_epoch_id: str | None = None

        # Causal lookahead buffer (used by sharpness_lookahead and hybrid_adaptive)
        self._la_entries: list[_FrameSignalEntry] = []
        self._la_reasons: set[str] = set()
        self._la_mandatory: bool = False
        self._la_start_ts: float | None = None

        # Segment / timeline buffers (used by shot_medoid_submodular and uniform_matched)
        self._segment_entries: list[_FrameSignalEntry] = []
        self._uniform_entries: list[_FrameSignalEntry] = []

    @property
    def samples(self) -> list[SampleRecord]:
        return sorted(self._samples_by_ordinal.values(), key=lambda s: s.original_decoded_frame_index)

    def _extract_signal(self, frame: DecodedFrame, *, include_compact: bool = False) -> _FrameSignalEntry:
        """Compute bounded-memory visual features once per frame; immune to downstream pixel clearing."""
        if frame.pixels is None:
            return _FrameSignalEntry(
                ordinal=frame.ordinal,
                pts=frame.pts,
                timestamp_us=frame.timestamp_us,
                timestamp_seconds=frame.timestamp_seconds,
                epoch_id=frame.epoch_id,
                phash_int=0,
                sharpness=0.0,
                luma_std=0.0,
                is_blank=True,
                lowres_luma=None,
                grid_hist=None,
                compact_hist=None,
            )

        lowres_rgb = resize_lowres(frame.pixels, self.cfg.lowres_max_dim)
        luma = _rgb_to_luma(lowres_rgb)
        luma_std = float(np.std(luma))
        grid_hist = compute_grid_hist_512(lowres_rgb)
        compact_hist = _compute_compact_hist_64(lowres_rgb) if include_compact else None

        return _FrameSignalEntry(
            ordinal=frame.ordinal,
            pts=frame.pts,
            timestamp_us=frame.timestamp_us,
            timestamp_seconds=frame.timestamp_seconds,
            epoch_id=frame.epoch_id,
            phash_int=_compute_phash_64_from_luma(luma),
            sharpness=_compute_sharpness_from_luma(luma),
            luma_std=luma_std,
            is_blank=luma_std < self.cfg.min_luma_std,
            lowres_luma=luma,
            grid_hist=grid_hist,
            compact_hist=compact_hist,
        )

    def _update_soft_quota(self, now_sec: float) -> bool:
        """Enforce rolling soft target <= soft_target_fps (duration-bounded for hybrid_adaptive)."""
        window_start = now_sec - self.cfg.soft_window_seconds
        while self.soft_sample_timestamps and self.soft_sample_timestamps[0] < window_start:
            self.soft_sample_timestamps.popleft()
        if self.strategy == "hybrid_adaptive":
            eff_window = min(self.cfg.soft_window_seconds, max(2.0, now_sec + 1.0))
            max_allowed = math.ceil(self.cfg.soft_target_fps * eff_window)
        else:
            max_allowed = self.cfg.soft_target_fps * self.cfg.soft_window_seconds
        return len(self.soft_sample_timestamps) < max_allowed

    def _min_recent_phash_distance(self, curr_phash: int) -> int:
        if self.recent_phashes:
            return min(phash_hamming_distance(h, curr_phash) for h in self.recent_phashes)
        if self.last_selected_phash is not None:
            return phash_hamming_distance(self.last_selected_phash, curr_phash)
        return 999

    def _upsert_sample(
        self,
        entry: _FrameSignalEntry,
        additional_reasons: set[str],
        make_mandatory: bool = False,
        scene_idx_override: int | None = None,
    ) -> SampleRecord:
        existing = self._samples_by_ordinal.get(entry.ordinal)
        if existing is not None:
            merged_reasons = tuple(sorted(set(existing.reasons) | additional_reasons))
            rec = SampleRecord(
                sampled_frame_id=existing.sampled_frame_id,
                video_id=existing.video_id,
                timeline_revision=existing.timeline_revision,
                sampler_revision=existing.sampler_revision,
                original_decoded_frame_index=existing.original_decoded_frame_index,
                timestamp_us=existing.timestamp_us,
                timestamp_seconds=existing.timestamp_seconds,
                reasons=merged_reasons,
                is_mandatory=existing.is_mandatory or make_mandatory,
                phash=existing.phash,
                scene_id=existing.scene_id,
                epoch_id=existing.epoch_id,
                pts=existing.pts if existing.pts is not None else entry.pts,
                sharpness=existing.sharpness if existing.sharpness is not None else round(entry.sharpness, 4),
            )
            self._samples_by_ordinal[entry.ordinal] = rec
            return rec

        sc_idx = self.scene_idx if scene_idx_override is None else scene_idx_override
        s_id = make_sample_id(self.video_id, self.timeline_revision, entry.ordinal, self.sampler_revision)
        rec = SampleRecord(
            sampled_frame_id=s_id,
            video_id=self.video_id,
            timeline_revision=self.timeline_revision,
            sampler_revision=self.sampler_revision,
            original_decoded_frame_index=entry.ordinal,
            timestamp_us=entry.timestamp_us,
            timestamp_seconds=entry.timestamp_seconds,
            reasons=tuple(sorted(additional_reasons)),
            is_mandatory=make_mandatory,
            phash=f"{entry.phash_int:016x}",
            scene_id=f"scene_{sc_idx:04d}",
            epoch_id=entry.epoch_id,
            pts=entry.pts,
            sharpness=round(entry.sharpness, 4),
        )
        self._samples_by_ordinal[entry.ordinal] = rec
        return rec

    def _commit_selected(
        self,
        entry: _FrameSignalEntry,
        reasons: set[str],
        is_mandatory: bool,
        *,
        bypass_soft_gates: bool = False,
    ) -> SampleRecord | None:
        """Apply pHash dedup + soft quota for non-mandatory samples and commit to manifest."""
        if not reasons:
            return None

        t_sec = entry.timestamp_seconds
        if not is_mandatory and not bypass_soft_gates:
            if entry.is_blank:
                return None
            phash_dist = (
                phash_hamming_distance(self.last_selected_phash, entry.phash_int)
                if self.last_selected_phash is not None
                else 999
            )
            if self.strategy != "hybrid_legacy" and self.recent_phashes:
                phash_dist = min(phash_dist, min(phash_hamming_distance(h, entry.phash_int) for h in self.recent_phashes))
            if phash_dist <= self.cfg.phash_min_distance:
                return None
            if self.strategy == "hybrid_adaptive" and t_sec is not None:
                min_soft_gap = min(0.45, 0.5 / max(0.5, self.cfg.soft_target_fps))
                if (t_sec - self.last_soft_commit_ts) < min_soft_gap and "burst" not in reasons:
                    return None
            if t_sec is not None:
                if not self._update_soft_quota(t_sec):
                    self.overflow_count += 1
                    return None
                self.soft_sample_timestamps.append(t_sec)
                if not is_mandatory:
                    self.last_soft_commit_ts = t_sec

        rec = self._upsert_sample(entry, reasons, make_mandatory=is_mandatory)
        if t_sec is not None:
            self.last_selected_ts_sec = t_sec
        self.last_selected_phash = entry.phash_int
        if entry.grid_hist is not None:
            self.last_selected_hist = entry.grid_hist
        self.recent_phashes.append(entry.phash_int)
        return rec

    def _effective_lookahead_seconds(self) -> float:
        if self.strategy not in ("sharpness_lookahead", "hybrid_adaptive"):
            return 0.0
        return min(self.cfg.lookahead_window_seconds, 0.45 * self.cfg.max_gap_seconds)

    def _score_lookahead_candidate(
        self,
        cand: _FrameSignalEntry,
        mean_hist: np.ndarray | None,
    ) -> float:
        if self.strategy == "sharpness_lookahead":
            return cand.sharpness
        # hybrid_adaptive composite quality: sharp & clear & representative
        sharp_log = math.log1p(max(0.0, cand.sharpness))
        clarity = min(1.0, max(0.4, cand.luma_std / 32.0))
        centrality = 1.0
        if mean_hist is not None and cand.grid_hist is not None:
            centrality = 1.0 - compute_hist_distance_from_hists(cand.grid_hist, mean_hist)
        return sharp_log * clarity * (0.65 + 0.35 * centrality)

    def _flush_lookahead_buffer(self) -> SampleRecord | None:
        """Resolve any pending lookahead trigger window by picking the best candidate frame."""
        if not self._la_entries:
            self._la_reasons.clear()
            self._la_mandatory = False
            self._la_start_ts = None
            return None

        cands = self._la_entries
        non_blank = [c for c in cands if not c.is_blank]
        pool = non_blank if non_blank else cands

        mean_hist: np.ndarray | None = None
        if self.strategy == "hybrid_adaptive" and len(pool) >= 3:
            hists = [c.grid_hist for c in pool if c.grid_hist is not None]
            if hists:
                mean_hist = np.mean(hists, axis=0)

        scored = sorted(
            pool,
            key=lambda c: (self._score_lookahead_candidate(c, mean_hist), -c.ordinal),
            reverse=True,
        )

        chosen = scored[0]
        if not self._la_mandatory and self.last_selected_phash is not None:
            # Prefer a non-duplicate candidate if top candidate duplicates a recent pHash
            for cand in scored:
                if self._min_recent_phash_distance(cand.phash_int) > self.cfg.phash_min_distance:
                    chosen = cand
                    break

        rec = self._commit_selected(chosen, set(self._la_reasons), self._la_mandatory)
        self._la_entries.clear()
        self._la_reasons.clear()
        self._la_mandatory = False
        self._la_start_ts = None
        return rec

    def _select_pre_burst(self, trigger_t: float, curr_ordinal: int) -> None:
        step = 1.0 / self.cfg.burst_fps
        pre_steps = int(round(self.cfg.burst_pre_seconds * self.cfg.burst_fps))
        pre_targets = [trigger_t - step * k for k in range(1, pre_steps + 1) if (trigger_t - step * k) >= 0]
        tolerance = 0.5 / self.cfg.burst_fps
        for target_t in pre_targets:
            best_f: _FrameSignalEntry | None = None
            best_diff = float("inf")
            best_sharp = -1.0
            for bf in self.ring_buffer:
                if bf.timestamp_seconds is not None and bf.ordinal != curr_ordinal:
                    diff = abs(bf.timestamp_seconds - target_t)
                    if diff <= tolerance:
                        if self.strategy == "hybrid_legacy":
                            take = diff < best_diff or (abs(diff - best_diff) < 1e-4 and bf.sharpness > best_sharp)
                        else:
                            take = (not bf.is_blank and bf.sharpness > best_sharp) or (
                                best_f is None and diff < best_diff
                            )
                        if take:
                            best_diff = diff
                            best_sharp = bf.sharpness
                            best_f = bf
            if best_f is None or best_f.ordinal == curr_ordinal:
                continue
            if self.strategy != "hybrid_legacy":
                # Respect blank guard, pHash dedup, and soft quota for non-legacy pre-burst
                if best_f.is_blank:
                    continue
                if best_f.ordinal not in self._samples_by_ordinal:
                    if self._min_recent_phash_distance(best_f.phash_int) <= self.cfg.phash_min_distance:
                        continue
                    if best_f.timestamp_seconds is not None and not self._update_soft_quota(best_f.timestamp_seconds):
                        self.overflow_count += 1
                        continue
                    if best_f.timestamp_seconds is not None:
                        self.soft_sample_timestamps.append(best_f.timestamp_seconds)
                    self.recent_phashes.append(best_f.phash_int)
            self._upsert_sample(best_f, {"burst"}, make_mandatory=False)

    def _process_frame_uniform(self, frame: DecodedFrame) -> None:
        entry = self._extract_signal(frame, include_compact=False)
        self._uniform_entries.append(entry)
        self.prev_entry = entry

    def _select_medoids_for_segment(
        self,
        segment: list[_FrameSignalEntry],
        segment_scene_idx: int,
        *,
        is_first_segment: bool = False,
        is_cut_end: bool = False,
        is_last_segment: bool = False,
    ) -> None:
        """Select keyframes inside one shot/max-gap-bounded segment using Facility-Location representativeness."""
        if not segment:
            return

        anchor_indices: list[int] = []
        if is_first_segment:
            first_idx = 0
            if segment[0].is_blank:
                for idx_i, ent in enumerate(segment):
                    if ent.timestamp_seconds is not None and ent.timestamp_seconds <= 1.0 and not ent.is_blank:
                        first_idx = idx_i
                        break
            self._upsert_sample(segment[first_idx], {"first"}, make_mandatory=True, scene_idx_override=segment_scene_idx)
            anchor_indices.append(first_idx)
            self.last_selected_ts_sec = segment[first_idx].timestamp_seconds
            self.last_selected_phash = segment[first_idx].phash_int
            self.recent_phashes.append(segment[first_idx].phash_int)

        if len(segment) > 1 and (segment[0].ordinal not in self._samples_by_ordinal) and segment[0].ordinal in self.shot_cut_ordinals:
            self._upsert_sample(segment[0], {"cut_after"}, make_mandatory=True, scene_idx_override=segment_scene_idx)
            anchor_indices.append(0)
            self.last_selected_ts_sec = segment[0].timestamp_seconds
            self.last_selected_phash = segment[0].phash_int
            self.recent_phashes.append(segment[0].phash_int)

        # Build compact descriptor matrix (N, 256) for Facility Location gain
        comp_vecs = np.stack([
            e.compact_hist if e.compact_hist is not None else np.zeros(256, dtype=np.float32)
            for e in segment
        ], axis=0)
        sim_mat = np.minimum(comp_vecs[:, None, :], comp_vecs[None, :, :]).sum(axis=-1)
        qualities = np.array([
            (0.65 + 0.35 * min(1.0, math.log1p(max(0.0, e.sharpness)) / 6.0))
            * (min(1.0, max(0.35, e.luma_std / 32.0)))
            * (1.0 if not e.is_blank else 0.1)
            for e in segment
        ], dtype=np.float32)

        # Initialize best covered similarity from any anchors already picked in this segment
        covered_sim = np.zeros(len(segment), dtype=np.float32)
        for a_idx in anchor_indices:
            covered_sim = np.maximum(covered_sim, sim_mat[a_idx])

        # Enforce max_gap anchors using Facility-Location within feasible intervals
        static_gap = self.cfg.max_static_gap_seconds or self.cfg.max_gap_seconds
        pos = 0
        while pos < len(segment):
            t_cur = segment[pos].timestamp_seconds
            if self.last_selected_ts_sec is not None and t_cur is not None:
                gap = t_cur - self.last_selected_ts_sec
                if gap >= self.cfg.max_gap_seconds:
                    curr_static = self._min_recent_phash_distance(segment[pos].phash_int) <= self.cfg.phash_min_distance
                    eff_gap = static_gap if (curr_static and static_gap > self.cfg.max_gap_seconds) else self.cfg.max_gap_seconds
                    if gap >= eff_gap:
                        # Feasible window around pos respecting gap tolerance (+0.15s lookahead or exact)
                        hi_t = t_cur + min(0.15, 0.45 * self.cfg.max_gap_seconds)
                        feasible = [
                            j for j in range(pos, len(segment))
                            if segment[j].timestamp_seconds is not None and segment[j].timestamp_seconds <= hi_t
                        ] or [pos]
                        non_blank_feas = [j for j in feasible if not segment[j].is_blank] or feasible
                        best_j = max(
                            non_blank_feas,
                            key=lambda j: float(np.maximum(0.0, sim_mat[j] - covered_sim).sum()) * float(qualities[j]),
                        )
                        winner = segment[best_j]
                        self._upsert_sample(winner, {"max_gap"}, make_mandatory=True, scene_idx_override=segment_scene_idx)
                        covered_sim = np.maximum(covered_sim, sim_mat[best_j])
                        self.last_selected_ts_sec = winner.timestamp_seconds
                        self.last_selected_phash = winner.phash_int
                        self.recent_phashes.append(winner.phash_int)
                        pos = best_j + 1
                        continue
            pos += 1

        # Add 1 interior medoid if the shot is moderately long (>= 0.8s) and has rich unrepresented visual content
        t0 = segment[0].timestamp_seconds
        t1 = segment[-1].timestamp_seconds
        if len(segment) >= 4 and t0 is not None and t1 is not None and (t1 - t0) >= 0.8:
            mean_sim_gain = np.maximum(0.0, sim_mat - covered_sim[None, :]).mean(axis=1) * qualities
            best_medoid = int(np.argmax(mean_sim_gain))
            cand = segment[best_medoid]
            if (
                cand.ordinal not in self._samples_by_ordinal
                and not cand.is_blank
                and float(mean_sim_gain[best_medoid]) >= 0.08
                and self._min_recent_phash_distance(cand.phash_int) > self.cfg.phash_min_distance
                and (cand.timestamp_seconds is None or self._update_soft_quota(cand.timestamp_seconds))
            ):
                if cand.timestamp_seconds is not None:
                    self.soft_sample_timestamps.append(cand.timestamp_seconds)
                self._upsert_sample(cand, {"change_trigger"}, make_mandatory=False, scene_idx_override=segment_scene_idx)
                self.last_selected_ts_sec = cand.timestamp_seconds
                self.last_selected_phash = cand.phash_int
                self.recent_phashes.append(cand.phash_int)

        if is_cut_end:
            end_ent = segment[-1]
            self.shot_cut_ordinals.add(end_ent.ordinal)
            self._upsert_sample(end_ent, {"cut_before"}, make_mandatory=True, scene_idx_override=segment_scene_idx)
            self.last_selected_ts_sec = end_ent.timestamp_seconds
            self.last_selected_phash = end_ent.phash_int
            self.recent_phashes.append(end_ent.phash_int)

        if is_last_segment:
            end_ent = segment[-1]
            if end_ent.is_blank and self.last_content_entry is not None:
                lc = self.last_content_entry
                if (
                    lc.ordinal not in self._samples_by_ordinal
                    and ((lc.timestamp_seconds or 0.0) - (self.last_selected_ts_sec or -999.0)) >= 0.4
                    and self._min_recent_phash_distance(lc.phash_int) > self.cfg.phash_min_distance
                ):
                    self._upsert_sample(lc, {"last_content"}, make_mandatory=False, scene_idx_override=segment_scene_idx)
            self._upsert_sample(end_ent, {"last"}, make_mandatory=True, scene_idx_override=segment_scene_idx)

    def _is_true_shot_cut(self, entry: _FrameSignalEntry, d_i: float, c_i: float) -> bool:
        """Return True if visual change is a genuine shot cut (filtering flash returns and motion-blur smears)."""
        raw_cut = (c_i > self.cfg.change_threshold_c) or (d_i > 0.35)
        if not raw_cut:
            return False
        if self.strategy == "hybrid_legacy" or self.prev_entry is None:
            return True

        t_sec = entry.timestamp_seconds
        if (
            self.last_cut_ts_sec is not None
            and t_sec is not None
            and (t_sec - self.last_cut_ts_sec) < self.cfg.min_cut_interval_seconds
            and self.pre_cut_hist is not None
            and entry.grid_hist is not None
            and compute_hist_distance_from_hists(self.pre_cut_hist, entry.grid_hist) < 0.25
        ):
            return False

        s_max = max(self.prev_entry.sharpness, entry.sharpness)
        s_min = min(self.prev_entry.sharpness, entry.sharpness)
        if (
            not self.prev_entry.is_blank
            and not entry.is_blank
            and s_max >= 50.0
            and s_min < 0.15 * s_max
            and d_i < 0.18
            and c_i < 0.60
        ):
            return False
        return True

    def _process_frame_shot_medoid(self, frame: DecodedFrame) -> None:
        entry = self._extract_signal(frame, include_compact=True)
        if not entry.is_blank:
            self.last_content_entry = entry

        if self.current_epoch_id is not None and frame.epoch_id != self.current_epoch_id:
            if self._segment_entries:
                self._select_medoids_for_segment(
                    self._segment_entries,
                    self.scene_idx,
                    is_first_segment=(len(self._samples_by_ordinal) == 0),
                )
                self._segment_entries.clear()
        self.current_epoch_id = frame.epoch_id

        is_cut = False
        if self.prev_entry is not None and self.prev_entry.grid_hist is not None and entry.grid_hist is not None:
            d_i = (
                compute_luma_difference_from_luma(self.prev_entry.lowres_luma, entry.lowres_luma)
                if self.prev_entry.lowres_luma is not None and entry.lowres_luma is not None
                else 0.0
            )
            c_i = compute_hist_distance_from_hists(self.prev_entry.grid_hist, entry.grid_hist)
            is_cut = self._is_true_shot_cut(entry, d_i, c_i)

        if is_cut:
            is_first_seg = (self.scene_idx == 0 and len(self._samples_by_ordinal) == 0)
            self._select_medoids_for_segment(
                self._segment_entries,
                self.scene_idx,
                is_first_segment=is_first_seg,
                is_cut_end=True,
            )
            self._segment_entries.clear()
            self.pre_cut_hist = self.prev_entry.grid_hist if self.prev_entry is not None else None
            self.scene_idx += 1
            if entry.timestamp_seconds is not None:
                self.last_cut_ts_sec = entry.timestamp_seconds
            self.shot_cut_ordinals.add(entry.ordinal)
            self._upsert_sample(entry, {"cut_after"}, make_mandatory=True)
            self.last_selected_ts_sec = entry.timestamp_seconds
            self.last_selected_phash = entry.phash_int
            if entry.grid_hist is not None:
                self.last_selected_hist = entry.grid_hist
            self.recent_phashes.append(entry.phash_int)

        self._segment_entries.append(entry)
        self.prev_entry = entry

    def process_frame(self, frame: DecodedFrame) -> SampleRecord | None:
        """Process one presentation-order decoded frame according to configured strategy."""
        self.total_frames += 1

        if self.strategy == "uniform_matched":
            self._process_frame_uniform(frame)
            return None
        if self.strategy == "shot_medoid_submodular":
            self._process_frame_shot_medoid(frame)
            return None

        t_sec = frame.timestamp_seconds
        if self.current_epoch_id is not None and frame.epoch_id != self.current_epoch_id:
            self._flush_lookahead_buffer()
            self.d_history.clear()
            self.pending_burst_targets.clear()
            self.ring_buffer.clear()
            self.last_burst_trigger_ts = -999.0
            self.last_soft_commit_ts = -999.0
        self.current_epoch_id = frame.epoch_id

        entry = self._extract_signal(frame, include_compact=False)
        if not entry.is_blank:
            self.last_content_entry = entry

        # Maintain bounded-memory signal ring buffer for pre-burst selection
        self.ring_buffer.append(entry)
        if t_sec is not None:
            cutoff = t_sec - self.cfg.burst_pre_seconds
            while self.ring_buffer:
                first_ts = self.ring_buffer[0].timestamp_seconds
                if first_ts is not None and first_ts < cutoff:
                    self.ring_buffer.popleft()
                else:
                    break

        # Flush expired lookahead window before assessing new triggers on current frame
        eff_la = self._effective_lookahead_seconds()
        if (
            eff_la > 0.0
            and self._la_entries
            and t_sec is not None
            and self._la_start_ts is not None
            and (t_sec - self._la_start_ts) > eff_la
        ):
            self._flush_lookahead_buffer()

        reasons: set[str] = set()
        is_mandatory = False
        immediate_first = False

        # First frame anchor with blank-frame deferral up to 1.0s
        if entry.ordinal == 0:
            reasons.add("first")
            is_mandatory = True
            immediate_first = True
            if entry.is_blank:
                self.pending_first_valid_anchor = True
                self.pending_first_entry = entry

        if self.pending_first_valid_anchor and entry.ordinal > 0:
            if t_sec is not None and t_sec <= 1.0 and not entry.is_blank:
                if 0 in self._samples_by_ordinal:
                    del self._samples_by_ordinal[0]
                reasons.add("first")
                is_mandatory = True
                immediate_first = True
                self.pending_first_valid_anchor = False
                self.pending_first_entry = None
            elif t_sec is not None and t_sec > 1.0:
                self.pending_first_valid_anchor = False
                self.pending_first_entry = None

        # Max gap check (mandatory anchor)
        if t_sec is not None and self.last_selected_ts_sec is not None:
            gap = t_sec - self.last_selected_ts_sec
            if gap >= self.cfg.max_gap_seconds:
                static_gap = self.cfg.max_static_gap_seconds or self.cfg.max_gap_seconds
                min_dist = self._min_recent_phash_distance(entry.phash_int)
                if min_dist <= self.cfg.phash_min_distance and static_gap > self.cfg.max_gap_seconds:
                    if gap >= static_gap:
                        reasons.add("max_gap")
                        is_mandatory = True
                else:
                    reasons.add("max_gap")
                    is_mandatory = True

        # Change & cut detection
        is_cut = False
        triggered = False
        if (
            self.prev_entry is not None
            and self.prev_entry.lowres_luma is not None
            and entry.lowres_luma is not None
            and self.prev_entry.grid_hist is not None
            and entry.grid_hist is not None
        ):
            d_i = compute_luma_difference_from_luma(self.prev_entry.lowres_luma, entry.lowres_luma)
            c_i = compute_hist_distance_from_hists(self.prev_entry.grid_hist, entry.grid_hist)

            if t_sec is not None:
                self.d_history.append((t_sec, d_i))
                cutoff_d = t_sec - 2.0
                while self.d_history and self.d_history[0][0] < cutoff_d:
                    self.d_history.popleft()

            d_arr = np.fromiter((d for _, d in self.d_history), dtype=np.float64) if self.d_history else np.array([d_i])
            med = float(np.median(d_arr))
            mad = float(np.median(np.abs(d_arr - med)))
            r_i = d_i / (med + 3.0 * mad + 0.01)

            if self._is_true_shot_cut(entry, d_i, c_i):
                is_cut = True
                self._flush_lookahead_buffer()
                self.pre_cut_hist = self.prev_entry.grid_hist
                if t_sec is not None:
                    self.last_cut_ts_sec = t_sec
                self.scene_idx += 1
                reasons.add("cut_after")
                is_mandatory = True
                self.shot_cut_ordinals.add(entry.ordinal)
                self.shot_cut_ordinals.add(self.prev_entry.ordinal)
                self._upsert_sample(self.prev_entry, {"cut_before"}, make_mandatory=True)

            triggered = (r_i > self.cfg.change_threshold_r and d_i > self.cfg.change_threshold_d) or (
                c_i > self.cfg.change_threshold_c
            )
            if triggered and not is_cut:
                reasons.add("change_trigger")

            # Cumulative visual drift detection for slow camera pan / gradual dissolve (Strategies C & E)
            if (
                not is_cut
                and not triggered
                and self.strategy in ("sharpness_lookahead", "hybrid_adaptive")
                and self.last_selected_hist is not None
                and t_sec is not None
                and self.last_selected_ts_sec is not None
                and (t_sec - self.last_selected_ts_sec) >= 0.4
            ):
                drift_c = compute_hist_distance_from_hists(self.last_selected_hist, entry.grid_hist)
                if drift_c >= self.cfg.drift_threshold_c:
                    triggered = True
                    reasons.add("change_trigger")

            if t_sec is not None:
                should_burst = (
                    is_cut and (t_sec - self.last_burst_trigger_ts) >= max(0.5, self.cfg.burst_post_seconds)
                    if self.strategy == "hybrid_adaptive"
                    else (triggered or is_cut)
                )
                if should_burst:
                    self.last_burst_trigger_ts = t_sec
                    self._select_pre_burst(t_sec, entry.ordinal)
                    step = 1.0 / self.cfg.burst_fps
                    post_steps = int(round(self.cfg.burst_post_seconds * self.cfg.burst_fps))
                    self.pending_burst_targets.extend([t_sec + step * k for k in range(1, post_steps + 1)])

        # Pending post-burst targets
        if self.pending_burst_targets and t_sec is not None:
            remaining_targets = []
            matched_burst = False
            tolerance = 0.5 / self.cfg.burst_fps
            for target in self.pending_burst_targets:
                if abs(t_sec - target) <= tolerance:
                    matched_burst = True
                elif target > t_sec:
                    remaining_targets.append(target)
            self.pending_burst_targets = remaining_targets
            if matched_burst:
                reasons.add("burst")

        record: SampleRecord | None = None
        if immediate_first or is_cut or eff_la <= 0.0:
            if is_cut or immediate_first:
                self._flush_lookahead_buffer()
            if reasons:
                record = self._commit_selected(entry, reasons, is_mandatory)
        else:
            # Causal lookahead path for non-cut, non-first frames (Strategies C & E)
            if self._la_entries:
                self._la_entries.append(entry)
                self._la_reasons |= reasons
                self._la_mandatory = self._la_mandatory or is_mandatory
            elif reasons:
                self._la_entries = [entry]
                self._la_reasons = set(reasons)
                self._la_mandatory = is_mandatory
                self._la_start_ts = t_sec
                if t_sec is None:
                    record = self._flush_lookahead_buffer()

        self.prev_entry = entry
        return record

    def finalize(self) -> SampleManifest:
        """Finalize sampling, flushing any open lookahead/segment buffers and preserving end anchors."""
        if self.strategy == "uniform_matched":
            if self._uniform_entries:
                ts_vals = [e.timestamp_seconds or 0.0 for e in self._uniform_entries]
                duration = max(0.0, ts_vals[-1] - ts_vals[0])
                fps = max(self.cfg.soft_target_fps, 1.0 / self.cfg.max_gap_seconds)
                n_steps = int(math.floor(duration * fps)) + 1
                step = 1.0 / fps
                grid_times = [ts_vals[0] + i * step for i in range(n_steps)]
                if duration > 0 and (len(grid_times) == 0 or (ts_vals[-1] - grid_times[-1]) > 0.2 * step):
                    grid_times.append(ts_vals[-1])
                ts_arr = np.array(ts_vals, dtype=np.float64)
                seen_idx: set[int] = set()
                for g_idx, gt in enumerate(grid_times):
                    idx = int(np.argmin(np.abs(ts_arr - gt)))
                    if idx in seen_idx:
                        continue
                    seen_idx.add(idx)
                    ent = self._uniform_entries[idx]
                    r_set = {"uniform_grid"}
                    is_mand = g_idx == 0 or idx == len(self._uniform_entries) - 1
                    if idx == 0:
                        r_set.add("first")
                    if idx == len(self._uniform_entries) - 1:
                        r_set.add("last")
                    self._upsert_sample(ent, r_set, make_mandatory=is_mand, scene_idx_override=0)
        elif self.strategy == "shot_medoid_submodular":
            if self._segment_entries:
                self._select_medoids_for_segment(
                    self._segment_entries,
                    self.scene_idx,
                    is_first_segment=(self.scene_idx == 0 and len(self._samples_by_ordinal) == 0),
                    is_last_segment=True,
                )
                self._segment_entries.clear()
        else:
            self._flush_lookahead_buffer()
            if self.pending_first_valid_anchor and self.pending_first_entry is not None:
                self._upsert_sample(self.pending_first_entry, {"first"}, make_mandatory=True)
                self.pending_first_valid_anchor = False
                self.pending_first_entry = None

            if self.prev_entry is not None:
                if (
                    self.prev_entry.is_blank
                    and self.last_content_entry is not None
                    and self.last_content_entry.ordinal not in self._samples_by_ordinal
                    and self.last_selected_ts_sec is not None
                    and self.last_content_entry.timestamp_seconds is not None
                    and (self.last_content_entry.timestamp_seconds - self.last_selected_ts_sec) >= 0.4
                    and self._min_recent_phash_distance(self.last_content_entry.phash_int) > self.cfg.phash_min_distance
                ):
                    self._upsert_sample(self.last_content_entry, {"last_content"}, make_mandatory=False)
                self._upsert_sample(self.prev_entry, {"last"}, make_mandatory=True)

        sorted_samples = tuple(sorted(self._samples_by_ordinal.values(), key=lambda s: s.original_decoded_frame_index))
        mandatory_count = sum(1 for s in sorted_samples if s.is_mandatory)
        soft_count = len(sorted_samples) - mandatory_count
        gap_violations = 0
        max_allowed_gap = (self.cfg.max_static_gap_seconds or self.cfg.max_gap_seconds) + 0.2

        for a, b in zip(sorted_samples, sorted_samples[1:]):
            if a.timestamp_seconds is not None and b.timestamp_seconds is not None:
                gap = b.timestamp_seconds - a.timestamp_seconds
                if gap > max_allowed_gap:
                    gap_violations += 1

        return SampleManifest(
            video_id=self.video_id,
            timeline_revision=self.timeline_revision,
            sampler_revision=self.sampler_revision,
            samples=sorted_samples,
            total_frames_examined=self.total_frames,
            mandatory_anchor_count=mandatory_count,
            soft_sample_count=soft_count,
            max_gap_violations=gap_violations,
            overflow_count=self.overflow_count,
            shot_cut_ordinals=tuple(sorted(self.shot_cut_ordinals)),
        )


def sample_frames_hybrid(
    frames: Iterator[DecodedFrame] | Sequence[DecodedFrame],
    config: SamplingConfig,
    video_id: str,
    timeline_revision: str,
    sampler_revision: str | None = None,
) -> SampleManifest:
    """Convenience function to run configured adaptive sampling on a sequence of decoded frames."""
    sampler = HybridAdaptiveSampler(
        config=config,
        video_id=video_id,
        timeline_revision=timeline_revision,
        sampler_revision=sampler_revision,
    )
    for f in frames:
        sampler.process_frame(f)
    return sampler.finalize()
