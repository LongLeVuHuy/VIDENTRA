"""Hybrid adaptive frame sampling with mandatory anchors, change triggers, bursts, and dedup."""
from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterator, Sequence
import uuid
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from videntra.config.ingestion import SamplingConfig
from videntra.ingestion.decode import DecodedFrame


# 32x32 DCT transform matrix for pHash
_N = 32
_I = np.arange(_N)
_J = np.arange(_N)
_DCT_MATRIX = np.cos(np.pi / _N * (_I[:, None] + 0.5) * _J[None, :])


def compute_phash_64(image_rgb: np.ndarray) -> int:
    """Compute 64-bit DCT perceptual hash from RGB uint8 image."""
    h, w = image_rgb.shape[:2]
    # Resize to 32x32 grayscale
    y_coords = (np.arange(32) * (h / 32.0)).astype(int)
    x_coords = (np.arange(32) * (w / 32.0)).astype(int)
    sample = image_rgb[y_coords[:, None], x_coords[None, :]]
    # Grayscale conversion
    gray = 0.299 * sample[:, :, 0] + 0.587 * sample[:, :, 1] + 0.114 * sample[:, :, 2]
    # 2D DCT: D = C^T * gray * C
    dct_2d = _DCT_MATRIX.T @ gray @ _DCT_MATRIX
    # Top-left 8x8 excluding DC component (0, 0)
    low_freq = dct_2d[:8, :8]
    med = np.median(low_freq[1:, 1:])
    # 64-bit boolean mask
    bits = (low_freq > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def phash_hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def resize_lowres(image_rgb: np.ndarray, max_dim: int = 256) -> np.ndarray:
    """Resize image so its maximum dimension does not exceed max_dim."""
    h, w = image_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return image_rgb
    scale = max_dim / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    y_coords = (np.arange(new_h) / scale).astype(int)
    x_coords = (np.arange(new_w) / scale).astype(int)
    y_coords = np.clip(y_coords, 0, h - 1)
    x_coords = np.clip(x_coords, 0, w - 1)
    return image_rgb[y_coords[:, None], x_coords[None, :]]


def compute_luma_difference(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """Mean absolute luma difference normalized in [0, 1]."""
    y1 = 0.299 * img1_rgb[:, :, 0] + 0.587 * img1_rgb[:, :, 1] + 0.114 * img1_rgb[:, :, 2]
    y2 = 0.299 * img2_rgb[:, :, 0] + 0.587 * img2_rgb[:, :, 1] + 0.114 * img2_rgb[:, :, 2]
    diff = np.abs(y1 - y2) / 255.0
    return float(np.mean(diff))


def compute_grid_hist_512(img_rgb: np.ndarray) -> np.ndarray:
    """Compute normalized 512-bin color histograms across a 4x4 spatial grid."""
    h, w = img_rgb.shape[:2]
    cell_h = max(1, h // 4)
    cell_w = max(1, w // 4)
    r_q = (img_rgb[:, :, 0] >> 5).astype(np.int32)
    g_q = (img_rgb[:, :, 1] >> 5).astype(np.int32)
    b_q = (img_rgb[:, :, 2] >> 5).astype(np.int32)
    idx_img = (r_q << 6) | (g_q << 3) | b_q

    grid_hists = []
    for gy in range(4):
        y0 = gy * cell_h
        y1 = h if gy == 3 else (gy + 1) * cell_h
        for gx in range(4):
            x0 = gx * cell_w
            x1 = w if gx == 3 else (gx + 1) * cell_w
            cell_idx = idx_img[y0:y1, x0:x1].ravel()
            hist = np.bincount(cell_idx, minlength=512).astype(np.float32)
            s = hist.sum()
            if s > 0:
                hist /= s
            grid_hists.append(hist)
    return np.stack(grid_hists, axis=0)


def compute_hist_distance(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """Normalized 4x4 spatial grid color histogram intersection distance in [0, 1]."""
    h1 = compute_grid_hist_512(img1_rgb)
    h2 = compute_grid_hist_512(img2_rgb)
    intersections = np.sum(np.minimum(h1, h2), axis=-1)
    mean_inter = float(np.mean(intersections))
    return float(np.clip(1.0 - mean_inter, 0.0, 1.0))


def compute_sharpness_score(img_rgb: np.ndarray) -> float:
    """Variance of 2D discrete Laplacian operator as sharpness metric."""
    y = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    y_pad = np.pad(y, ((1, 1), (1, 1)), mode="edge")
    lap = (
        4.0 * y_pad[1:-1, 1:-1]
        - y_pad[:-2, 1:-1] - y_pad[2:, 1:-1]
        - y_pad[1:-1, :-2] - y_pad[1:-1, 2:]
    )
    return float(np.var(lap))


def is_blank_or_fade_frame(img_rgb: np.ndarray, min_luma_std: float = 8.0) -> bool:
    """Returns True if the frame has low luma standard deviation (e.g. solid black or white)."""
    y = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    return float(np.std(y)) < min_luma_std


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
        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, str(p), compression="zstd")


def make_sample_id(video_id: str, timeline_revision: str, ordinal: int, sampler_rev: str) -> str:
    key = f"{video_id}:{timeline_revision}:{ordinal}:{sampler_rev}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


class HybridAdaptiveSampler:
    """Stateful hybrid sampler implementing mandatory anchors, change triggers, bursts, and pHash dedup."""

    def __init__(
        self,
        config: SamplingConfig,
        video_id: str,
        timeline_revision: str,
        sampler_revision: str = "videntra_hybrid_v1",
    ):
        self.cfg = config
        self.video_id = video_id
        self.timeline_revision = timeline_revision
        self.sampler_revision = sampler_rev = sampler_revision

        self.last_selected_ts_sec: float | None = None
        self.last_selected_phash: int | None = None
        self.prev_frame: DecodedFrame | None = None
        self.prev_lowres: np.ndarray | None = None

        self.scene_idx = 0
        self._samples_by_ordinal: dict[int, SampleRecord] = {}
        self.recent_phashes: deque[int] = deque(maxlen=5)
        self.pending_first_valid_anchor: bool = False
        self.pending_first_frame: DecodedFrame | None = None
        self.pending_first_lowres: np.ndarray | None = None

        self.d_history: deque[tuple[float, float]] = deque()  # (ts, d_i)
        self.ring_buffer: deque[DecodedFrame] = deque()  # stores last burst_pre_seconds of frames
        self.pending_burst_targets: list[float] = []
        self.soft_sample_timestamps: deque[float] = deque()  # for 60s soft quota
        self.overflow_count = 0
        self.total_frames = 0
        self.shot_cut_ordinals: set[int] = set()
        self.current_epoch_id: str | None = None

    @property
    def samples(self) -> list[SampleRecord]:
        return sorted(self._samples_by_ordinal.values(), key=lambda s: s.original_decoded_frame_index)

    def _update_soft_quota(self, now_sec: float) -> bool:
        """Enforce rolling soft target <= 1.0 fps over 60s."""
        window_start = now_sec - self.cfg.soft_window_seconds
        while self.soft_sample_timestamps and self.soft_sample_timestamps[0] < window_start:
            self.soft_sample_timestamps.popleft()
        max_allowed = self.cfg.soft_target_fps * self.cfg.soft_window_seconds
        return len(self.soft_sample_timestamps) < max_allowed

    def _find_sample_by_ordinal(self, ordinal: int) -> int | None:
        return ordinal if ordinal in self._samples_by_ordinal else None

    def _update_or_add_sample_reasons(
        self,
        frame: DecodedFrame,
        additional_reasons: set[str],
        make_mandatory: bool = False,
        lowres_img: np.ndarray | None = None,
    ) -> None:
        existing = self._samples_by_ordinal.get(frame.ordinal)
        if existing is not None:
            merged_reasons = tuple(sorted(set(existing.reasons) | additional_reasons))
            is_mand = existing.is_mandatory or make_mandatory
            self._samples_by_ordinal[frame.ordinal] = SampleRecord(
                sampled_frame_id=existing.sampled_frame_id,
                video_id=existing.video_id,
                timeline_revision=existing.timeline_revision,
                sampler_revision=existing.sampler_revision,
                original_decoded_frame_index=existing.original_decoded_frame_index,
                timestamp_us=existing.timestamp_us,
                timestamp_seconds=existing.timestamp_seconds,
                reasons=merged_reasons,
                is_mandatory=is_mand,
                phash=existing.phash,
                scene_id=existing.scene_id,
                epoch_id=existing.epoch_id,
            )
        else:
            if lowres_img is None:
                lowres_img = resize_lowres(frame.pixels, self.cfg.lowres_max_dim) if frame.pixels is not None else None
            p_int = compute_phash_64(lowres_img) if lowres_img is not None else 0
            s_id = make_sample_id(self.video_id, self.timeline_revision, frame.ordinal, self.sampler_revision)
            rec = SampleRecord(
                sampled_frame_id=s_id,
                video_id=self.video_id,
                timeline_revision=self.timeline_revision,
                sampler_revision=self.sampler_revision,
                original_decoded_frame_index=frame.ordinal,
                timestamp_us=frame.timestamp_us,
                timestamp_seconds=frame.timestamp_seconds,
                reasons=tuple(sorted(additional_reasons)),
                is_mandatory=make_mandatory,
                phash=f"{p_int:016x}",
                scene_id=f"scene_{self.scene_idx:04d}",
                epoch_id=frame.epoch_id,
            )
            self._samples_by_ordinal[frame.ordinal] = rec

    def process_frame(self, frame: DecodedFrame) -> SampleRecord | None:
        """Process one presentation-order decoded frame."""
        self.total_frames += 1
        t_sec = frame.timestamp_seconds

        # Epoch change detection & state reset
        if self.current_epoch_id is not None and frame.epoch_id != self.current_epoch_id:
            self.d_history.clear()
            self.pending_burst_targets.clear()
            self.ring_buffer.clear()
        self.current_epoch_id = frame.epoch_id

        # Maintain ring buffer for pre-burst window
        self.ring_buffer.append(frame)
        if t_sec is not None:
            cutoff = t_sec - self.cfg.burst_pre_seconds
            while self.ring_buffer:
                first_ts = self.ring_buffer[0].timestamp_seconds
                if first_ts is not None and first_ts < cutoff:
                    self.ring_buffer.popleft()
                else:
                    break

        lowres = resize_lowres(frame.pixels, self.cfg.lowres_max_dim) if frame.pixels is not None else None
        curr_phash = compute_phash_64(lowres) if lowres is not None else 0

        reasons = set()
        is_mandatory = False

        # First frame is mandatory anchor, but mark if blank to replace with first content frame if one appears within 1.0s
        if frame.ordinal == 0:
            reasons.add("first")
            is_mandatory = True
            if lowres is not None and is_blank_or_fade_frame(lowres):
                self.pending_first_valid_anchor = True
                self.pending_first_frame = frame
                self.pending_first_lowres = lowres

        # Check pending first valid anchor
        if self.pending_first_valid_anchor and frame.ordinal > 0:
            if t_sec is not None and t_sec <= 1.0 and lowres is not None and not is_blank_or_fade_frame(lowres):
                # Replace initial blank frame with this first content frame
                if 0 in self._samples_by_ordinal:
                    del self._samples_by_ordinal[0]
                reasons.add("first")
                is_mandatory = True
                self.pending_first_valid_anchor = False
            elif t_sec is not None and t_sec > 1.0:
                self.pending_first_valid_anchor = False

        # Max gap check (mandatory anchor)
        if t_sec is not None and self.last_selected_ts_sec is not None:
            gap = t_sec - self.last_selected_ts_sec
            if gap >= self.cfg.max_gap_seconds:
                static_gap = self.cfg.max_static_gap_seconds or self.cfg.max_gap_seconds
                min_dist = min(
                    (phash_hamming_distance(h, curr_phash) for h in self.recent_phashes),
                    default=999,
                ) if self.recent_phashes else (
                    phash_hamming_distance(self.last_selected_phash, curr_phash) if self.last_selected_phash is not None else 999
                )
                if min_dist <= self.cfg.phash_min_distance and static_gap > self.cfg.max_gap_seconds:
                    if gap >= static_gap:
                        reasons.add("max_gap")
                        is_mandatory = True
                else:
                    reasons.add("max_gap")
                    is_mandatory = True

        # Change & cut detection
        d_i = 0.0
        c_i = 0.0
        r_i = 0.0
        is_cut = False

        if self.prev_lowres is not None and lowres is not None:
            d_i = compute_luma_difference(self.prev_lowres, lowres)
            c_i = compute_hist_distance(self.prev_lowres, lowres)

            # Rolling 2s history for d_i
            if t_sec is not None:
                self.d_history.append((t_sec, d_i))
                cutoff_d = t_sec - 2.0
                while self.d_history and self.d_history[0][0] < cutoff_d:
                    self.d_history.popleft()

            d_arr = np.array([d for _, d in self.d_history], dtype=float) if self.d_history else np.array([d_i])
            med = float(np.median(d_arr))
            mad = float(np.median(np.abs(d_arr - med)))
            r_i = d_i / (med + 3.0 * mad + 0.01)

            # Cut detection: c_i > 0.35 or significant change
            if c_i > self.cfg.change_threshold_c or d_i > 0.35:
                is_cut = True
                self.scene_idx += 1
                reasons.add("cut_after")
                is_mandatory = True
                self.shot_cut_ordinals.add(frame.ordinal)

                # Mark previous frame as cut_before
                if self.prev_frame is not None:
                    self.shot_cut_ordinals.add(self.prev_frame.ordinal)
                    self._update_or_add_sample_reasons(
                        self.prev_frame,
                        {"cut_before"},
                        make_mandatory=True,
                        lowres_img=self.prev_lowres,
                    )

            # Trigger condition: (r_i > 3 and d_i > 0.04) or c_i > 0.35
            triggered = (r_i > self.cfg.change_threshold_r and d_i > self.cfg.change_threshold_d) or (c_i > self.cfg.change_threshold_c)
            if triggered and not is_cut:
                reasons.add("change_trigger")

            if (triggered or is_cut) and t_sec is not None:
                step = 1.0 / self.cfg.burst_fps

                # 1. Select pre-burst frames from ring buffer in [t_sec - burst_pre_seconds, t_sec)
                pre_steps = int(round(self.cfg.burst_pre_seconds * self.cfg.burst_fps))
                pre_targets = [t_sec - step * k for k in range(1, pre_steps + 1) if (t_sec - step * k) >= 0]
                for target_t in pre_targets:
                    best_f = None
                    best_diff = float("inf")
                    best_sharp = -1.0
                    for bf in self.ring_buffer:
                        if bf.timestamp_seconds is not None:
                            diff = abs(bf.timestamp_seconds - target_t)
                            if diff <= (0.5 / self.cfg.burst_fps):
                                sharp = compute_sharpness_score(bf.pixels) if bf.pixels is not None else 0.0
                                if diff < best_diff or (abs(diff - best_diff) < 1e-4 and sharp > best_sharp):
                                    best_diff = diff
                                    best_sharp = sharp
                                    best_f = bf
                    if best_f is not None and best_f.ordinal != frame.ordinal:
                        self._update_or_add_sample_reasons(best_f, {"burst"}, make_mandatory=False)

                # 2. Schedule post-burst targets in (t_sec, t_sec + burst_post_seconds]
                post_steps = int(round(self.cfg.burst_post_seconds * self.cfg.burst_fps))
                post_targets = [t_sec + step * k for k in range(1, post_steps + 1)]
                self.pending_burst_targets.extend(post_targets)

        # Check pending burst targets
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

        # Decision logic
        record = None
        if reasons:
            accept = False
            if is_mandatory:
                accept = True
            else:
                # Soft candidate: check pHash dedup
                phash_dist = phash_hamming_distance(self.last_selected_phash, curr_phash) if self.last_selected_phash is not None else 999
                if phash_dist > self.cfg.phash_min_distance:
                    # Check soft quota
                    if t_sec is None or self._update_soft_quota(t_sec):
                        accept = True
                        if t_sec is not None:
                            self.soft_sample_timestamps.append(t_sec)
                    else:
                        self.overflow_count += 1
                else:
                    # Dropped by pHash dedup
                    pass

            if accept:
                s_id = make_sample_id(self.video_id, self.timeline_revision, frame.ordinal, self.sampler_revision)
                record = SampleRecord(
                    sampled_frame_id=s_id,
                    video_id=self.video_id,
                    timeline_revision=self.timeline_revision,
                    sampler_revision=self.sampler_revision,
                    original_decoded_frame_index=frame.ordinal,
                    timestamp_us=frame.timestamp_us,
                    timestamp_seconds=t_sec,
                    reasons=tuple(sorted(reasons)),
                    is_mandatory=is_mandatory,
                    phash=f"{curr_phash:016x}",
                    scene_id=f"scene_{self.scene_idx:04d}",
                    epoch_id=frame.epoch_id,
                )
                self._samples_by_ordinal[frame.ordinal] = record
                if t_sec is not None:
                    self.last_selected_ts_sec = t_sec
                self.last_selected_phash = curr_phash
                self.recent_phashes.append(curr_phash)

        self.prev_frame = frame
        self.prev_lowres = lowres
        return record

    def finalize(self) -> SampleManifest:
        """Finalize sampling, ensuring last frame is preserved as mandatory anchor."""
        if self.pending_first_valid_anchor and self.pending_first_frame is not None:
            self._update_or_add_sample_reasons(
                self.pending_first_frame,
                {"first"},
                make_mandatory=True,
                lowres_img=self.pending_first_lowres,
            )
            self.pending_first_valid_anchor = False

        if self.prev_frame is not None:
            self._update_or_add_sample_reasons(
                self.prev_frame,
                {"last"},
                make_mandatory=True,
                lowres_img=self.prev_lowres,
            )

        sorted_samples = tuple(sorted(self._samples_by_ordinal.values(), key=lambda s: s.original_decoded_frame_index))

        # Count statistics and check max-gap violations
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
    sampler_revision: str = "videntra_hybrid_v1",
) -> SampleManifest:
    """Convenience function to run hybrid adaptive sampling on a sequence of decoded frames."""
    sampler = HybridAdaptiveSampler(
        config=config,
        video_id=video_id,
        timeline_revision=timeline_revision,
        sampler_revision=sampler_revision,
    )
    for f in frames:
        sampler.process_frame(f)
    return sampler.finalize()
