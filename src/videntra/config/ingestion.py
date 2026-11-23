"""Typed ingestion and sampling configuration contracts."""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any


def _check_no_unknown_fields(cls_name: str, data: dict[str, Any], allowed_fields: set[str]) -> None:
    unknown = set(data.keys()) - allowed_fields
    if unknown:
        raise ValueError(f"Unknown fields for {cls_name}: {sorted(unknown)}")


@dataclass(frozen=True)
class StreamSelectionConfig:
    video_stream_index: int | None = None
    audio_stream_index: int | None = None
    allow_missing_audio: bool = True

    def validate(self) -> None:
        if self.video_stream_index is not None and self.video_stream_index < 0:
            raise ValueError("video_stream_index must be non-negative")
        if self.audio_stream_index is not None and self.audio_stream_index < 0:
            raise ValueError("audio_stream_index must be non-negative")


@dataclass(frozen=True)
class DecodeConfig:
    thread_count: int = 0
    max_frames: int | None = None
    error_policy: str = "strict"

    def validate(self) -> None:
        if self.thread_count < 0:
            raise ValueError("thread_count must be non-negative")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be strictly positive")
        if self.error_policy not in ("strict", "quarantine", "ignore"):
            raise ValueError(f"Invalid error_policy: {self.error_policy}")


@dataclass(frozen=True)
class TimelineConfig:
    row_group_size: int = 1000
    time_unit: str = "us"

    def validate(self) -> None:
        if self.row_group_size <= 0:
            raise ValueError("row_group_size must be strictly positive")
        if self.time_unit != "us":
            raise ValueError(f"Unsupported time_unit: {self.time_unit}")


VALID_SAMPLING_STRATEGIES = (
    "uniform_matched",
    "hybrid_legacy",
    "sharpness_lookahead",
    "shot_medoid_submodular",
    "hybrid_adaptive",
)

SAMPLING_STRATEGY_ALIASES: dict[str, str] = {
    "hybrid_v2_adaptive": "hybrid_adaptive",
    "hybrid_v1_legacy": "hybrid_legacy",
}


def normalize_sampling_strategy(strategy: str) -> str:
    """Resolve legacy versioned strategy names to canonical strategy identifiers."""
    return SAMPLING_STRATEGY_ALIASES.get(str(strategy), str(strategy))


@dataclass(frozen=True)
class SamplingConfig:
    lowres_max_dim: int = 256
    max_gap_seconds: float = 2.0
    burst_pre_seconds: float = 0.5
    burst_post_seconds: float = 1.0
    burst_fps: float = 4.0
    soft_target_fps: float = 1.0
    soft_window_seconds: float = 60.0
    ocr_probe_fps: float = 1.0
    change_threshold_r: float = 3.0
    change_threshold_d: float = 0.04
    change_threshold_c: float = 0.35
    phash_min_distance: int = 4
    optical_flow_enabled: bool = False
    max_static_gap_seconds: float | None = None
    strategy: str = "hybrid_adaptive"
    lookahead_window_seconds: float = 0.15
    drift_threshold_c: float = 0.38
    min_cut_interval_seconds: float = 0.20
    min_luma_std: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", normalize_sampling_strategy(self.strategy))

    def validate(self) -> None:
        if self.lowres_max_dim < 32:
            raise ValueError("lowres_max_dim must be at least 32")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be strictly positive")
        if self.max_static_gap_seconds is not None and self.max_static_gap_seconds <= 0:
            raise ValueError("max_static_gap_seconds must be strictly positive")
        if self.burst_pre_seconds < 0:
            raise ValueError("burst_pre_seconds must be non-negative")
        if self.burst_post_seconds < 0:
            raise ValueError("burst_post_seconds must be non-negative")
        if self.burst_fps <= 0:
            raise ValueError("burst_fps must be strictly positive")
        if self.soft_target_fps <= 0:
            raise ValueError("soft_target_fps must be strictly positive")
        if self.soft_window_seconds <= 0:
            raise ValueError("soft_window_seconds must be strictly positive")
        if self.ocr_probe_fps <= 0:
            raise ValueError("ocr_probe_fps must be strictly positive")
        if self.phash_min_distance < 0 or self.phash_min_distance > 64:
            raise ValueError("phash_min_distance must be between 0 and 64")
        if self.strategy not in VALID_SAMPLING_STRATEGIES:
            raise ValueError(f"Invalid sampling strategy '{self.strategy}'; expected one of {VALID_SAMPLING_STRATEGIES}")
        if self.lookahead_window_seconds < 0:
            raise ValueError("lookahead_window_seconds must be non-negative")
        if not (0.0 < self.drift_threshold_c <= 1.0):
            raise ValueError("drift_threshold_c must be in (0, 1]")
        if self.min_cut_interval_seconds < 0:
            raise ValueError("min_cut_interval_seconds must be non-negative")
        if self.min_luma_std < 0:
            raise ValueError("min_luma_std must be non-negative")

    @classmethod
    def for_duration(cls, duration_seconds: float) -> "SamplingConfig":
        """Generate optimized sampling configuration based on video duration."""
        if duration_seconds <= 15.0:
            return cls(max_gap_seconds=1.0, soft_target_fps=1.5, max_static_gap_seconds=3.0)
        elif duration_seconds <= 60.0:
            return cls(max_gap_seconds=1.5, soft_target_fps=1.0, max_static_gap_seconds=4.0)
        else:
            return cls(max_gap_seconds=2.0, soft_target_fps=1.0, max_static_gap_seconds=4.5)


@dataclass(frozen=True)
class JobPolicyConfig:
    max_retries: int = 2
    lease_timeout_seconds: int = 300
    worker_cpu_count: int = 1

    def validate(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be strictly positive")
        if self.worker_cpu_count < 1:
            raise ValueError("worker_cpu_count must be at least 1")


@dataclass(frozen=True)
class IngestionConfig:
    schema_version: str = "1.0"
    recipe_id: str = "m1_default"
    stream_selection: StreamSelectionConfig = field(default_factory=StreamSelectionConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    jobs: JobPolicyConfig = field(default_factory=JobPolicyConfig)

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if not self.recipe_id.strip():
            raise ValueError("recipe_id cannot be empty")
        self.stream_selection.validate()
        self.decode.validate()
        self.timeline.validate()
        self.sampling.validate()
        self.jobs.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionConfig":
        allowed = {"schema_version", "recipe_id", "stream_selection", "decode", "timeline", "sampling", "jobs"}
        _check_no_unknown_fields("IngestionConfig", data, allowed)

        ss_data = data.get("stream_selection", {})
        _check_no_unknown_fields("StreamSelectionConfig", ss_data, {"video_stream_index", "audio_stream_index", "allow_missing_audio"})

        dec_data = data.get("decode", {})
        _check_no_unknown_fields("DecodeConfig", dec_data, {"thread_count", "max_frames", "error_policy"})

        tl_data = data.get("timeline", {})
        _check_no_unknown_fields("TimelineConfig", tl_data, {"row_group_size", "time_unit"})

        samp_data = data.get("sampling", {})
        _check_no_unknown_fields("SamplingConfig", samp_data, {
            "lowres_max_dim", "max_gap_seconds", "burst_pre_seconds", "burst_post_seconds",
            "burst_fps", "soft_target_fps", "soft_window_seconds", "ocr_probe_fps",
            "change_threshold_r", "change_threshold_d", "change_threshold_c",
            "phash_min_distance", "optical_flow_enabled", "max_static_gap_seconds",
            "strategy", "lookahead_window_seconds", "drift_threshold_c",
            "min_cut_interval_seconds", "min_luma_std",
        })

        jobs_data = data.get("jobs", {})
        _check_no_unknown_fields("JobPolicyConfig", jobs_data, {"max_retries", "lease_timeout_seconds", "worker_cpu_count"})

        instance = cls(
            schema_version=data.get("schema_version", "1.0"),
            recipe_id=data.get("recipe_id", "m1_default"),
            stream_selection=StreamSelectionConfig(**ss_data),
            decode=DecodeConfig(**dec_data),
            timeline=TimelineConfig(**tl_data),
            sampling=SamplingConfig(**samp_data),
            jobs=JobPolicyConfig(**jobs_data),
        )
        instance.validate()
        return instance

    @classmethod
    def from_file(cls, path: Path | str) -> "IngestionConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_file(self, path: Path | str) -> None:
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

