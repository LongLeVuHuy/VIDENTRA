"""Configuration without machine-specific paths."""
from dataclasses import dataclass
from pathlib import Path
import os

from .ingestion import (
    IngestionConfig,
    StreamSelectionConfig,
    DecodeConfig,
    TimelineConfig,
    SamplingConfig,
    JobPolicyConfig,
    VALID_SAMPLING_STRATEGIES,
)


@dataclass(frozen=True)
class Paths:
    data: Path
    models: Path
    indexes: Path
    results: Path

    @classmethod
    def from_environment(cls) -> "Paths":
        names = ("DATA", "MODEL", "INDEX", "RESULTS")
        missing = [f"VIDENTRA_{n}_ROOT" for n in names if not os.environ.get(f"VIDENTRA_{n}_ROOT")]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        return cls(*(Path(os.environ[f"VIDENTRA_{n}_ROOT"]).resolve() for n in names))


__all__ = [
    "Paths",
    "IngestionConfig",
    "StreamSelectionConfig",
    "DecodeConfig",
    "TimelineConfig",
    "SamplingConfig",
    "JobPolicyConfig",
    "VALID_SAMPLING_STRATEGIES",
]
