"""Configuration without machine-specific paths."""
from dataclasses import dataclass
from pathlib import Path
import os

from .device import (
    resolve_device,
    preferred_cuda_dtype,
    get_device_info,
    collect_environment_report,
)
from .ingestion import (
    IngestionConfig,
    StreamSelectionConfig,
    DecodeConfig,
    TimelineConfig,
    SamplingConfig,
    JobPolicyConfig,
    VALID_SAMPLING_STRATEGIES,
)
from .paths import (
    CACHE_MIGRATION_MAP,
    CACHE_SUBDIRS,
    DEFAULT_AUGMENTED_EMBEDDINGS_PATH,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_E5_CAPTION_FAISS_PATH,
    DEFAULT_E5_CAPTION_MANIFEST_PATH,
    DEFAULT_E5_FAISS_PATH,
    DEFAULT_E5_MANIFEST_PATH,
    DEFAULT_E5_QUERY_CACHE_PATH,
    DEFAULT_HUBNESS_RADII_PATH,
    DEFAULT_KEYFRAME_FAISS_PATH,
    DEFAULT_METADATA_DB_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_QUERY_PLANS_CACHE_PATH,
    DEFAULT_SIGLIP2_QUERY_CACHE_PATH,
    DEFAULT_VIDEOPRISM_CLIP_FAISS_PATH,
    DEFAULT_VIDEOPRISM_CLIP_MANIFEST_PATH,
    DEFAULT_VIDEOPRISM_QUERY_CACHE_PATH,
    DEFAULT_VIDEOPRISM_SUBCLIPS_MANIFEST_PATH,
    DEFAULT_VIDEOPRISM_SUBCLIPS_NPY_PATH,
    PROCESSED_MIGRATION_MAP,
    PROCESSED_SUBDIRS,
    resolve_artifact_path,
    resolve_env_db_path,
    resolve_env_faiss_path,
    resolve_keyframe_faiss_path,
    resolve_metadata_db_path,
    resolve_processed_root,
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
    "resolve_device",
    "preferred_cuda_dtype",
    "get_device_info",
    "collect_environment_report",
    "IngestionConfig",
    "StreamSelectionConfig",
    "DecodeConfig",
    "TimelineConfig",
    "SamplingConfig",
    "JobPolicyConfig",
    "VALID_SAMPLING_STRATEGIES",
    "CACHE_MIGRATION_MAP",
    "CACHE_SUBDIRS",
    "DEFAULT_AUGMENTED_EMBEDDINGS_PATH",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_E5_CAPTION_FAISS_PATH",
    "DEFAULT_E5_CAPTION_MANIFEST_PATH",
    "DEFAULT_E5_FAISS_PATH",
    "DEFAULT_E5_MANIFEST_PATH",
    "DEFAULT_E5_QUERY_CACHE_PATH",
    "DEFAULT_HUBNESS_RADII_PATH",
    "DEFAULT_KEYFRAME_FAISS_PATH",
    "DEFAULT_METADATA_DB_PATH",
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_QUERY_PLANS_CACHE_PATH",
    "DEFAULT_SIGLIP2_QUERY_CACHE_PATH",
    "DEFAULT_VIDEOPRISM_CLIP_FAISS_PATH",
    "DEFAULT_VIDEOPRISM_CLIP_MANIFEST_PATH",
    "DEFAULT_VIDEOPRISM_QUERY_CACHE_PATH",
    "DEFAULT_VIDEOPRISM_SUBCLIPS_MANIFEST_PATH",
    "DEFAULT_VIDEOPRISM_SUBCLIPS_NPY_PATH",
    "PROCESSED_MIGRATION_MAP",
    "PROCESSED_SUBDIRS",
    "resolve_artifact_path",
    "resolve_env_db_path",
    "resolve_env_faiss_path",
    "resolve_keyframe_faiss_path",
    "resolve_metadata_db_path",
    "resolve_processed_root",
]
