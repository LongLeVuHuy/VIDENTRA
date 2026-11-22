"""Single Source of Truth (SSOT) for self-documenting data/processed and data/cache paths."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_ROOT / "processed"
DEFAULT_CACHE_DIR = DEFAULT_DATA_ROOT / "cache"

PROCESSED_SUBDIRS: tuple[str, ...] = (
    "catalog",
    "visual_keyframes",
    "visual_clips",
    "text_evidence",
)
CACHE_SUBDIRS: tuple[str, ...] = (
    "query_embeddings",
    "query_plans",
    "calibration",
)

# Canonical processed artifact paths grouped by domain
DEFAULT_METADATA_DB_PATH = DEFAULT_PROCESSED_DIR / "catalog" / "corpus_metadata.db"

DEFAULT_KEYFRAME_FAISS_PATH = DEFAULT_PROCESSED_DIR / "visual_keyframes" / "siglip2_keyframes_768d.faiss"
DEFAULT_AUGMENTED_EMBEDDINGS_PATH = DEFAULT_PROCESSED_DIR / "visual_keyframes" / "siglip2_yolo_augmented_864d.npy"

DEFAULT_VIDEOPRISM_CLIP_FAISS_PATH = DEFAULT_PROCESSED_DIR / "visual_clips" / "videoprism_whole_clips_768d.faiss"
DEFAULT_VIDEOPRISM_CLIP_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "visual_clips" / "videoprism_whole_clips_manifest.json"
DEFAULT_VIDEOPRISM_SUBCLIPS_NPY_PATH = DEFAULT_PROCESSED_DIR / "visual_clips" / "videoprism_subclips_768d.npy"
DEFAULT_VIDEOPRISM_SUBCLIPS_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "visual_clips" / "videoprism_subclips_manifest.jsonl"

DEFAULT_E5_FAISS_PATH = DEFAULT_PROCESSED_DIR / "text_evidence" / "e5_ocr_asr_384d.faiss"
DEFAULT_E5_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "text_evidence" / "e5_ocr_asr_manifest.json"
DEFAULT_E5_CAPTION_FAISS_PATH = DEFAULT_PROCESSED_DIR / "text_evidence" / "e5_captions_384d.faiss"
DEFAULT_E5_CAPTION_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "text_evidence" / "e5_captions_manifest.json"

# Canonical cache artifact paths grouped by cache role
DEFAULT_SIGLIP2_QUERY_CACHE_PATH = DEFAULT_CACHE_DIR / "query_embeddings" / "siglip2_query_768d_cache.npz"
DEFAULT_VIDEOPRISM_QUERY_CACHE_PATH = DEFAULT_CACHE_DIR / "query_embeddings" / "videoprism_query_768d_cache.npz"
DEFAULT_E5_QUERY_CACHE_PATH = DEFAULT_CACHE_DIR / "query_embeddings" / "e5_query_384d_cache.npz"
DEFAULT_QUERY_PLANS_CACHE_PATH = DEFAULT_CACHE_DIR / "query_plans" / "query_decomposition_plans.jsonl"
DEFAULT_HUBNESS_RADII_PATH = DEFAULT_CACHE_DIR / "calibration" / "siglip2_keyframe_hubness_radii.npy"

# Legacy flat filenames mapped to canonical relative paths under data/processed/ & data/cache/
PROCESSED_MIGRATION_MAP: tuple[tuple[str, str], ...] = (
    ("metadata.db", "catalog/corpus_metadata.db"),
    ("faiss_index.bin", "visual_keyframes/siglip2_keyframes_768d.faiss"),
    ("augmented_embeddings.npy", "visual_keyframes/siglip2_yolo_augmented_864d.npy"),
    ("faiss_videoprism_clip_768.bin", "visual_clips/videoprism_whole_clips_768d.faiss"),
    ("videoprism_clip_manifest.json", "visual_clips/videoprism_whole_clips_manifest.json"),
    ("videoprism_subclips_768.npy", "visual_clips/videoprism_subclips_768d.npy"),
    ("videoprism_subclips_map.jsonl", "visual_clips/videoprism_subclips_manifest.jsonl"),
    ("faiss_text_e5_384.bin", "text_evidence/e5_ocr_asr_384d.faiss"),
    ("e5_text_manifest.json", "text_evidence/e5_ocr_asr_manifest.json"),
    ("faiss_e5_caption_384.bin", "text_evidence/e5_captions_384d.faiss"),
    ("e5_caption_manifest.json", "text_evidence/e5_captions_manifest.json"),
)
CACHE_MIGRATION_MAP: tuple[tuple[str, str], ...] = (
    ("query_text_embeddings_768d.npz", "query_embeddings/siglip2_query_768d_cache.npz"),
    ("query_videoprism_768d.npz", "query_embeddings/videoprism_query_768d_cache.npz"),
    ("query_e5_embeddings_384d.npz", "query_embeddings/e5_query_384d_cache.npz"),
    ("query_decomposition_cache.jsonl", "query_plans/query_decomposition_plans.jsonl"),
    ("keyframe_hubness_radii.npy", "calibration/siglip2_keyframe_hubness_radii.npy"),
)


def resolve_processed_root(path: Path | str) -> Path:
    """Resolve the processed root directory from either a root dir, domain subdir, or DB/index file path."""
    p = Path(path)
    if p.name in PROCESSED_SUBDIRS and not p.suffix:
        return p.parent
    if p.suffix or p.is_file():
        parent = p.parent
        return parent.parent if parent.name in PROCESSED_SUBDIRS else parent
    return p


def resolve_artifact_path(processed_dir: Path | str, canonical_rel: str, legacy_name: str) -> Path:
    """Resolve a processed artifact preferring the structured layout, falling back to legacy flat layout."""
    root = resolve_processed_root(processed_dir)
    canonical = root / canonical_rel
    if canonical.exists():
        return canonical
    legacy = root / legacy_name
    if legacy.exists():
        return legacy
    return canonical


def resolve_metadata_db_path(processed_dir: Path | str) -> Path:
    """Resolve SQLite corpus catalog DB path under processed_dir."""
    return resolve_artifact_path(processed_dir, "catalog/corpus_metadata.db", "metadata.db")


def resolve_keyframe_faiss_path(processed_dir: Path | str) -> Path:
    """Resolve SigLIP2 keyframe 768d FAISS index path under processed_dir."""
    return resolve_artifact_path(processed_dir, "visual_keyframes/siglip2_keyframes_768d.faiss", "faiss_index.bin")


def resolve_env_db_path(explicit: Path | str | None = None) -> Path:
    """Resolve SQLite DB path from explicit arg, VIDENTRA_DB_PATH, or canonical default (with legacy fallback)."""
    if explicit is not None:
        return Path(explicit)
    env_val = os.environ.get("VIDENTRA_DB_PATH")
    if env_val:
        return Path(env_val)
    return resolve_metadata_db_path(DEFAULT_PROCESSED_DIR)


def resolve_env_faiss_path(explicit: Path | str | None = None) -> Path:
    """Resolve keyframe FAISS index path from explicit arg, VIDENTRA_FAISS_PATH, or canonical default."""
    if explicit is not None:
        return Path(explicit)
    env_val = os.environ.get("VIDENTRA_FAISS_PATH")
    if env_val:
        return Path(env_val)
    return resolve_keyframe_faiss_path(DEFAULT_PROCESSED_DIR)
