"""Adaptive sampling, independent OCR probes, and baseline comparison."""
from .hybrid import (
    HybridAdaptiveSampler,
    SampleRecord,
    SampleManifest,
    sample_frames_hybrid,
    compute_phash_64,
    phash_hamming_distance,
)
from .probes import (
    select_ocr_probes,
    OCRProbeRecord,
    OCRProbeManifest,
)
from .compare import (
    generate_uniform_baseline,
    compute_structural_metrics,
    StructuralMetrics,
)

__all__ = [
    "HybridAdaptiveSampler",
    "SampleRecord",
    "SampleManifest",
    "sample_frames_hybrid",
    "compute_phash_64",
    "phash_hamming_distance",
    "select_ocr_probes",
    "OCRProbeRecord",
    "OCRProbeManifest",
    "generate_uniform_baseline",
    "compute_structural_metrics",
    "StructuralMetrics",
]
