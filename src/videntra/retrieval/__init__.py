"""Retrieval engines for visual, text, and multimodal moments."""
from .visual import RankedMoment, VisualRetrievalEngine, temporal_iou

try:
    from .unified_pipeline import UnifiedMultimodalSearchPipeline
except ModuleNotFoundError as exc:
    # Allow minimal worker bundles (e.g., Kaggle feature extractors) that omit
    # unified_pipeline.py or faiss to import videntra.retrieval.visual cleanly.
    if exc.name not in {"videntra.retrieval.unified_pipeline", "faiss"}:
        raise
    UnifiedMultimodalSearchPipeline = None  # type: ignore[assignment]

__all__ = ["RankedMoment", "VisualRetrievalEngine", "temporal_iou", "UnifiedMultimodalSearchPipeline"]

