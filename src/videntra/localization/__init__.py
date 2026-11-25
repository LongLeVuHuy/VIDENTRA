"""Localization package for moment boundary proposals and refinement."""
from .proposals import TemporalProposal, generate_proposals
from .refine import refine_moment_boundaries
from .span_grounder import (
    CrossEventSpanNet,
    JointMultiEventGrounder,
    SpanGrounderLGBM,
    extract_multievent_frame_features,
    extract_multievent_span_features,
)

__all__ = [
    "CrossEventSpanNet",
    "JointMultiEventGrounder",
    "SpanGrounderLGBM",
    "TemporalProposal",
    "extract_multievent_frame_features",
    "extract_multievent_span_features",
    "generate_proposals",
    "refine_moment_boundaries",
]
