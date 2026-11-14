"""Localization package for moment boundary proposals and refinement."""
from .proposals import TemporalProposal, generate_proposals
from .refine import refine_moment_boundaries

__all__ = [
    "TemporalProposal",
    "generate_proposals",
    "refine_moment_boundaries",
]
