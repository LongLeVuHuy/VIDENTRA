"""Evaluation package for offline ranking and localization benchmark metrics."""
from .evaluator import EvaluationReport, evaluate_predictions

__all__ = ["EvaluationReport", "evaluate_predictions"]
