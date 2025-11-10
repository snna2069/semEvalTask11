"""
SemEval 2026 Task 11 - Plausibility Prediction

"""

__version__ = "0.1.0"

from .inference import PlausibilityPredictor, process_jsonl_file
from .evaluate import (
    calculate_accuracy,
    calculate_tce,
    calculate_intra_plausibility,
    calculate_cross_plausibility,
    run_evaluation
)

__all__ = [
    'PlausibilityPredictor',
    'process_jsonl_file',
    'calculate_accuracy',
    'calculate_tce',
    'calculate_intra_plausibility',
    'calculate_cross_plausibility',
    'run_evaluation'
]