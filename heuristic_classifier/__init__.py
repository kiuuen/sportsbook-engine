"""Facade package for the heuristic classification pipeline.

Re-exports the canonical public components so consumers can import them
directly from ``heuristic_classifier`` as a drop-in replacement for the
legacy monolith module.
"""

from heuristic_classifier.deducer import RouteTemplateDeducer
from heuristic_classifier.engine import HeuristicClassifier
from heuristic_classifier.lexicon import LexiconManager
from heuristic_classifier.models import ClassificationResult
from heuristic_classifier.noise import NoiseAssistant
from heuristic_classifier.prober import AutomatedReplayProber

__all__ = [
    "ClassificationResult",
    "LexiconManager",
    "RouteTemplateDeducer",
    "AutomatedReplayProber",
    "NoiseAssistant",
    "HeuristicClassifier",
]
