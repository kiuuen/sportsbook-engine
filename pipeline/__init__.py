"""Modular ingestion pipeline package.

Re-exports the public API as a drop-in replacement for the legacy
``pipeline_engine_legacy`` monolith::

    from pipeline import PipelineEngine
"""

from pipeline.engine import PipelineEngine

__all__ = ["PipelineEngine"]
