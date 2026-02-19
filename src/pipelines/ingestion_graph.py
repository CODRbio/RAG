"""Backward-compatible pipeline module path.

This wrapper keeps legacy imports (`src.pipelines.ingestion_graph`) working
while the implementation lives in `src.graphs.ingestion_graph`.
"""

from src.graphs.ingestion_graph import IngestionState, build_ingestion_graph

__all__ = ["IngestionState", "build_ingestion_graph"]
