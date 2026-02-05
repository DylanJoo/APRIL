"""
Result Aggregator Module

Post-ranking module for aggregating results from multiple sub-query reranking runs.
Supports various aggregation strategies including RRF, coverage-based, and MMR.
"""
from .base import ResultAggregator, PassThroughAggregator, RRFAggregator, AggregatedResult
from .coverage import CoverageAggregator, MMRAggregator

__all__ = [
    'ResultAggregator',
    'PassThroughAggregator',
    'RRFAggregator',
    'AggregatedResult',
    'CoverageAggregator',
    'MMRAggregator',
]
