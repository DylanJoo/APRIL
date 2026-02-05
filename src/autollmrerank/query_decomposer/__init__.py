"""
Query Decomposer Module

Simple utility for representing decomposed queries.
Complex query reformulation should be handled externally (in a separate repo).
This module provides just the data structures for passing sub-queries to the reranker.
"""
from .base import QueryDecomposer, PassThroughDecomposer, DecomposedQuery

__all__ = [
    'QueryDecomposer',
    'PassThroughDecomposer', 
    'DecomposedQuery',
]
