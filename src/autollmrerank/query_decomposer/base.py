"""
Base class for query decomposition.

Query decomposition is a pre-ranking module that breaks down complex queries
into multiple simpler sub-queries. Each sub-query can then be used for 
independent reranking, and results are aggregated in post-ranking.

This design allows for:
- Handling complex queries with multiple intents
- Improving recall by exploring different query aspects
- Supporting multi-hop reasoning through decomposed steps
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DecomposedQuery:
    """Represents a decomposed query with sub-queries and optional weights."""
    original_query: str
    sub_queries: List[str]
    weights: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.weights is None:
            # Default to equal weights
            self.weights = [1.0 / len(self.sub_queries)] * len(self.sub_queries)
        assert len(self.sub_queries) == len(self.weights), \
            "Number of sub-queries must match number of weights"


class QueryDecomposer(ABC):
    """
    Abstract base class for query decomposition strategies.
    
    Implementations can include:
    - PassThroughDecomposer: No decomposition, returns original query
    - LLMDecomposer: Uses LLM to decompose query into sub-queries
    - RuleBasedDecomposer: Uses rules/patterns for decomposition
    - HybridDecomposer: Combines multiple strategies
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    @abstractmethod
    def decompose(self, query: str, **kwargs) -> DecomposedQuery:
        """
        Decompose a single query into sub-queries.
        
        Args:
            query: The original query string to decompose.
            **kwargs: Additional arguments for decomposition.
            
        Returns:
            DecomposedQuery containing the original query and its sub-queries.
        """
        pass

    def decompose_batch(
        self, 
        queries: List[str], 
        **kwargs
    ) -> List[DecomposedQuery]:
        """
        Decompose multiple queries. Default implementation processes sequentially.
        Subclasses can override for batch optimization.
        
        Args:
            queries: List of query strings to decompose.
            **kwargs: Additional arguments for decomposition.
            
        Returns:
            List of DecomposedQuery objects.
        """
        return [self.decompose(query, **kwargs) for query in queries]


class PassThroughDecomposer(QueryDecomposer):
    """
    A no-op decomposer that returns the original query as a single sub-query.
    Useful as a baseline or when no decomposition is desired.
    """

    def decompose(self, query: str, **kwargs) -> DecomposedQuery:
        return DecomposedQuery(
            original_query=query,
            sub_queries=[query],
            weights=[1.0],
            metadata={"method": "passthrough"}
        )
