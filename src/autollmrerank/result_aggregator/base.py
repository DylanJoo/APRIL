"""
Base class for result aggregation.

Result aggregation is a post-ranking module that combines results from multiple
sub-query reranking runs into a final unified ranking. This is the counterpart
to query decomposition.

Aggregation strategies include:
- Reciprocal Rank Fusion (RRF): Combine by reciprocal ranks
- Score-based fusion: Weighted combination of scores
- Coverage-based aggregation: Maximize coverage of query aspects
- Maximal Marginal Relevance (MMR): Balance relevance and diversity
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import copy

from ..utils import Result


@dataclass
class AggregatedResult:
    """Result of aggregating multiple sub-query rankings."""
    qid: str
    query: str  # Original query
    hits: List[Dict[str, Any]]
    sub_query_results: List[Result] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class ResultAggregator(ABC):
    """
    Abstract base class for result aggregation strategies.
    
    Implementations include:
    - RRFAggregator: Reciprocal Rank Fusion
    - CoverageAggregator: Coverage-based aggregation
    - MMRAggregator: Maximal Marginal Relevance
    - WeightedScoreAggregator: Weighted score combination
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    @abstractmethod
    def aggregate(
        self,
        sub_query_results: List[Result],
        weights: Optional[List[float]] = None,
        original_query: Optional[str] = None,
        qid: Optional[str] = None,
        **kwargs
    ) -> Result:
        """
        Aggregate results from multiple sub-query reranking runs.
        
        Args:
            sub_query_results: List of Result objects from sub-query rerankings.
            weights: Optional weights for each sub-query result.
            original_query: The original query before decomposition.
            qid: Query ID.
            **kwargs: Additional aggregation parameters.
            
        Returns:
            A single Result with aggregated hits.
        """
        pass

    def aggregate_batch(
        self,
        batch_sub_results: List[List[Result]],
        batch_weights: Optional[List[List[float]]] = None,
        original_queries: Optional[List[str]] = None,
        qids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Result]:
        """
        Aggregate results for multiple queries.
        
        Args:
            batch_sub_results: List of sub-query results per query.
            batch_weights: Optional weights per query.
            original_queries: Original queries.
            qids: Query IDs.
            **kwargs: Additional parameters.
            
        Returns:
            List of aggregated Results.
        """
        results = []
        for i, sub_results in enumerate(batch_sub_results):
            weights = batch_weights[i] if batch_weights else None
            query = original_queries[i] if original_queries else None
            qid = qids[i] if qids else None
            results.append(self.aggregate(
                sub_results, weights, query, qid, **kwargs
            ))
        return results


class PassThroughAggregator(ResultAggregator):
    """
    A no-op aggregator that returns the first result unchanged.
    Useful when no aggregation is needed (single query, no decomposition).
    """

    def aggregate(
        self,
        sub_query_results: List[Result],
        weights: Optional[List[float]] = None,
        original_query: Optional[str] = None,
        qid: Optional[str] = None,
        **kwargs
    ) -> Result:
        if not sub_query_results:
            raise ValueError("sub_query_results cannot be empty")
        
        result = copy.deepcopy(sub_query_results[0])
        if original_query:
            result.query = original_query
        if qid:
            result.qid = qid
        return result


class RRFAggregator(ResultAggregator):
    """
    Reciprocal Rank Fusion aggregator.
    
    Combines rankings by summing reciprocal ranks across all sub-query results.
    RRF(d) = sum over queries q: 1 / (k + rank(d, q))
    """

    def __init__(self, config: Optional[Any] = None, k: int = 60):
        """
        Args:
            config: Optional configuration.
            k: Constant for RRF formula (default 60).
        """
        super().__init__(config)
        self.k = k

    def aggregate(
        self,
        sub_query_results: List[Result],
        weights: Optional[List[float]] = None,
        original_query: Optional[str] = None,
        qid: Optional[str] = None,
        **kwargs
    ) -> Result:
        if not sub_query_results:
            raise ValueError("sub_query_results cannot be empty")
        
        if weights is None:
            weights = [1.0] * len(sub_query_results)
        
        # Collect RRF scores for each document
        doc_scores: Dict[str, float] = {}
        doc_contents: Dict[str, Dict[str, Any]] = {}
        
        for weight, result in zip(weights, sub_query_results):
            for rank, hit in enumerate(result.hits, start=1):
                docid = hit['docid']
                rrf_score = weight / (self.k + rank)
                doc_scores[docid] = doc_scores.get(docid, 0.0) + rrf_score
                
                # Store document content if not already stored
                if docid not in doc_contents:
                    doc_contents[docid] = hit.get('content_dict', {})
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build result hits
        hits = []
        for rank, (docid, score) in enumerate(sorted_docs, start=1):
            hits.append({
                'docid': docid,
                'score': score,
                'rank': rank,
                'content_dict': doc_contents.get(docid, {})
            })
        
        # Use first result as template
        first_result = sub_query_results[0]
        return Result(
            qid=qid or first_result.qid,
            query=original_query or first_result.query,
            hits=hits
        )
