"""
Multi-Query Assembler for reranking with multiple sub-queries.

This assembler takes a list of sub-queries (provided externally, not generated here)
and launches reranking for each sub-query, then aggregates the results into a 
single ranking list.

Design goals:
- Simple: Query decomposition is handled externally (not in this framework)
- Flexible: Supports different base reranking strategies and aggregation methods
- Efficient: Can batch process sub-queries together
"""
import copy
from typing import Optional, List, Any, Dict
from tqdm import tqdm

from ..utils import Result
from ..prompt_builder import PromptBuilder
from ..result_parser import ResultParser
from ..result_aggregator import RRFAggregator, ResultAggregator
from .base import RerankStrategy


class MultiQueryAssembler(RerankStrategy):
    """
    Assembler that handles multi-query reranking with result aggregation.
    
    This is a wrapper around a base reranking strategy that:
    1. Takes a Result with multiple sub-queries attached
    2. Runs the base strategy for each sub-query
    3. Aggregates the sub-query results using RRF or another method
    
    Usage:
        # During reranking, attach sub-queries to the Result object
        result.sub_queries = ["What is ML?", "ML applications"]
        result.sub_query_weights = [0.6, 0.4]  # optional
        
        # The assembler will rerank with each sub-query and aggregate
        reranked = assembler.run(init_results=[result], ...)
    """

    def __init__(
        self,
        config,
        prompt_builder: PromptBuilder,
        llm_provider: Any,
        result_parser: ResultParser,
        base_strategy: Optional[RerankStrategy] = None,
        aggregator: Optional[ResultAggregator] = None,
    ):
        """
        Initialize the multi-query assembler.
        
        Args:
            config: Configuration object.
            prompt_builder: PromptBuilder instance.
            llm_provider: LLM provider instance.
            result_parser: ResultParser instance.
            base_strategy: The underlying reranking strategy to use.
                          If None, will be created based on config.
            aggregator: Result aggregator. Defaults to RRFAggregator(k=60).
        """
        super().__init__(config, prompt_builder, llm_provider, result_parser)
        
        # Set base strategy - if not provided, use the default from config
        self._base_strategy = base_strategy
        
        # Default aggregator is RRF
        self._aggregator = aggregator or RRFAggregator(config=config)

    def _ensure_base_strategy(self) -> None:
        """Validate that base strategy is set."""
        if self._base_strategy is None:
            raise ValueError(
                "Base strategy not set. Call set_base_strategy() first "
                "or provide base_strategy in constructor."
            )

    def set_base_strategy(self, strategy: RerankStrategy) -> None:
        """Set the base reranking strategy."""
        self._base_strategy = strategy

    def set_aggregator(self, aggregator: ResultAggregator) -> None:
        """Set the result aggregator."""
        self._aggregator = aggregator

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:
        """
        Run multi-query reranking with aggregation.
        
        For each Result:
        - If it has sub_queries attached, rerank with each and aggregate
        - Otherwise, rerank normally with the original query
        
        Args:
            init_results: Initial results to rerank. Each Result can have:
                - sub_queries: List[str] - sub-queries to use
                - sub_query_weights: List[float] - optional weights
            rank_start: Start rank for reranking.
            rank_end: End rank for reranking.
            batch_size: Batch size for processing.
            num_runs: Number of reranking passes.
            **kwargs: Additional arguments passed to base strategy.
            
        Returns:
            List of reranked Results with aggregated scores.
        """
        self._ensure_base_strategy()
        
        reranked_results = []
        
        for result in init_results:
            # Check if this result has sub-queries attached
            sub_queries = getattr(result, 'sub_queries', None)
            
            if sub_queries and len(sub_queries) > 1:
                # Multi-query reranking
                weights = getattr(result, 'sub_query_weights', None)
                aggregated = self._rerank_multi_query(
                    result=result,
                    sub_queries=sub_queries,
                    weights=weights,
                    rank_start=rank_start,
                    rank_end=rank_end,
                    batch_size=batch_size,
                    num_runs=num_runs,
                    **kwargs
                )
                reranked_results.append(aggregated)
            else:
                # Single query reranking - use base strategy directly
                reranked = self._base_strategy.run(
                    init_results=[result],
                    rank_start=rank_start,
                    rank_end=rank_end,
                    batch_size=batch_size,
                    num_runs=num_runs,
                    **kwargs
                )
                reranked_results.extend(reranked)
        
        return reranked_results

    def _rerank_multi_query(
        self,
        result: Result,
        sub_queries: List[str],
        weights: Optional[List[float]],
        rank_start: int,
        rank_end: int,
        batch_size: int,
        num_runs: int,
        **kwargs
    ) -> Result:
        """
        Rerank a single result with multiple sub-queries and aggregate.
        
        Args:
            result: The original Result object.
            sub_queries: List of sub-queries to use.
            weights: Optional weights for each sub-query.
            rank_start: Start rank.
            rank_end: End rank.
            batch_size: Batch size.
            num_runs: Number of passes.
            
        Returns:
            Aggregated Result.
        """
        sub_results = []
        original_query = result.query
        
        for sub_query in sub_queries:
            # Create a copy with the sub-query
            sub_result = copy.deepcopy(result)
            sub_result.query = sub_query
            
            # Rerank with sub-query
            reranked = self._base_strategy.run(
                init_results=[sub_result],
                rank_start=rank_start,
                rank_end=rank_end,
                batch_size=batch_size,
                num_runs=num_runs,
                **kwargs
            )
            sub_results.extend(reranked)
        
        # Aggregate all sub-query results
        aggregated = self._aggregator.aggregate(
            sub_query_results=sub_results,
            weights=weights,
            original_query=original_query,
            qid=result.qid
        )
        
        return aggregated

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:
        """
        Run a single pass of reranking.
        Delegates to the base strategy's run_pass.
        """
        self._ensure_base_strategy()
        return self._base_strategy.run_pass(
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
            batch_size=batch_size,
        )


def attach_sub_queries(
    result: Result,
    sub_queries: List[str],
    weights: Optional[List[float]] = None
) -> Result:
    """
    Attach sub-queries to a Result object for multi-query reranking.
    
    This is a helper function to prepare Results for MultiQueryAssembler.
    
    Args:
        result: The Result object to modify.
        sub_queries: List of sub-queries.
        weights: Optional weights for each sub-query. Defaults to equal weights.
        
    Returns:
        The modified Result with sub_queries attached.
    """
    result.sub_queries = sub_queries
    if weights is not None:
        result.sub_query_weights = weights
    else:
        result.sub_query_weights = [1.0 / len(sub_queries)] * len(sub_queries)
    return result
