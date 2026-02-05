"""
Extended AutoLLMReranker with pre-ranking and post-ranking modules.

This wrapper extends the base AutoLLMReranker to support:
- Pre-ranking: Query decomposition into sub-queries
- Post-ranking: Coverage-based result aggregation

The pipeline becomes:
1. QueryDecomposer: query -> [sub_query_1, sub_query_2, ...]
2. Reranker: For each sub_query, rerank documents
3. ResultAggregator: Combine sub-query results into final ranking
"""
import os
import copy
from typing import Optional, Tuple, List, Dict, Union, Any
from tqdm import tqdm

from .utils import Result, batch_iterator
from .wrapper import AutoLLMReranker
from .query_decomposer import (
    QueryDecomposer, 
    PassThroughDecomposer, 
    LLMDecomposer,
    DecomposedQuery
)
from .result_aggregator import (
    ResultAggregator, 
    PassThroughAggregator, 
    RRFAggregator,
    CoverageAggregator,
    MMRAggregator
)


class ExtendedAutoLLMReranker(AutoLLMReranker):
    """
    Extended reranker with query decomposition and result aggregation.
    
    This class wraps the base reranking functionality and adds:
    - Pre-ranking query decomposition
    - Post-ranking result aggregation
    
    Example usage:
        reranker = ExtendedAutoLLMReranker.from_prebuilt(
            method_name="RankGPT",
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            decomposer="llm",
            aggregator="coverage",
        )
        
        reranked_run = reranker.rerank(
            run=initial_run,
            queries=queries,
            corpus=corpus
        )
    """

    @classmethod
    def from_prebuilt(
        cls,
        method_name: str,
        model_name_or_path: str,
        decomposer: Optional[str] = None,
        aggregator: Optional[str] = None,
        decomposer_config: Optional[Dict] = None,
        aggregator_config: Optional[Dict] = None,
        **kwargs
    ) -> "ExtendedAutoLLMReranker":
        """
        Create an extended reranker from a prebuilt configuration.
        
        Args:
            method_name: Name of the reranking method (e.g., "RankGPT").
            model_name_or_path: Path to the LLM model.
            decomposer: Decomposer type ("passthrough", "llm", or None).
            aggregator: Aggregator type ("passthrough", "rrf", "coverage", "mmr", or None).
            decomposer_config: Additional config for the decomposer.
            aggregator_config: Additional config for the aggregator.
            **kwargs: Additional arguments for base reranker.
            
        Returns:
            Configured ExtendedAutoLLMReranker instance.
        """
        import importlib.resources as pkg_resources
        from .config_manager import ConfigManager
        
        default_path = pkg_resources.files("autollmrerank.configs").joinpath(f"{method_name}.yaml")
        path = pkg_resources.files("autollmrerank.configs").joinpath(f"{method_name}.yaml")
        path = path if path.exists() else default_path

        llmconfig = {'model_name_or_path': model_name_or_path}
        llmconfig.update(kwargs.pop('llm', {}))
        
        # Add decomposer/aggregator to config
        extended_config = {
            'decomposer': decomposer or 'passthrough',
            'aggregator': aggregator or 'passthrough',
            'decomposer_config': decomposer_config or {},
            'aggregator_config': aggregator_config or {},
        }
        kwargs.update(extended_config)
        
        config = ConfigManager(path=path, llm=llmconfig, **kwargs).get_config()
        return cls(config, **kwargs)

    def __init__(self, config, **kwargs) -> None:
        """
        Initialize the extended reranker.
        
        Args:
            config: Configuration object.
            **kwargs: Additional arguments including decomposer and aggregator settings.
        """
        super().__init__(config, **kwargs)
        
        # Initialize decomposer
        decomposer_type = getattr(config, 'decomposer', 'passthrough')
        decomposer_config = getattr(config, 'decomposer_config', {})
        self._decomposer = self._create_decomposer(
            decomposer_type, decomposer_config
        )
        
        # Initialize aggregator
        aggregator_type = getattr(config, 'aggregator', 'passthrough')
        aggregator_config = getattr(config, 'aggregator_config', {})
        self._aggregator = self._create_aggregator(
            aggregator_type, aggregator_config
        )

    def _create_decomposer(
        self, 
        decomposer_type: str, 
        decomposer_config: Dict
    ) -> QueryDecomposer:
        """Create a query decomposer based on type."""
        decomposer_map = {
            'passthrough': PassThroughDecomposer,
            'llm': LLMDecomposer,
        }
        
        if decomposer_type not in decomposer_map:
            raise ValueError(
                f"Unknown decomposer type: {decomposer_type}. "
                f"Available: {list(decomposer_map.keys())}"
            )
        
        decomposer_cls = decomposer_map[decomposer_type]
        
        if decomposer_type == 'llm':
            # LLM decomposer needs the LLM provider
            return decomposer_cls(
                llm_provider=self.assembler._llm,
                config=self.config,
                **decomposer_config
            )
        else:
            return decomposer_cls(config=self.config, **decomposer_config)

    def _create_aggregator(
        self, 
        aggregator_type: str, 
        aggregator_config: Dict
    ) -> ResultAggregator:
        """Create a result aggregator based on type."""
        aggregator_map = {
            'passthrough': PassThroughAggregator,
            'rrf': RRFAggregator,
            'coverage': CoverageAggregator,
            'mmr': MMRAggregator,
        }
        
        if aggregator_type not in aggregator_map:
            raise ValueError(
                f"Unknown aggregator type: {aggregator_type}. "
                f"Available: {list(aggregator_map.keys())}"
            )
        
        aggregator_cls = aggregator_map[aggregator_type]
        return aggregator_cls(config=self.config, **aggregator_config)

    def set_decomposer(self, decomposer: QueryDecomposer) -> None:
        """Set a custom query decomposer."""
        self._decomposer = decomposer

    def set_aggregator(self, aggregator: ResultAggregator) -> None:
        """Set a custom result aggregator."""
        self._aggregator = aggregator

    @AutoLLMReranker.timer
    def rerank(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        query_batch_size: int = 32,
        use_decomposition: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Rerank with optional query decomposition and result aggregation.
        
        Args:
            run: Initial run to be reranked {qid: {docid: score}}.
            queries: Query ID to query string mapping.
            corpus: Document ID to content mapping.
            query_batch_size: Batch size for processing queries.
            use_decomposition: Whether to use query decomposition.
            
        Returns:
            Reranked run {qid: {docid: score}}.
        """
        if not use_decomposition:
            # Fall back to base reranking without decomposition
            return super().rerank(run, queries, corpus, query_batch_size)
        
        # Step 1: Convert run to results
        init_results = self.convert_run_to_result(run, queries, corpus)
        
        # Step 2: Decompose queries
        decomposed_queries = self._decompose_queries(queries)
        
        # Step 3: Rerank for each sub-query and aggregate
        reranked_results = []
        
        for batch_results in tqdm(
            batch_iterator(init_results, size=query_batch_size),
            desc=f"Reranking with decomposition (batch size {query_batch_size})",
            total=len(init_results) // query_batch_size + 1
        ):
            batch_reranked = self._rerank_with_decomposition(
                batch_results, decomposed_queries
            )
            reranked_results.extend(batch_reranked)
        
        # Step 4: Sort and convert back to run format
        for r in reranked_results:
            r.sort_by(field='score')
        
        reranked_run = {}
        for result in reranked_results:
            reranked_run[result.qid] = {}
            for rank, hit in enumerate(result.hits, start=1):
                hit['rank'] = rank
                if 'score' in hit:
                    reranked_run[result.qid].update({hit['docid']: hit['score']})
                else:
                    reranked_run[result.qid].update({hit['docid']: 1/rank})
        
        return reranked_run

    def _decompose_queries(
        self, 
        queries: Dict[str, str]
    ) -> Dict[str, DecomposedQuery]:
        """Decompose all queries using the configured decomposer."""
        decomposed = {}
        query_list = list(queries.items())
        
        # Batch decomposition for efficiency
        qids = [qid for qid, _ in query_list]
        query_texts = [q for _, q in query_list]
        
        decomposed_list = self._decomposer.decompose_batch(query_texts)
        
        for qid, dec in zip(qids, decomposed_list):
            decomposed[qid] = dec
        
        return decomposed

    def _rerank_with_decomposition(
        self,
        batch_results: List[Result],
        decomposed_queries: Dict[str, DecomposedQuery]
    ) -> List[Result]:
        """
        Rerank a batch of results using decomposed queries.
        
        For each query:
        1. Get sub-queries from decomposition
        2. Rerank with each sub-query
        3. Aggregate results
        """
        aggregated_results = []
        
        for result in batch_results:
            qid = result.qid
            decomposed = decomposed_queries.get(qid)
            
            if decomposed is None or len(decomposed.sub_queries) <= 1:
                # No decomposition or single query - use standard reranking
                reranked = self.assembler.run(
                    init_results=[result],
                    rank_start=0,
                    rank_end=min(self.config.rank_end, self.config.top_k),
                    batch_size=1,
                    num_runs=self.config.num_runs,
                )
                aggregated_results.extend(reranked)
            else:
                # Multiple sub-queries - rerank each and aggregate
                sub_results = []
                
                for sub_query in decomposed.sub_queries:
                    # Create a modified result with the sub-query
                    sub_result = copy.deepcopy(result)
                    sub_result.query = sub_query
                    
                    # Rerank with sub-query
                    reranked = self.assembler.run(
                        init_results=[sub_result],
                        rank_start=0,
                        rank_end=min(self.config.rank_end, self.config.top_k),
                        batch_size=1,
                        num_runs=self.config.num_runs,
                    )
                    sub_results.extend(reranked)
                
                # Aggregate sub-query results
                aggregated = self._aggregator.aggregate(
                    sub_query_results=sub_results,
                    weights=decomposed.weights,
                    original_query=decomposed.original_query,
                    qid=qid
                )
                aggregated_results.append(aggregated)
        
        return aggregated_results
