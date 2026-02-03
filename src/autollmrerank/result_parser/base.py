""" Parse the outputs and results, and return the updated results.
Apply different parsing depending with diffrent LLM outputs.
* non-parallel reranking methods, len(output) == len(results), 
* parallel reranking methods: the output length equals to the number of queries.

This module uses the Strategy Pattern to handle different output types:
- ResponseParsingStrategy: list of permutation (e.g., RankGPT)
- SwapParsingStrategy: List[bool] (e.g., Pairwise topk)
- AbsoluteScoresParsingStrategy: List[List[float]] (e.g., Pairwise All, Pointwise)
- PartialScoresParsingStrategy: List[List[float]] (e.g., APRIL, Setwise)

Each reranking method can specify its own parsing strategy, or the ResultParser
can auto-detect the appropriate strategy based on output type for backward compatibility.
"""
import copy
from typing import List, Optional, Union, Any
from ..utils import Result
from .strategies import (
    ParsingStrategy,
    ResponseParsingStrategy,
    SwapParsingStrategy,
    AbsoluteScoresParsingStrategy,
    PartialScoresParsingStrategy,
)


class ResultParser:
    """
    Result parser that uses strategy pattern for parsing different LLM output types.
    
    Can be used in two modes:
    1. With explicit strategy: Pass a ParsingStrategy instance for consistent parsing
    2. Auto-detection (default): Automatically selects strategy based on output type
    
    Example with explicit strategy:
        parser = ResultParser(strategy=ResponseParsingStrategy(use_alpha=True))
        results = parser.parse(outputs, results, rank_start, rank_end)
    
    Example with auto-detection (backward compatible):
        parser = ResultParser(use_alpha=True)
        results = parser.parse(outputs, results, rank_start, rank_end)
    """

    def __init__(
        self, 
        use_alpha: bool = False, 
        strategy: Optional[ParsingStrategy] = None
    ):
        """
        Initialize the ResultParser.
        
        Args:
            use_alpha: Whether to use alphabetical indices (A, B, C...) instead of numbers.
                      Only used for ResponseParsingStrategy in auto-detection mode.
            strategy: Optional explicit parsing strategy. If provided, this strategy
                     will be used for all outputs. If None, strategy is auto-detected.
        """
        self._use_alpha = use_alpha
        self._strategy = strategy
        
        # Pre-initialize strategies for auto-detection mode
        self._response_strategy = ResponseParsingStrategy(use_alpha=use_alpha)
        self._swap_strategy = SwapParsingStrategy()
        self._absolute_scores_strategy = AbsoluteScoresParsingStrategy()
        self._partial_scores_strategy = PartialScoresParsingStrategy()

    def parse(
        self, 
        outputs: Union[List[List[Union[float, int]]], List[str], List[bool]],
        results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
    ) -> List[Result]:
        """
        Parse outputs and update results.
        
        Args:
            outputs: List of LLM outputs to parse
            results: List of Result objects to update
            rank_start: Start index for ranking
            rank_end: End index for ranking
            
        Returns:
            Updated list of Result objects
        """
        assert len(outputs) == len(results), "outputs and results must have the same length."

        for index, (output, result) in enumerate(zip(outputs, results)):
            if self._strategy is not None:
                # Use explicit strategy
                parsed_result = self._strategy.parse_single(
                    output, result, rank_start, rank_end
                )
            else:
                # Auto-detect strategy based on output type
                parsed_result = self._auto_parse(output, result, rank_start, rank_end)
            results[index] = parsed_result
        return results
    
    def _auto_parse(
        self,
        output: Any,
        result: Result,
        rank_start: int,
        rank_end: int,
    ) -> Result:
        """
        Auto-detect the appropriate strategy and parse the output.
        
        This provides backward compatibility with the original if-else logic.
        """
        if isinstance(output, str):  # e.g., RankGPT
            return self._response_strategy.parse_single(output, result, rank_start, rank_end)
        elif isinstance(output, bool):  # e.g., Pairwise topk
            return self._swap_strategy.parse_single(output, result, rank_start, rank_end)
        elif isinstance(output, list):  # e.g., Pairwise or Pointwise
            if len(output) == len(result.hits):
                return self._absolute_scores_strategy.parse_single(output, result, rank_start, rank_end)
            else:  # e.g. APRIL, setwise heapsort
                return self._partial_scores_strategy.parse_single(output, result, rank_start, rank_end)
        else:
            raise TypeError(f"Unsupported outputs type: {type(output)}, {output}")
    
    # Keep legacy methods for backward compatibility (deprecated)
    def _parse_scores(self, scores: List[float], result: Result, rank_start: int, rank_end: int) -> Result:
        """Deprecated: Use PartialScoresParsingStrategy instead."""
        return self._partial_scores_strategy.parse_single(scores, result, rank_start, rank_end)

    def _parse_responses(self, permutation: str, result: Result, rank_start: int, rank_end: int) -> Result:
        """Deprecated: Use ResponseParsingStrategy instead."""
        return self._response_strategy.parse_single(permutation, result, rank_start, rank_end)

    def _parse_swap(self, swap: bool, result: Result, target: int) -> Result:
        """Deprecated: Use SwapParsingStrategy instead."""
        return self._swap_strategy.parse_single(swap, result, 0, target)

    def _parse_absolute_scores(self, scores: List[Union[int, float]], result: Result) -> Result:
        """Deprecated: Use AbsoluteScoresParsingStrategy instead."""
        return self._absolute_scores_strategy.parse_single(scores, result, 0, None)

    def _clean_response(self, response: str) -> str:
        """Deprecated: This is now handled by ResponseParsingStrategy."""
        return self._response_strategy._clean_response(response)

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        """Deprecated: This is now handled by ResponseParsingStrategy."""
        return self._response_strategy._remove_duplicate(response)

