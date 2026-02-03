""" 
Parsing strategies for different LLM output types.

This module implements the Strategy Pattern to replace the type-checking if-else 
chains in the result parser. Each strategy handles a specific type of LLM output:

- ResponseParsingStrategy: For listwise ranking (e.g., RankGPT) - parses string permutations
- SwapParsingStrategy: For pairwise/setwise topk - parses boolean swap decisions
- AbsoluteScoresParsingStrategy: For pointwise/pairwise all - parses full score lists
- PartialScoresParsingStrategy: For APRIL/setwise - parses partial score lists
"""
import copy
from abc import ABC, abstractmethod
from typing import List, Union, Any

from ..utils import Result


class ParsingStrategy(ABC):
    """
    Abstract base class for parsing strategies.
    
    Each concrete strategy implements a specific way to parse LLM outputs
    and update the result accordingly.
    """
    
    @abstractmethod
    def parse_single(
        self,
        output: Any,
        result: Result,
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> Result:
        """
        Parse a single output and update the result.
        
        Args:
            output: The LLM output to parse (type depends on strategy)
            result: The Result object to update
            rank_start: Start index for ranking
            rank_end: End index for ranking
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Updated Result object
        """
        pass


class ResponseParsingStrategy(ParsingStrategy):
    """
    Strategy for parsing string permutation responses (e.g., RankGPT).
    
    Expects output to be a string containing document indices that represents
    the reranked order of documents.
    """
    
    def __init__(self, use_alpha: bool = False):
        self._use_alpha = use_alpha
    
    def parse_single(
        self,
        output: str,
        result: Result,
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> Result:
        response = self._clean_response(output)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] 
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
        return result
    
    def _clean_response(self, response: str) -> str:
        ALPH_START_IDX = 64  # ASCII 'A' starts at 65, so we use 64 to map 'A' to 1
        new_response = ""
        if self._use_alpha:
            for c in response:
                if not c.isalpha():
                    new_response += " "
                else:
                    new_response += str(ord(c) - ALPH_START_IDX)
            new_response = new_response.strip()
        else:
            for c in response:
                if not c.isdigit():
                    new_response += " "
                else:
                    new_response += c
            new_response = new_response.strip()
        return new_response
    
    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response


class SwapParsingStrategy(ParsingStrategy):
    """
    Strategy for parsing swap decisions (e.g., Pairwise TopK).
    
    Expects output to be a boolean indicating whether to swap two documents.
    Note: rank_end is used as 'target' for this strategy.
    """
    
    def parse_single(
        self,
        output: bool,
        result: Result,
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> Result:
        target = rank_end  # For swap, rank_end is used as target
        if output is False:  # means passage [1] > [2] (hits[rank_end-1] > hits[rank_end-2])
            return result

        init_hits = copy.deepcopy(result.hits)
        result.hits[target - 1] = init_hits[target - 2]
        result.hits[target - 2] = init_hits[target - 1]
        return result


class AbsoluteScoresParsingStrategy(ParsingStrategy):
    """
    Strategy for parsing absolute scores (e.g., Pointwise, PairAll).
    
    Expects output to be a list of scores, one for each document.
    Assigns scores to documents and fills remaining with decreasing scores.
    """
    
    def parse_single(
        self,
        output: List[Union[int, float]],
        result: Result,
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> Result:
        init_hits = copy.deepcopy(result.hits)
        min_score = min(output) - 1

        for i in range(len(init_hits)):
            if i < len(output):
                result.hits[i]["score"] = output[i]
            else:
                result.hits[i]["score"] = min_score
                min_score -= 1

        return result


class PartialScoresParsingStrategy(ParsingStrategy):
    """
    Strategy for parsing partial scores (e.g., APRIL, Setwise HeapSort).
    
    Expects output to be a list of scores for a subset of documents,
    which are then sorted and used to reorder the specified range.
    """
    
    def parse_single(
        self,
        output: List[float],
        result: Result,
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> Result:
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        permutation = [(idx, s) for idx, s in zip(range(len(output)), output)]
        permutation.sort(key=lambda x: x[1], reverse=True)
        for j, (p, s) in enumerate(permutation):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[p])
        return result
