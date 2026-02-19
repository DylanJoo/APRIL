""" 
Parse the outputs and results, and return the updated results.
Apply different parsing depending with diffrent LLM outputs.
* non-parallel reranking methods, len(output) == len(results), 
* parallel reranking methods: the output length equals to the number of queries.

- respones: list of permutation (e.g., RankGPT)
- swap: List[bool] (e.g., Pairwise topk) # TODO: should be fixed. Use the doc-index pair instead.
- scores: 
    * absoluate scores: List[List[float]] (e.g., Pairwise All, Pointwise)
    * partial scores: List[List[float]] (e.g., APRIL, Setwise)
"""
import copy
from typing import List, Optional, Tuple, Callable, Dict, Union
from abc import ABC, abstractmethod
from ..utils import Result

class ResultParser(ABC):

    def __init__(self, use_alpha=False):
        self._use_alpha = use_alpha

    # TODO: parse all, or maybe multithreading
    # TODO: make the meaning of `rank_start` and `rank_end` similar across methods?
    def parse(
        self, 
        outputs: Union[List[List[Union[float, int]]], List[str]],
        results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
    ) -> Result:

        assert len(outputs) == len(results), "outputs and results must have the same length."

        for index, (output, result) in enumerate(zip(outputs, results)):
            if isinstance(output, str): # "[1] > [3] > [5] > ... "
                parsed_result = self._parse_responses(output, result, rank_start, rank_end)
            elif isinstance(output, bool): # True if swapping (for top-k)
                parsed_result = self._parse_swap(output, result, rank_end)
            elif isinstance(output, list): # e.g., Pairwise or Pointwise
                if len(output) == len(result.hits):
                    parsed_result = self._parse_absolute_scores(output, result)
                else: 
                    parsed_result = self._parse_scores(output, result, rank_start, rank_end) 
            else:
                raise TypeError(f"Unsupported outputs type: {type(output)}, {output}")
            results[index] = parsed_result
        return results

    # NOTE: this is suitable for continuous items sorting. But not for setwise items initially
    # NOTE: consider to add a design of APRIL
    def _parse_scores(self, scores: List[float], result: Result, rank_start: int, rank_end: int) -> Result:
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        permutation = [(idx, s) for idx, s in zip(range(len(scores)), scores)]
        permutation.sort(key=lambda x: x[1], reverse=True)
        permutation = [(idx, s) for idx, s in permutation if idx < len(cut_range)]
        for j, (p, s) in enumerate(permutation):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[p])
        return result

    def _parse_responses(self, permutation: str, result, rank_start: int, rank_end: int):
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] 
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
        return result

    def _parse_swap(self, swap: bool, result: Result, target: int) -> Result:
        if swap is False: # means passage [1] > [2] (hits[rank_end-1] > hits[rank_end-2])
            return result

        init_hits = copy.deepcopy(result.hits)
        result.hits[target - 1] = init_hits[target - 2]
        result.hits[target - 2] = init_hits[target - 1]
        return result

    def _parse_absolute_scores(self, scores: List[Union[int, float, str]], result: Result):
        """ Assign the scores from top to bottom, and fill the rest with decreasing scores. """
        init_hits = copy.deepcopy(result.hits)
        if isinstance(scores[0], str):
            scores = [float(s) for s in scores]
        min_score = min(scores) - 1

        for i in range(len(init_hits)):
            if i <= len(scores) - 1:
                result.hits[i]["score"] = scores[i]
            else:
                result.hits[i]["score"] = min_score
                min_score -= 1

        return result

    # TODO: use regular expression? 
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

