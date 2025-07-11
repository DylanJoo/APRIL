""" Just use one parser maybe enough.  """
import copy
from typing import List, Optional, Tuple, Callable, Dict, Union
from abc import ABC, abstractmethod
from ..utils import Result

class ResultParser(ABC):

    def __init__(self, use_alpha=False):
        self._use_alpha = use_alpha

    # [TODO] parse all, or maybe multithreading
    def parse(
        self, 
        outputs: Union[List[List[Union[float, int]]], List[str]],
        results: List[Result],
        rank_start: int = None,
        rank_end: int = None,
    ) -> Result:
        assert len(outputs) == len(results), "outputs and results must have the same length."

        for index, (output, result) in enumerate(zip(outputs, results)):
            if isinstance(output, str): # e.g., RankGPT
                parsed_result = self._parse_responses(output, result, rank_start, rank_end)
            if isinstance(output, float): # e.g., Pairwise topk
                parsed_result = self._parse_swap(output, result, rank_end)
            elif all(isinstance(x, list) for x in outputs): # e.g., Pairwise All, Pointwise
                parsed_result = self._parse_scores(output, result)
            else:
                raise TypeError(f"Unsupported outputs type: {type(outputs)}")
            results[index] = parsed_result
        return results

    def _parse_responses(self, permutation: str, result, rank_start: int, rank_end: int):

        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] 
        # print(f"response: {response}, original_rank: {original_rank}")

        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
        return result

    # [NOTE] dylan: i dont think the score matter in this ranking, ignore it for now.
    def _parse_swap(self, swap: float, result: Result, rank_end: int) -> Result:
        if swap < 0.5: # means passage [1] > [2] (hits[rank_end-1] > hits[rank_end-2])
            return result

        init_hits = copy.deepcopy(result.hits)
        result.hits[rank_end - 1] = init_hits[rank_end - 2]
        result.hits[rank_end - 2] = init_hits[rank_end - 1]
        return result

    def _parse_scores(self, scores: List[Union[int, float]], result: Result):

        init_hits = copy.deepcopy(result.hits)
        min_score = min(scores) - 1

        for i in range(len(init_hits)):
            if i <= len(scores) - 1:
                result.hits[i]["score"] = scores[i]
            else:
                result.hits[i]["score"] = min_score
                min_score -= 1

        return result

    def _clean_response(self, response: str) -> str:
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

