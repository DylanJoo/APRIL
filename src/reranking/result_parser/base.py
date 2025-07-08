""" Just use one parser maybe enough.  """
import copy
from typing import List, Optional, Tuple, Callable, Dict, Union
from abc import ABC, abstractmethod
from ..utils import Result

class ResultParser(ABC):
    """Base class for all parser"""

    def __init__(self, use_alpha=False):
        self._use_alpha = use_alpha

    def parse(
        self, 
        outputs: Union[List[List[Union[float, int]]], List[str]],
        results: List[Result],
        rank_start: int = None,
        rank_end: int = None,
    ) -> Result:
        assert len(outputs) == len(results), "outputs and results must have the same length."

        for index, (output, result) in enumerate(zip(outputs, results)):
            if isinstance(output, str):
                parsed_result = self._parse_responses(output, result, rank_start, rank_end)
            elif all(isinstance(x, list) for x in outputs):
                parsed_result = self._parse_scores(output, result)
            else:
                raise TypeError(f"Unsupported outputs type: {type(outputs)}")
            results[index] = parsed_result
        return results

    def _parse_responses(
        self, 
        permutation: str, 
        result,
        rank_start: int, 
        rank_end: int, 
    ):
        print(f"permutation: {permutation}")
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] 
        print(f"response: {response}, original_rank: {original_rank}")

        # [NOTE] separate this as a standalone function?
        # assign the rank to the unappeared document (assuming they are irrelevant)
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        return result

    def _parse_scores(
        self, 
        scores: List[Union[int, float]], 
        result: Result,
    ) -> List[Result]:
        """ Only focus on the top-k docs, the other will be the same order?"""

        old_hits = copy.deepcopy(result.hits)
        min_score = min(scores) - 1

        for i, score in enumerate(scores):
            result.hits[i]["score"] = score

        if len(scores) < len(old_hits):
            for i in range(len(scores), len(old_hits)):
                result.hits[i]["score"] = result.hits[i-1]["score"] - 1

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

