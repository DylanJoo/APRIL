"""
>>>>> [TODO] scoring function return or pseud-score return
"""
import copy
from typing import List, Optional, Union, Callable, Dict, Tuple

class TextListParser:
    """ A parser for text lists that can handle both numerical and alphabetical identifiers. """
    def __init__(
        self, 
        use_alpha=False, 
        variable_passages=False,
    ):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
        else:
            self.id_type = "numerical"

        self.max_doc_lenth = 1024

    # def parse(self, text: str) -> List[str]:
    #     """ Parses a text into a list of strings, each representing a passage. """
    #     passages = text.strip().split("\n")
    #     return [passage.strip() for passage in passages if passage.strip()]
    def parse_and_update(
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
