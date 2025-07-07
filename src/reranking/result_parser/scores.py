import copy
from typing import List, Optional, Union, Callable, Dict, Tuple
from ..utils import Result
from .base import BaseResultParser

class ScoreParser(BaseResultParser):
    """ A parser for probabilistic ranking permutations."""

    def parse_scores(
        self, 
        scores: List[Union[int, float]], 
        result: Result,
        rank_start: int = 0,
        rank_end: Optional[int] = None,
    ) -> List[Result]:
        """ Only focus on the top-k docs, the other will be the same order?"""

        old_hits = copy.deepcopy(result.hits[rank_start:rank_end])
        min_score = min(scores) - 1

        for i, score in enumerate(scores):
            result.hits[i]["score"] = s

        if len(scores) < len(old_hits):
            for i in range(len(scores), len(old_hits)):
                result.hits[i]["score"] = result.hits[i-1]["score"] - 1

        return result
