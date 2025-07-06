import copy
from typing import List, Optional, Union, Callable, Dict, Tuple
from ..utils import Result
from .base import BaseResultParser

class ProbParser(BaseResultParser):
    """ A parser for probabilistic ranking permutations."""

#     reranked_results = []
#     for result in init_results:
#         qid = result.qid
#         docids = [doc.docid for doc in result.hits]
#         scores = all_scores[qid]
#         doc_score_pairs = list(zip(docids, scores))
#         doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
#
#         result.hits = result.update_hits_from_scores(doc_score_pairs)
#         reranked_results.append(result)
#
#     return reranked_results

    def parse_and_update(
        self, 
        scores: List[Union[int, float]], 
        results: List[Result],
    ) -> List[Result]:
        """ 
        Only focus on the top-k docs, the other will be the same order?"""

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
