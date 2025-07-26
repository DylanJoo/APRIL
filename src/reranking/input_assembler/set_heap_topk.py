# NOTE: consider change the name to swap topk?
import math
import copy
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

import pdb

class SetBubbleTopK(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        num_runs: int = 10,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]

        for i_run in range(num_runs):

            for curr_end in tqdm(   
                range(rank_end, rank_start, -self._step_size), 
                desc=f"Setwise Bubble (the {i_run+1} run)"
            ):
                results = self.run_pass(results, rank_start, rank_end, curr_end)

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        curr_end: int,
    ) -> List[Result]:

        permutations = [None for _ in range(len(results))]

        # I > (J, K, L, ...)
        curr_start = max(0, curr_end - self._window_size)
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[tuple(range(curr_end - self._window_size, curr_end))],
        )
        outputs = self._llm.generate(prompts, dist_logp=True)

        # NOTE: make separate setsize and window size
        for index, output in enumerate(outputs):
            permutation = np.array(output).argsort()[::-1]
            permutation = [str(p+1) for p in permutation] # index starts from 1
            permutations[index] = " > ".join(permutation)

        reranked_results = self._result_parser.parse(
            outputs=permutations,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results

    # def _parse_responses(self, permutation: str, result, rank_start: int, rank_end: int):
    #     response = self._clean_response(permutation)
    #     response = [int(x) - 1 for x in response.split()]
    #     response = self._remove_duplicate(response)
    #     cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
    #     original_rank = [tt for tt in range(len(cut_range))]
    #     response = [ss for ss in response if ss in original_rank]
    #     response = response + [tt for tt in original_rank if tt not in response] 
    #     for j, x in enumerate(response):
    #         result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
    #     return result

    # Reference for FIRST
    # def run_pass(
    #     self,
    #     results: List[Result],
    #     rank_start: int,
    #     rank_end: int,
    #     curr_end: int,
    # ) -> List[Result]:
    #
    #     permutations = [None for _ in range(len(results))]
    #
    #     # I > (J, K, L, ...)
    #     curr_start = max(0, curr_end - self._window_size)
    #     prompts = self._prompt_builder.create_prompt_batched(
    #         results=results, 
    #         rank_start=0,
    #         rank_end=rank_end, 
    #         idx_pairs=[tuple(range(curr_end - self._window_size, curr_end))],
    #     )
    #     outputs = self._llm.generate(prompts, dist_logp=True)
    #
    #     # NOTE: make separate setsize and window size
    #     for index, output in enumerate(outputs):
    #         permutation = np.array(output).argsort()[::-1].tolist()
    #         permutations[index] = permutation[:self._window_size]  # this is FIRST
    #
    #     reranked_results = self._result_parser.parse(
    #         outputs=winner,
    #         results=results,
    #         rank_start=rank_start,
    #         rank_end=rank_end,
    #     )
    #     return reranked_results
