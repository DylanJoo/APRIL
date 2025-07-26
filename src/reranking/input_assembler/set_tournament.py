import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class SetTouranment(RerankStrategy):

    def grouping(self, result: Result, size: int = 10) -> Result:
        # NOTE: The ranking lists should be in the same length.
        # NOTE: double check the logit
        idx_pairs = []
        for i in range(0, len(result.hits), size):
            idx_pair = [i for i in range(i, size) if (i+size) < len(result.hits) - 1]
            idx_pairs.append(tuple(idx_pairs))
        return idx_pairs

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        num_runs: Optional[int] = 1,
        **kwargs
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]
        for i_run in range(num_runs):
            groups = self.grouping(init_results[0], size=self._window_size)
            rerank_results = self.run_pass(init_results, idx_pairs=groups)

        # Assign reciprocal rank
        for result in rerank_results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return rerank_results

    def run_pass(
        self,
        results: List[Result],
        idx_pairs: List[Tuple[int, int, int]] = None,
    ) -> List[Result]:

        # NOTE: align all the rank_end with the length of result? 
        prompts = self._prompt_builder.create_prompt(results=results, rank_end=len(results[0].hits))

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results

    # NOTE: referene
    # def run_pass(
    #     self,
    #     results: List[Result],
    #     rank_start: int,
    #     rank_end: int,
    #     curr_end: int,
    # ) -> List[Result]:
    #     """ 
    #     Args:
    #         rank_start: The starting rank for reranking.
    #         rank_end: The ending rank for reranking.
    #         curr_end: The current end rank for the reranking pass.
    #     
    #     Example: (rank_start=100) [(97, 98, 99)]
    #     """
    #     idx_pairs = permutations(range(curr_end - self._window_size, curr_end), 2) # pairwise
    #
    #     scores = [0 for _ in range(len(results))]
    #
    #     for idx_pairs in idx_pairs:
    #         prompts = self._prompt_builder.create_prompt_batched(
    #             results=results, 
    #             rank_start=0,
    #             rank_end=rank_end, 
    #             idx_pairs=[idx_pair]
    #         )
    #         outputs = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)
    #
    #         for index, output in enumerate(outputs):
    #             scores[index] += output # p(i > j) - p(j > i)
    #
    #     reranked_results = self._result_parser.parse(
    #         outputs=outputs,
    #         results=results,
    #         rank_start=rank_start,
    #         rank_end=rank_end,
    #     )
    #     return reranked_results
