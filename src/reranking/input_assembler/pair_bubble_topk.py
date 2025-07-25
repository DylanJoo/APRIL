# NOTE: consider change the name to swap topk?
import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

class PairBubbleTopK(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 32,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]

        for i_run in range(num_runs):

            for curr_end in tqdm(   
                range(rank_end, rank_start, -self._step_size), 
                desc=f"Pairwise Bubble (the {i_run+1} run)",
            ):
                rerank_results = self.run_pass(rerank_results, rank_start, rank_end, curr_end)

        # Assign reciprocal rank
        for result in rerank_results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return rerank_results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        curr_end: int,
    ) -> List[Result]:
        """ 
        Args:
            rank_start: The starting rank for reranking.
            rank_end: The ending rank for reranking.
            curr_end: The current end rank for the reranking pass.
        
        Example: (rank_start=100) [(97, 98, 99)]
        """
        swaps = [None for _ in range(len(results))]

        # I > J
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[(curr_end-2, curr_end-1)]
        )
        outputs_ij = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)

        # J > I
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[(curr_end-2, curr_end-1)]
        )
        outputs_ji = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)

        for index, (output_ij, output_ji) in enumerate(zip(outputs_ij, outputs_ji)):
            swaps[index] = (output_ij > output_ji)

        reranked_results = self._result_parser.parse(
            outputs=swaps,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results
