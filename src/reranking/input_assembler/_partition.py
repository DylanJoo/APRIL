import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from .base import RerankStrategy
from ..utils import Result

class Dev(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        num_runs: Optional[int] = 1,
        **kwargs
    ) -> List[Result]:

        # w_end = [rank_end, rank_end - step_size, ...] 
        rerank_results = [copy.deepcopy(result) for result in init_results]

        # find a pivot for the first run
        for i_run in range(num_runs):

            for w_end in tqdm(   
                range(rank_end, rank_start, -self._step_size), 
                desc=f"Pairwise Bubble (the {i_run+1} run) from {rank_start}",
            ):

            rank_pivot = (rank_start + rank_end) // 2)
            reranked_results, rank_pivot = self.run_pass(rerank_results, rank_start, rank_end, rank_pivot)
            rank_end = rank_pivot

            if rank_end <= rank_start:
                break

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
        rank_pivot: Optional[int] = None,
    ) -> List[Result]:

        prompts = self._prompt_builder.create_prompt_batched(results=results, rank_end=rank_end)
        outputs = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)

        prompts = self._prompt_builder.create_prompt_batched(results=results, rank_end=rank_end, reverse=True)
        outputs_reverse = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)
        outputs = [ (o - o_reverse) > 0 for o, o_reverse in zip(outputs, outputs_reverse)]
        rank_pivot = outputs.index(0)

        # save the results maybe ?
        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_end=rank_end,
        )
        return reranked_results, rank_pivot

