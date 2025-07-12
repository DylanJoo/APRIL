import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from .base import RerankStrategy
from ..utils import Result

class April(RerankStrategy):

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

        for i_run in range(num_runs):
            for w_end in tqdm(   
                range(rank_end, rank_start, -self._step_size), 
                desc=f"APRIL (the {i_run+1} run) from {rank_start}",
            ):
                w_start = max(rank_start, w_end - self._window_size)
                rerank_results = self.run_pass(rerank_results, w_start, w_end)

                # ignore the last pass as it was done and also not a full window
                if w_start == rank_start: 
                    break

            # update the rank_start for the next run
            rank_start = rank_start + self._step_size

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
    ) -> List[Result]:

        prompts = self._prompt_builder.create_prompt_batched(results=results, rank_end=rank_end)
        outputs = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results
