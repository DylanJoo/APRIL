import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from .base import RerankStrategy
from ..utils import Result

class April(RerankStrategy):
    """ To make sure the prompts with same body can be done in the same batch. iterate over the query """

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

            for curr_end in tqdm(
                range(rank_end, rank_start, -self._step_size),
                desc=f"Setwise APRIL (the {i_run+1} run)"
            ):
                results = self.run_pass(results, rank_start, rank_end, curr_end)


        # for index, result in enumerate(rerank_results):
        #
        #     for w_end in tqdm(   
        #         range(rank_end, rank_start, -self._step_size), 
        #         desc=f"APRIL for query:{index}",
        #     ):
        #     # for w_end in range(rank_end, rank_start, -self._step_size):
        #         w_start = max(rank_start, w_end - self._window_size)
        #         w_size = w_end - w_start
        #         idx_pairs = [(i, j) for i in range(w_size) for j in range(w_size) if i != j]
        #
        #         ## prefix caching ## [NOTE] this seems not necessary. it takes 4.5s per window
        #         prompts = self._prompt_builder.create_prompt(result=result, rank_start=w_start, rank_end=w_end)
        #         outputs = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)
        #
        #         # the last window
        #         scores = [0 for _ in range(w_size)]
        #         for (i, j), output in zip(idx_pairs, outputs):
        #             scores[i] += output
        #             scores[j] += 1 - output
        #
        #         # outputs
        #         result = self._result_parser.parse(
        #             outputs=[scores],
        #             results=[result],
        #             rank_start=w_start,
        #             rank_end=w_end,
        #         )[0]
        #
        #         # ignore the last pass as it was done and also not a full window
        #         if w_start == rank_start: 
        #             break
        #
        #     # update the rerank result
        #     rerank_results[index] = result
        #
        # # update the rank_start for the next run
        # rank_start = rank_start + self._step_size
        #
        # # Assign reciprocal rank
        # for result in rerank_results:
        #     for rank, hit in enumerate(result.hits, start=1):
        #         hit['score'] = float(1 / rank)
        #         hit['rank'] = rank
        #
        # return rerank_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("APRIL does not support `run_pass`. Use run instead.")
