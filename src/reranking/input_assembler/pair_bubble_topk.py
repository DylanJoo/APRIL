import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from .base import RerankStrategy
from ..utils import Result

class PairBubbleTopK(RerankStrategy):
    """ 
    To better compare between squential dependence reranking (e.g., listwise).  The pairwise reranking run LLM calls teratively.  
    [TODO] add number of passes as a parameter.
    """
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
                desc=f"Pairwise Bubble (the {i_run+1} run) from {rank_start}",
            ):
                w_start = max(rank_start, w_end - self._window_size)
                rerank_results = self.run_pass(rerank_results, w_start, w_end)

                # ignore the last pass as it was done and also not a full window
                # if w_start == rank_start: 
                #     break  # [TODO] test the impact of this

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

        # do the reverse
        prompts = self._prompt_builder.create_prompt_batched(results=results, rank_end=rank_end, reverse=True)
        outputs_reverse = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)
        outputs = [ (o - o_reverse) > 0 for o, o_reverse in zip(outputs, outputs_reverse)]

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results

