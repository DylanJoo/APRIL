import copy
from typing import Optional, Tuple, List, Dict, Union, Any
from tqdm import tqdm

from ..utils import Result
from .base import RerankStrategy

class SlidingWindow(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]

        # Listwise Window Bubble
        for i_run in range(num_runs):

            for curr_end in tqdm(
                range(rank_end, rank_start, -self._step_size),
                desc=f"Listwise Window Bubble (the {i_run + 1} run)",
            ):
                curr_start = max(curr_end - self._window_size, rank_start)
                rerank_results = self.run_pass(rerank_results, curr_start, curr_end)

        # Assign reciprocal rank
        for result in rerank_results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return rerank_results

    ## [TODO] maybe we need to set the batch size if one query requires huge amount of prompts/inference. 
    # [NOTE] window size for using logits might have limited to 9, this is not used for now
    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
    ) -> List[Result]:

        prompts = self._prompt_builder.create_prompt_batched(results, rank_start, rank_end)
        outputs = self._llm.generate(prompts)

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results
