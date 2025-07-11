import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

class WindowBubble(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        **kwargs
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]

        # Bubble sort (from bottom-window to top-window)
        end_pos = rank_end
        start_pos = rank_end - self._window_size

        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.run_pass(rerank_results, start_pos, end_pos)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size

        for result in rerank_results: # reciprocal rank
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1/rank)
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
        outputs = self._llm.generate(prompts=prompts, prob=self._rerank_mode.use_logits)

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results

