import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

import pdb

class Search(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 10,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]

        for i_run in range(num_runs):
            for curr_end in tqdm(
                range(rank_end, rank_start, -self._step_size),
                desc=f"Listwise Window Bubble (the {i_run + 1} run)",
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

        curr_start = max(0, curr_end - self._window_size)
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end
        )
        outputs = self._llm.generate(prompts)
        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        prompts = self._prompt_builder.create_prompt_batched(
            results=reranked_results, 
            rank_start=curr_start, 
            rank_end=curr_end,
        )
        outputs = self._llm.generate(prompts)

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=reranked_results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        return reranked_results
