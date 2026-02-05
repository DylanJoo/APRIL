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
                desc=f"Pairwise Bubble (the {i_run+1} run)",
            ):
                if curr_end - 2 < rank_start: ## NOTE: the last item will be move to the top
                    break
                results = self.run_pass(results, rank_start=rank_start, rank_end=rank_end, target=curr_end)

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
        target: int,
    ) -> List[Result]:

        swaps = [None for _ in range(len(results))]

        # bottom > top
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[(target-1, target-2)] 
        )
        outputs_ij = self._llm.generate(prompts, binary_probs=True)

        # top > bottom
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[(target-2, target-1)]
        )
        outputs_ji = self._llm.generate(prompts, binary_probs=True)

        # aggregation
        for index, (output_ij, output_ji) in enumerate(zip(outputs_ij, outputs_ji)):
            swaps[index] = (output_ij > output_ji)

        reranked_results = self._result_parser.parse(
            outputs=swaps,
            results=results,
            rank_start=rank_start,
            rank_end=target,
        )
        return reranked_results
