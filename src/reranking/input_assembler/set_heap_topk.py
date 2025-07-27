# NOTE: consider change the name to swap topk?
import math
import copy
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

import pdb

class SetMaxHeapTopK(RerankStrategy):

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

        for index, result in tqdm(
            enumerate(results), 
            desc="Setwise HeapSort", ntotal=len(results)
        ):
            # 0. Get the last parent
            i_parent = len(result.hits) // self._window_size - 1

            # 1. build maxheap (traverse each paraents)
            for i_visit in range(i_parent, -1, -1):
                result = self.run_pass(result, target=i_visit)

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return results

    def run_pass(self, results: List[Result], target: int) -> List[Result]:

        # Get the comparing set-subtree 
        # target and its child nodes: i * n_childs + {1/2/.../n_childs}
        idx_pair = tuple(target, target * self._set_size + i for i in range(1, self._set_size + 1))
        curr_start, curr_end = min(idx_pair), max(idx_pairs) + 1
        breakpoint()

        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=0,
            rank_end=rank_end, 
            idx_pairs=[idx_pair]
        )
        # NOTE: Do we need try all the combintations of the set? now i did it
        outputs = self._llm.generate(prompts, dist_logp=True)
        breakpoint()

        # TODO: looking for better implementation
        # NOTE: the first item is target, the last n_child items are the child nodes. 
        # NOTE: the items in the middile are remaining the same
        # NOTE: Do we need to also swap the order of the entire set-subtree? I did it now
        max_1 = max(outputs)
        max_2 = max([i for i in outputs if i != max_output])
        dummy = (max_1 + max_2) / 2
        final_outputs = [dummy for _ iln range(curr_start, curr_end)] 
        final_outputs[0] = outputs[0]
        final_outputs[-len(idx_pair):] = outputs[1:len(idx_pair)]

        reranked_results = self._result_parser.parse(
            outputs=permutations,
            results=results,
            rank_start=min(idx_pair),
            rank_end=max(idx_pair) + 1
        )
        return reranked_results
