import math
import copy
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

import pdb

class PairMaxHeapTopK(RerankStrategy):

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

        for index, result in tqdm(enumerate(init_results), desc="Pairwise HeapSort"):

            sorted_hits = []

            # 0. Get the last parent (index - 1) // d
            i_parent = (len(result.hits) - 2) // self._window_size
            # 1. build maxheap (traverse each paraents)
            for i_visit in range(i_parent, -1, -1):
                result = self.run_pass(result, target=i_visit)
            # 2 swap the top1 with the last element
            result.hits[0], result.hits[-1] = result.hits[-1], result.hits[0]
            # 3 pop the largest (already at the end of the list)
            sorted_hits.append(result.hits.pop(-1))

            # Iteration until we have enough sorted hits
            while len(sorted_hits) < num_runs: # TODO: maybe we should use variable top_k

                # Guard: stop if no more hits to process
                if len(result.hits) == 0:
                    break

                # iter-1: build maxheap for the remaining hits (only from the root)
                result = self.run_pass(result, target=0)
                # iter-2: swap the top1 with the last element
                result.hits[0], result.hits[-1] = result.hits[-1], result.hits[0]
                # iter-3: pop the largest (already at the end of the list)
                sorted_hits.append(result.hits.pop(-1))
                # print(f"Sorted hits: {len(sorted_hits)}, Remaining hits: {len(result.hits)}")

            # 4. Append the sorted hits to the result
            results[index].hits = sorted_hits + result.hits

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank
        return results

    def run_pass(self, result: Result, target: int) -> List[Result]:
        left = target * self._window_size + 1
        right = target * self._window_size + 2
        swap_right, swap_left = 0, 0

        if left >= len(result.hits):
            return result

        if left < len(result.hits):
            prompt = self._prompt_builder.create_prompt(
                result=result,
                rank_start=0,
                rank_end=len(result.hits),
                idx_pairs=[(target, left), (left, target)]
            )
            outputs = self._llm.generate(prompt, binary_probs=True)
            swap_left = (outputs[1] - outputs[0]) # Left>Root - Root>Left

        if right < len(result.hits):
            prompt = self._prompt_builder.create_prompt(
                result=result,
                rank_start=0,
                rank_end=len(result.hits),
                idx_pairs=[(target, right), (right, target)]
            )
            outputs = self._llm.generate(prompt, binary_probs=True)
            swap_right = (outputs[1] - outputs[0]) # Right>Root - Left>Left

        if (swap_right > 0) and (swap_right > swap_left):
            result.hits[target], result.hits[right] = result.hits[right], result.hits[target]
            result = self.run_pass(result, target=right)

        elif (swap_left > 0) and (swap_left > swap_right):
            result.hits[target], result.hits[left] = result.hits[left], result.hits[target]
            result = self.run_pass(result, target=left)

        return result
