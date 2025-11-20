import math
import copy
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

import pdb

class SetTournamentTopK(RerankStrategy):

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

        for index, result in tqdm(enumerate(init_results), desc="Setwise HeapSort"):

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

                # iter-1: build maxheap for the remaining hits (only from the root)
                result = self.run_pass(result, target=0)
                # iter-2: swap the top1 with the last element
                result.hits[0], result.hits[-1] = result.hits[-1], result.hits[0]
                # iter-3: pop the largest (already at the end of the list)
                sorted_hits.append(result.hits.pop(-1))
                print(f"Sorted hits: {len(sorted_hits)}, Remaining hits: {len(result.hits)}")

            # 4. Append the sorted hits to the result
            results[index].hits = sorted_hits + result.hits

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank
        return results

    def run_pass(self, result: Result, target: int) -> List[Result]:

        if target * self._window_size + 1 >= len(result.hits):
            return result

        # Get the comparing set-subtree 
        ## target (current root): i
        ## target's child nodes: i * n_childs + {1/2/.../n_childs}
        idx_pair = [target] + [target * self._window_size + i for i in range(1, self._window_size + 1)]
        idx_pair = tuple(idx for idx in idx_pair if idx < len(result.hits))

        prompt = self._prompt_builder.create_prompt(
            result=result,
            rank_start=0,
            rank_end=len(result.hits),
            idx_pairs=[idx_pair]
        )
        prompt += '['
        outputs = self._llm.generate(prompt, dist_logp=True)[0]
        outputs = outputs[:len(idx_pair)]

        ### version 1 (only swap)
        i_max = np.argmax(outputs)
        if i_max != 0:
            max_idx = idx_pair[i_max]
            result.hits[target], result.hits[max_idx] = result.hits[max_idx], result.hits[target]
            result = self.run_pass(result, target=max_idx)

        return result

# # NOTE: Do we need try all the combintations of the set? now i did it
# # TODO: looking for better implementation
# # NOTE: the first item is target, the last n_child items are the child nodes. 
# # NOTE: the items in the middile are remaining the same
# # NOTE: Do we need to also swap the order of the entire set-subtree? I did it now
# max_2, max_1 = np.sort(outputs)[-2:]
# dummy = (max_1 + max_2) / 2
# final_outputs = [dummy for _ in range(curr_start, curr_end)] 
# final_outputs[0] = outputs[0]
# final_outputs[1-len(idx_pair):] = outputs[1:len(idx_pair)]
# print(f"Final outputs: {outputs}, max_1: {max_1}, max_2: {max_2}")
#
# result = self._result_parser.parse(
#     outputs=[final_outputs],
#     results=[result],
#     rank_start=curr_start,
#     rank_end=curr_end,
# )[0]
#
# # After reranking, we need to sift down if largest element is not at the root
# if outputs[0] != max_1:
#     idx_swap = np.argsort(outputs[:len(idx_pair)])[-1]
#     i_target = idx_pair[idx_swap]
#     result = self.run_pass(result, target=i_target)
#
# return result
