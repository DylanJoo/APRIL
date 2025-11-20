import re
import math
import copy
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

import pdb

class DnC(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 10,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        bucket_idx = [(i, i + self._window_size) for i in range(rank_start, rank_end, self._window_size)]

        n_rels_matrix = np.zeros( (len(results), len(bucket_idx)) )

        # Larger than top 10 results in relevant bucket
        for _ in range(num_runs):

            # get the ranking for relevant passages
            for i_col, (i, j) in enumerate(bucket_idx):
                results, n_rels = self.run_pass(results, i, j)
                # n_rels_matrix[:, i_col] += n_rels

            # rearrange the relevant passages
            for i_row, result in enumerate(results):

                hits_1, hits_0 = [], []
                for i_col, (i, j) in enumerate(bucket_idx):
                    # n_rels = n_rels_matrix[i_row, i_col] 
                    n_rels = self._window_size - self._window_size // len(bucket_idx)
                    hits_1 += result.hits[i: int(i + n_rels)]
                    hits_0 += result.hits[int(i + n_rels):j]

                results[i_row].hits = hits_1 + hits_0
            bucket_idx.pop()

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return results

    def run_pass(
        self,
        results: List[Result],
        curr_start: int,
        curr_end: int,
    ) -> List[Result]:

        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end,
        )
        outputs_filter = self._llm.generate(prompts)

        reranked_results = self._result_parser.parse(
            outputs=outputs_filter,
            results=results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        n_rels = []
        for index, output in enumerate(outputs_filter):
            # n_rel = len(set(re.findall(r"[\d+]", output.split('[x]')[0])))
            n_rel = len(set(re.findall(r"[\d+]", output.split('|')[0])))
            n_rels.append(n_rel)
            
        return reranked_results, n_rels
