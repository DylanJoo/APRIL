import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

import pdb

class Dev(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 10,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]

        for index, result in tqdm(
            enumerate(results), total=len(results), 
            desc="Dev Reranking"
        ):
            ## initialize buckets (0, 20), (20, 40), ...
            bucket_idx = [(i, i + self._window_size) for i in \
                    range(rank_start, rank_end, self._window_size)]
            n_buckets = len(result.hits) // self._window_size # TODO: ceiling?
            advanced_size = self._window_size // (n_buckets - 1)
            result_buckets = [
                Result(qid=result.qid, query=result.query, hits=result.hits[i:j]) for i, j in bucket_idx
            ]

            ## rerank the buckets first
            for i_bucket in range(1, n_buckets):
                result_buckets[i_bucket] = self.run_pass(
                    result_buckets[i_bucket], 
                    rank_start=0, 
                    rank_end=self._window_size,
                )

            while all([len(result_buckets[i].hits) != 0 for i in range(1, n_buckets)]):
                print([len(result_buckets[i].hits) for i in range(1, n_buckets)])

                advanced = []
                for i_bucket in range(1, n_buckets):
                    ## append the reranked hits to the first bucket
                    temp_result = copy.deepcopy(result_buckets[i_bucket])
                    result_buckets[i_bucket].hits = temp_result.hits[advanced_size:]
                    advanced.extend(temp_result.hits[:advanced_size])

                result_buckets[0].hits = advanced + result_buckets[0].hits

                ## combine the first and the canidate buckets
                for curr_end in range(self._window_size * 2, 0, -self._step_size):
                    curr_start = max(0, curr_end - self._window_size)
                    result_buckets[0] = self.run_pass(result_buckets[0], curr_start, curr_end)
                    if curr_start ==0:
                        break

            ## collect back
            results[index] = Result(
                qid=result.qid, 
                query=result.query, 
                hits=sum([rb.hits for rb in result_buckets], []),
            )

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return results

    def run_pass(
        self,
        result: List[Result],
        rank_start: int,
        rank_end: int,
    ) -> List[Result]:

        prompt = self._prompt_builder.create_prompt(
            result=result, 
            rank_start=rank_start, 
            rank_end=rank_end
        )
        output = self._llm.generate(prompt)[0]
        output = output[:(rank_end - rank_start)]

        reranked_result = self._result_parser.parse(
            outputs=[output],
            results=[result],
            rank_start=rank_start,
            rank_end=rank_end,
        )[0]
        return reranked_result
