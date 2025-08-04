import re
import math
import copy
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
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        bucket_idx = [(i, i + self._window_size) for i in range(rank_start, rank_end, self._window_size)]

        for index, result in tqdm(enumerate(init_results), 
            desc="Running DnC Reranking",
            total=len(init_results),
        ):

            result_buckets = [Result(qid=result.qid, query=result.query, hits=result.hits[i: j]) for i, j in bucket_idx]

            # Larger than top 10 results in relevant bucket
            for _ in range(2):

                hits_irrel, hits_rel = [], []
                for bucket in result_buckets:
                    hits_0, hits_1 = self.run_pass(bucket, 0, len(bucket.hits))
                    hits_irrel += hits_0
                    hits_rel += hits_1

                hits = hits_irrel + hits_rel
                result_buckets = [Result(qid=result.qid, query=result.query, hits=hits[i: j]) for i, j in bucket_idx]

            results[index].hits = sum([r.hits for r in result_buckets], [])

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

        prompts = self._prompt_builder.create_prompt(
            result=results, 
            rank_start=curr_start, 
            rank_end=curr_end,
            filtering_postfix=True
        )
        output = self._llm.generate(prompts)[0]

        n_rel = len(re.findall(r"[\d+]", output.split('[x]')[0]))

        reranked_results = self._result_parser.parse(
            outputs=[output],
            results=[results],
            rank_start=curr_start,
            rank_end=curr_end,
        )[0]

        print("Number of relevant hits:", n_rel)
        ## split into two parts
        hits_0 = reranked_results.hits[:n_rel]
        hits_1 = reranked_results.hits[n_rel:]

        return hits_0, hits_1
