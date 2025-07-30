import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class Dev(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_end: int,
        batch_size: Optional[int] = 32,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        reranked_results = [None for _ in init_results]
        results = [copy.deepcopy(result) for result in init_results]
        all_points = {}

        for index, result in tqdm(
            enumerate(results), total=len(results), 
            desc="Dev Reranking"
        ):

            for hit in result.hits:
                hit['score'] = 0.0

            for i_run in range(num_runs):
                tour_scores = [result.hits[i]['score'] for i in range(rank_end)]

                ## Tour with pivot
                if i_run == 0:
                    pivot = 5
                else:
                    pivot -= 1

                excluded = [pivot] + [i for i in range(rank_end - i_run * 10, rank_end) if i != pivot]
                idx_pairs = [(pivot, j) for j in range(rank_end) if j not in excluded] + \
                            [(i, pivot) for i in range(rank_end) if i not in excluded]

                scores = self.run_pass(
                    result=result,
                    rank_start=0,
                    rank_end=rank_end,
                    pivot=pivot,
                    idx_pairs=idx_pairs,
                    batch_size=batch_size
                )

                for i, score in enumerate(scores):
                    if i <= rank_end -1:
                        tour_scores[i] += score

                result = self._result_parser.parse([tour_scores], [result])[0]

            reranked_results[index] = result
        return reranked_results

    def run_pass(
        self, 
        result, 
        rank_start: int, 
        rank_end: int, 
        pivot: int,
        idx_pairs: List[Tuple[int, int]],
        batch_size: Optional[int] = 32,
        **kwargs: Any
    ):
        ## create prompts
        win_differences = [0 for _ in result.hits]
        prompts = self._prompt_builder.create_prompt(result, rank_start=rank_start, rank_end=len(result.hits), idx_pairs=idx_pairs)

        scores = []
        for batch_prompts in batch_iterator(prompts, batch_size):
            batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
            scores.extend(batch_scores)

        for (i, j), score in zip(idx_pairs, scores):
            if j == pivot:
                win_differences[i] += score
            if i == pivot:
                win_differences[j] -= score

        return win_differences
