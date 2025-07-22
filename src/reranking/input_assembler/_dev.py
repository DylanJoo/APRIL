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

            # Initialize points
            for hit in result.hits:
                hit['score'] = 0.0

            for i_run in tqdm(range(num_runs), desc=f"Pairiwise Dev for {result.qid}"):

                tour_scores = [result.hits[i]['score'] for i in range(rank_end)]
                # bottom_pivot = cand_pivot[len(cand_pivot) // 2] if len(cand_pivot) > 0 else pivot + 1

                ## Tour with pivot
                pivot = tour_scores.index(min([s for s in tour_scores if s >= 0]))
                idx_pairs = [(pivot, j) for j in range(rank_end) if j != pivot] + \
                            [(i, pivot) for i in range(rank_end) if i != pivot]

                scores = self.run_pass(
                    result=result,
                    rank_start=0,
                    rank_end=rank_end,
                    pivot=pivot,
                    idx_pairs=idx_pairs,
                    batch_size=batch_size
                )
                for i, score in enumerate(scores):
                    tour_scores[i] += score

                pivot = scores.index(0)
                print(f"Scores of first run: {tour_scores}, pivot={pivot}")

                ## Tour with top-pivot (select the smallest positive score)
                top_pivot_score = min([s for s in tour_scores if s > 0])
                top_pivot = tour_scores.index(top_pivot_score) if len(top_pivot_score) > 0 else rank_end // 2
                top_idx_pairs = [(i, top_pivot) for i in range(top_pivot) if i != top_pivot] + \
                                [(top_pivot, j) for j in range(top_pivot) if j != top_pivot]

                if len(top_idx_pairs) != 0:
                    scores = self.run_pass(
                        result=result,
                        rank_start=0,
                        rank_end=rank_end,
                        pivot=top_pivot,
                        idx_pairs=top_idx_pairs,
                        batch_size=batch_size
                    )
                    for i, score in enumerate(scores):
                        tour_scores[i] += score
                print(f"Second run {tour_scores}")

                ## Tour with bottom-pivot
                cand_pivot = [i for i, s in enumerate(points) if (s == 0) and i in range(pivot + 1, rank_end)]
                bottom_pivot = cand_pivot[len(cand_pivot) // 2] if len(cand_pivot) > 0 else pivot + 1
                bottom_pivot = pivot + 1
                bottom_idx_pairs = [(i, bottom_pivot) for i in range(pivot + 1, rank_end) if i != bottom_pivot] + \
                                   [(bottom_pivot, j) for j in range(pivot + 1, rank_end) if j != bottom_pivot]

                if len(bottom_idx_pairs) != 0:
                    scores = self.run_pass(
                        result=result,
                        rank_start=0,
                        rank_end=rank_end,
                        pivot=bottom_pivot,
                        idx_pairs=bottom_idx_pairs,
                        batch_size=batch_size
                    )
                    for i, score in enumerate(scores):
                        tour_scores[i] += score

                result = self._result_parser.parse([tour_scores], [result])[0]
                tour_scores = sorted(tour_scores, reverse=True)

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
        prompts = self._prompt_builder.create_prompt(result, rank_start=rank_start, rank_end=rank_end, idx_pairs=idx_pairs)

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
