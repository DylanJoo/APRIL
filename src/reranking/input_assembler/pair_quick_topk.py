import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class PairQuickTopK(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 32,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        reranked_results = [copy.deepcopy(result) for result in init_results]
        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}

        for index, result in tqdm(
            enumerate(results), total=len(results),
            desc="PairQuickTopK Reranking"
        ):

            ## Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            ## Create prompts for enumerating pairs
            i_run = 0
            pivot = len(result.hits) // 2
            idx_pairs = [(pivot, j) for j in range(len(result.hits)) if j != pivot] + [(i, pivot) for i in range(len(result.hits)) if i != pivot]

            while (pivot > 1) and (i_run < num_runs):
                print(f"Index: {index}, Pivot: {pivot}, Iteration: {i_run}")
                prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end, idx_pairs=idx_pairs) # Same as PairALL

                ## Iterate over pairs
                scores = []
                for batch_prompts in batch_iterator(prompts, batch_size):
                    batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
                    scores.extend(batch_scores)

                ## Pairwise score aggregation
                curr_scores = [0 for _ in result.hits]
                for (i, j), score in zip(idx_pairs, scores):
                    score = math.log(score) if self.config.score_aggregation == 'symsumlog' else score
                    if (j == pivot) and (i == pivot):
                        continue
                    if j == pivot:
                        all_scores[result.qid][i] += score * math.log(abs(i-pivot))
                        curr_scores[i] += score
                    if i == pivot:
                        all_scores[result.qid][j] -= score * math.log(abs(j-pivot))
                        curr_scores[j] -= score

                i_run += 1
                curr_scores = sorted(curr_scores, reverse=True)
                pivot = curr_scores.index(0) - 1
                # idx_pairs = [(pivot, j) for j in range(len(result.hits)) if j < pivot] + [(i, pivot) for i in range(len(result.hits)) if i < pivot]
                idx_pairs = [(pivot, j) for j in range(len(result.hits)) if j < max(pivot, 10)] + [(i, pivot) for i in range(len(result.hits)) if i < max(pivot, 10)]

                ## Update # Iteratively add the scores with different anchor
                result = self._result_parser.parse([all_scores[result.qid]], [result])[0]

            reranked_results[index] = result
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("PairAll does not support `run_pass`. Use run instead.")
