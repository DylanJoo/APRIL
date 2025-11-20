# NOTE: rank_start is not used.
import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class PairAll(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}
        
        for index, result in enumerate(results):

            ## Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            ## Create prompts for all pairs
            idx_pairs = [(i, j) for i in range(len(result.hits)) for j in range(len(result.hits)) if i != j]
            prompts = self._prompt_builder.create_prompt(
                    result, 
                    rank_start=0, rank_end=rank_end,
                    idx_pairs=idx_pairs,
            )

            ## Iterate over pairs
            scores = []
            for batch_prompts in tqdm(
                batch_iterator(prompts, batch_size),
                desc=f"Batch processing with {batch_size} pairs",
            ):
                batch_scores = self._llm.generate(batch_prompts, binary_probs=True)
                scores.extend(batch_scores)

            ## Score aggregation
            for (i, j), score in zip(idx_pairs, scores):
                score = math.log(score) if self.config.score_aggregation == 'symsumlog' else score
                all_scores[result.qid][i] += score
                all_scores[result.qid][j] += (1-score)

        ## Update results with scores
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("PairAll does not support `run_pass`. Use run instead.")
