import math
import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class April(RerankStrategy):
    """To better compare between squential dependence reranking (e.g., listwise).  The pairwise reranking run LLM calls teratively. """

    def run(
        self,
        init_results: List[Result],
        rank_end: int,
        batch_size: Optional[int] = 32,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}
        
        for index, result in enumerate(results):

            ## Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            ## Create prompts for enumerating pairs
            idx_pairs = [(i, j) for i in range(len(result.hits)) for j in range(len(result.hits)) if i != j]
            prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end)

            ## Iterate over pairs
            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
                scores.extend(batch_scores)

            ## Pairwise score aggregation
            for (i, j), score in zip(idx_pairs, scores):
                score = math.log(score) if self.config.score_aggregation == 'symsumlog' else score
                all_scores[result.qid][i] += score
                all_scores[result.qid][j] += (1 - score)

        # Update and return reranked results
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("PairAll does not support `run_pass`. Use run instead.")
