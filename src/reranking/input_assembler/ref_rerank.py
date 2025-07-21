import math
import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class RefRerank(RerankStrategy):
    """To better compare between squential dependence reranking (e.g., listwise).  The pairwise reranking run LLM calls teratively. """

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 32,
        **kwargs: Any
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}
        
        for index, result in enumerate(results):

            ## Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            ## Create prompts for enumerating pairs
            # [NOTE] Testing only the relevant anchor
            A = self.config.anchor_index if self.config.anchor_index is not None else 0
            idx_pairs = [(A, j) for j in range(len(result.hits)) if j != A] + [(i, A) for i in range(len(result.hits)) if i != A]
            prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end, idx_pairs=idx_pairs)

            ## Iterate over pairs
            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
                scores.extend(batch_scores)

            ## Pairwise score aggregation
            for (i, j), score in zip(idx_pairs, scores):
                score = math.log(score) if self.config.score_aggregation == 'symsumlog' else score
                if j == A:
                    all_scores[result.qid][i] += score
                if i == A:
                    all_scores[result.qid][j] -= score

        # Update and return reranked results
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("RefRank does not support `run_pass`. Use run instead.")
