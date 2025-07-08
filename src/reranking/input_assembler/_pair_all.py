import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import RerankMode, Result, batch_iterator
from .base import RerankStrategy

class PairAll(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 32, # for fair comparison, this one would should be as same as the query-batch size of listwise
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}
        
        # [NOTE] As the pairall would run larger batch. LLM is less likely to be the bottleneck.
        for result in results:
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end)
            idx_pairs = [(i, j) for i in range(len(result.hits)) for j in range(len(result.hits)) if i != j]
            assert len(prompts) == len(idx_pairs), f"Mismatch between prompts and index pairs, got {len(prompts)} and {len(idx_pairs)} index pairs."

            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
                scores.extend(batch_scores)
            assert len(scores) == len(idx_pairs), "Mismatch between responses and prompts"

            # r(d_i) = P(d_i > d_j) + ( 1 - P(d_j > d_i))
            for (i, j), score in zip(idx_pairs, scores):
                all_scores[result.qid][i] += score
                all_scores[result.qid][j] += (1 - score)

        # Update and return reranked results
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("PairAll does not support run_pass. Use run instead.")
