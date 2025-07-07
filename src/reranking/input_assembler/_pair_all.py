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
        batch_size: Optional[int] = 8,
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {result.qid: [0 for _ in result.hits] for result in init_results}
        
        # [NOTE] As the pairall would run larger batch. LLM is less likely to be the bottleneck.
        for result in results:
            prompts = self._prompt_builder.create_prompt(result, rank_start, rank_end)
            idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]
            assert len(prompts) == len(idx_pairs), "Mismatch between prompts and index pairs"

            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(prompts=batch_prompts, prob=self._rerank_mode.use_logits)
                scores.extend(batch_scores)
            assert len(scores) == len(idx_pairs), "Mismatch between responses and prompts"

            # r(d_i) = P(d_i > d_j) + ( 1 - P(d_j > d_i))
            for (i, j), score in zip(id_pairs, scores):
                all_scores[qid][i] += score
                all_scores[qid][j] += (1 - score)

        # Update and return reranked results
        reranked_results = []
        reranked_results = self._result_parser.parse(
            response_scores=[all_scores[r.qid] for r in results],
            results=results,
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("PairAll does not support run_pass. Use run instead.")
