import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

class Judge(RerankStrategy):

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
            prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end)

            ## Iterate over pairs
            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(
                    batch_prompts, 
                    binary_probs=(self.config.result_parser_name == "binary_prob"),
                    dist_logp=(self.config.result_parser_name == "dist_logp"),
                    rating_logp=(self.config.result_parser_name == "rating_logp"),
                    expected_rating=(self.config.result_parser_name == "expected_rating"),
                )
                scores.extend(batch_scores)

            ## Score aggregation
            for i, score in enumerate(scores):
                all_scores[result.qid][i] = score

        ## Update results with scores
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    def run_pass(self, **kwargs: Any):
        raise NotImplementedError("Judge does not support `run_pass`. Use `run` instead.")
