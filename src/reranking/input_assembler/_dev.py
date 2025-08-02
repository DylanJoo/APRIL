import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

import pdb

class Dev(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 10,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        results = [copy.deepcopy(result) for result in init_results]

        for i_run in range(num_runs):
            for curr_end in tqdm(
                range(rank_end, rank_start, -self._step_size),
                desc=f"Listwise Window Bubble (the {i_run + 1} run)",
            ):
                if curr_end - self._window_size < rank_start: 
                    break
                results = self.run_pass(results, rank_start, rank_end, curr_end)

        # Assign reciprocal rank
        for result in results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        curr_end: int,
    ) -> List[Result]:

        curr_start = max(0, curr_end - self._window_size)
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end
        )
        outputs = self._llm.generate(prompts)
        # reranked_results = self._result_parser.parse(
        #     outputs=outputs,
        #     results=results,
        #     rank_start=curr_start,
        #     rank_end=curr_end,
        # )
        print('Reranked:', "\n".join(outputs))

        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end,
            filtering_postfix=True
        )
        prompts = [p for p in prompts]
        outputs_f = self._llm.generate(prompts)
        outputs_f = [o.split(" ||| ")[0] for o in outputs_f]
        print('Truncated Reranked:', "\n".join(outputs))

        reranked_results = self._result_parser.parse(
            outputs=[o2 + " > " + o1 for o1, o2 in zip(outputs_f, outputs)],
            results=results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        return reranked_results

        # for index, result in tqdm(
        #     enumerate(init_results), total=len(init_results), 
        #     desc="Dev Reranking"
        # ):
        #     ## initialize buckets (0, 20), (20, 40), ... (80, 100)
        #     bucket_idx = [(i, i + self._window_size) for i in range(rank_start, rank_end, self._window_size)]
        #     result_buckets = [Result(qid=result.qid, query=result.query, hits=result.hits[i:j]) for i, j in bucket_idx]
        #
        #     ## filter out the negative hits
        #     prompts_f = self._prompt_builder.create_prompt_batched(
        #         results=result_buckets,
        #         rank_start=0,
        #         rank_end=self._window_size,
        #         filtering_postfix=True
        #     )
        #     prompts_f = [p+'[' for p in prompts_f]
        #     outputs_f = self._llm.generate(prompts_f, dist_logp=True, irrelevant_filtering=True)
        #     outputs = [output[:self._window_size] for output in outputs_f]
        #     for i, output in enumerate(outputs):
        #         for j, score in enumerate(output):
        #             result_buckets[i].hits[j]['score'] = score
        #
        #     ## collect back
        #     results[index] = Result(
        #         qid=result.qid, 
        #         query=result.query, 
        #         hits=sum([rb.hits for rb in result_buckets], []),
        #     )
        #     results[index].sort_by('score')

