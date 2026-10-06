import copy
from typing import Optional, Tuple, List, Dict, Union, Any
from tqdm import tqdm

from ..utils import Result
from .base import RerankStrategy

class SlidingWindow(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        num_runs: int = 1,
        **kwargs
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]

        for i_run in range(num_runs):

            for curr_end in tqdm(
                range(rank_end, rank_start, -self._step_size),
                desc=f"Listwise Window Bubble (the {i_run + 1} run)",
            ):
                if curr_end - self._window_size < rank_start:
                    break
                rerank_results = self.run_pass(rerank_results, rank_start, rank_end, curr_end)

        # Assign reciprocal rank
        for result in rerank_results:
            for rank, hit in enumerate(result.hits, start=1):
                hit['score'] = float(1 / rank)
                hit['rank'] = rank

        return rerank_results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        curr_end: int,
    ) -> List[Result]:

        curr_start = max(0, curr_end - self._window_size)

        # Skipping the empty listwise exmaple
        active = [i for i, r in enumerate(results) if len(r.hits[curr_start:curr_end]) >= 2]
        if not active:
            return results
        active_results = [results[i] for i in active] 
        # in the batch, keep only the query-docs that should be reranked

        prompts = self._prompt_builder.create_prompt_batched(
            results=active_results,
            rank_start=curr_start,
            rank_end=curr_end
        )
        prompts = [p + '[' for p in prompts]
        outputs = self._llm.generate(prompts)
        outputs = ['[' + o for o in outputs]

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=active_results,
            rank_start=curr_start,
            rank_end=curr_end,
        )
        for i, result in zip(active, reranked_results):
            results[i] = result
        return results

class SlidingWindowFIRST(SlidingWindow):

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        curr_end: int,
    ) -> List[Result]:

        curr_start = max(0, curr_end - self._window_size)
        # Skip queries whose window is empty or has a single passage (e.g. pooled runs
        # with a large rank_end): nothing to rank, so bypass the LLM call for them.
        active = [i for i, r in enumerate(results) if len(r.hits[curr_start:curr_end]) >= 2]
        if not active:
            return results
        active_results = [results[i] for i in active]

        prompts = self._prompt_builder.create_prompt_batched(
            results=active_results,
            rank_start=curr_start,
            rank_end=curr_end
        )
        prompts = [p + '[' for p in prompts]
        outputs = self._llm.generate(prompts, dist_logp=True)
        outputs = [o[:(curr_end - curr_start)] for o in outputs]

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=active_results,
            rank_start=curr_start,
            rank_end=curr_end,
        )
        for i, result in zip(active, reranked_results):
            results[i] = result
        return results

class SlidingWindowPlus(SlidingWindow):

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
            rank_end=curr_end,
        )
        outputs = self._llm.generate(prompts)
        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        prompts = self._prompt_builder.create_prompt_batched(
            results=reranked_results, 
            rank_start=curr_start, 
            rank_end=curr_end,
            filtering=True
        )
        outputs = self._llm.generate(prompts)
        outputs = [o.split('[x]')[0] for o in outputs]

        reranked_results = self._result_parser.parse(
            outputs=outputs,
            results=reranked_results,
            rank_start=curr_start,
            rank_end=curr_end,
        )

        return results
