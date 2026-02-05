import re
import copy
from typing import Optional, Tuple, List, Dict, Union, Any
from tqdm import tqdm

from ..utils import Result
from .base import RerankStrategy

class Dev(RerankStrategy):

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
        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end
        )
        outputs = self._llm.generate(prompts)
        # results = self._result_parser.parse(
        #     outputs=outputs,
        #     results=results,
        #     rank_start=curr_start,
        #     rank_end=curr_end,
        # )

        prompts = self._prompt_builder.create_prompt_batched(
            results=results, 
            rank_start=curr_start, 
            rank_end=curr_end,
            filtering=True
        )
        outputs_f = self._llm.generate(prompts)
        
        # filter the outputs1 
        for i in range(len(outputs)):
            filtered = set(re.findall(r"\d+", outputs_f[i]))
            for idx in filtered:
                outputs[i].replace(f"[{idx}]", "")

        # print(f"Filter: {outputs_f}")
        # print(f"Outputs: {outputs}")
        reranked_results = self._result_parser.parse(
            outputs,
            results=results,
            rank_start=curr_start,
            rank_end=curr_end,
        )
        return reranked_results
