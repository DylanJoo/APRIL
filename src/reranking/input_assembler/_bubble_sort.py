"""
[NOTE] In the assembler class, we probably need different implementations for different reranking strategies. 
e.g., bubble sort with sliding window or sth else

`run_pass()` is the main function to run a single pass of reranking.
`run()` is the main function to run the entire reranking process.

[NOTE] argument setting
Add some method-specific configs to config files (instead of the rerankmode)
"""
import copy
import random
import re
import json
from typing import Optional, Tuple, List, Dict, Union, Any

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ftfy import fix_text

from reranking.utils import RerankMode, Result
from reranking.prompt_builder import PromptBuilder
from reranking.result_parser import ResultParser

ALPH_START_IDX = ord('A')-1

class BubbleSort:

    def __init__(
        self, 
        config, 
        rerank_mode: RerankMode,
        prompt_builder: PromptBuilder,
        llm_provider: Any,  # [TODO] add llm provider
        result_parser: ResultParser,
    ):
        ## [Module] Write a provider module to handle differnet backends
        self.config = config
        self._prompt_builder = prompt_builder
        self._llm = llm_provider
        self._result_parser = result_parser

        # attrobutes
        self._rerank_mode = rerank_mode
        self._window_size = self.config.window_size
        self._step_size = self.config.step_size

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:
        r"""Given a list of result files, return a list of reranked results.
        Args:
            init_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
        """
        # [TODO] set batch_
        rerank_results = [copy.deepcopy(result) for result in init_results]

        end_pos = rank_end
        start_pos = rank_end - self._window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.run_pass(rerank_results, start_pos, end_pos)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size
        return rerank_results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:

        ## Create prompts for each result in the batch
        ## [TODO] maybe we need to set the batch size if one query requires huge amount of prompts/inference. 
        prompts = self._prompt_builder.create_prompt_batched(results, rank_start, rank_end)
        with open('/home/hltcoe/jhueiju/APRIL/prompt.json', 'w') as f:
            json.dump(prompts, f, indent=4)

        # [NOTE] window size for using logits might have limited to 9, this is not used for now
        # [NOTE] return input length?
        responses = self._llm.generate(prompts=[prompt for prompt, _ in prompts], prob=self._rerank_mode.use_logits)

        assert len(responses) == len(prompts), "outputs and prompts should have the same length"

        reranked_results = self._result_parser.parse_response(
            response_texts=responses,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        # for index, (result, (prompt, in_token_count)) in enumerate(zip(results, prompts)):
        #     permutation, out_token_count = batched_results[index]
        #     ranking_exec_info = RankingExecInfo(prompt, permutation, in_token_count, out_token_count)
        #     if result.ranking_exec_summary is None:
        #         result.ranking_exec_summary = []
        #     result.ranking_exec_summary.append(ranking_exec_info)
        return reranked_results
