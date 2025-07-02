"""
[NOTE] In the assembler class, we probably need different implementations for different reranking strategies. 
e.g., bubble sort with sliding window or sth else

`run_pass()` is the main function to run a single pass of reranking.
`run()` is the main function to run the entire reranking process.
"""
import copy
import random
import re
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
        model_name_or_path: str,
        rerank_mode: RerankMode = RerankMode.RANK_GPT,
        include_system_message: bool = False,
        system_message: str = None,
        context_size: int = 4096,
        window_size: int = 20,
        step_size: int = 10,
        batched: bool = False,
        backend: str = 'vllm',
    ) -> None:

        self._model_name_or_path = model_name_or_path
        self._rerank_mode = rerank_mode

        print(f"[Model] {model_name_or_path}") 
        print(f"{rerank_mode}")

        ## [Module] Formatting the prompt
        self.formatter = PromptBuilder(
            model_name_or_path=model_name_or_path,
            rerank_mode=rerank_mode,
            include_system_message=include_system_message,
            system_message=system_message,
            variable_passages=False,
            use_alpha=False,  # default to False for numerical ordering
        )

        ## [Module] Write a provider module to handle differnet backends
        if backend == 'vllm':
            from reranking.llm_provider.vllm_api import LLM
        if backend == 'litellm':
            from reranking.llm_provider.litellm_api import LLM
        self._llm = LLM(
            model_name_or_path=model_name_or_path,
            temperature=0.0, top_p=1.0, 
            logprobs=None if rerank_mode == RerankMode.RANK_GPT else 30,
            max_tokens=100 if rerank_mode == RerankMode.RANK_GPT else 2,
        )
        # true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        # false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
        # self._llm.set_classification(true_list, false_list)

        ## [Module] Formatting the prompt
        self.processor = ResultParser(rerank_mode=rerank_mode)

        ## [Attibutes]
        # self._context_size = context_size
        self._window_size = window_size
        self._step_size = step_size
        # self._batched = batched

    def run(
        self,
        retrieved_results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:
        r"""Given a list of result files, return a list of reranked results.
        Args:
            retrieved_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        """
        rerank_results = [copy.deepcopy(result) for result in retrieved_results]

        end_pos = rank_end
        start_pos = rank_end - self._window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.run_pass(rerank_results, start_pos, end_pos, logging)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size
        return rerank_results

    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:

        ## Create prompts for each result in the batch
        ### 1. [Listwise] (RankGPT): listwise prompts for sliding windows
        ### 2. [Pairwise] (AllPair): Pairwise prompts for exhausted ranking
        ### 3. [Pairwise] (APRIL): listwise prompts for sliding windows
        prompts = self.formatter.create_prompt_batched(results, rank_start, rank_end)

        # [NOTE] window size for using logits might have limited to 9, this is not used for now
        # if current_window_size is None:
        # current_window_size = self._window_size 
        # assert current_window_size <= 9, "using logits with numerical ordering can only supports window size <= 9"

        # [NOTE] return input length?
        responses = self._llm.generate(
            prompts=[prompt for prompt, _ in prompts], 
            prob=self._rerank_mode.use_logits
        )

        assert len(responses) == len(prompts), "outputs and prompts should have the same length"

        reranked_results = self.processor.parse_response(
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
        return results


    # def _replace_number(self, s: str, use_alpha) -> str:
    #     if use_alpha:
    #         return re.sub(r"\[([A-z]+)\]", r"(\1)", s)
    #     else:
    #         return re.sub(r"\[(\d+)\]", r"(\1)", s)

    # def run_llm_batched(
    #     self,
    #     prompt_texts: List[Union[str, List[Dict[str, str]]]],
    #     current_window_size: Optional[int] = None,
    #     use_logits: bool = False,
    # ) -> List[Tuple[str, int]]:
    #     """Run batched inference with appropriate restrictions for code vs text reranking
    #         [prompt for prompt, _ in prompts], 
    #         use_logits=use_logits
    #         current_window_size=rank_end - rank_start
    #
    #     params = SamplingParams(
    #         min_tokens=min_new_tokens,
    #         max_tokens=max_new_tokens, 
    #         temperature=temp,
    #         logprobs=30,
    #     )
    #     """
    #     temp = 0.
    #     if current_window_size is None:
    #         current_window_size = self._window_size
    #
    #     # [NOTE] Stream like the arugment calls
    #     if use_logits:
    #         max_new_tokens = 2
    #         min_new_tokens = 2
    #         assert current_window_size <= 9, "using logits with numerical ordering can only supports window size <= 9"
    #         params = None
    #         outputs = self._llm.generate(prompt_texts, prob=True)
    #         arr = [self._get_logits_single_digit_batched(output, use_alpha=use_alpha) for output in outputs]
    #         return [(s, len(s)) for s, __ in arr]
    #     else:
    #         params = None
    #         outputs = self._llm.generate(prompt_texts , prob=False)
    #         return [(output, 0) for output in outputs]
    #         # return [
    #         #     (output.outputs[0].text, len(output.outputs[0].token_ids))
    #         #     for output in outputs
    #         # ]

