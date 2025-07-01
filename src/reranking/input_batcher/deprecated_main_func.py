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

class SlidingWindow:

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

        print(f"Using model: {model_name_or_path} for reranking with mode: {rerank_mode}")
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

        ## [Instance] attibutes
        # self._context_size = context_size
        self._window_size = window_size
        self._step_size = step_size
        # self._batched = batched

    def permutation_pipeline_batched(
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

        ## Run LLM on the batched prompts
        responses = self.run_llm_batched(
            prompt_texts=[prompt for prompt, _ in prompts], 
            use_logits=False,
            current_window_size=rank_end - rank_start
        )

        ## Parse permutations
        # [NOTE] (1) scoring (2) bubble sort
        reranked_results = self.processor.parse_response(
            response_texts=[response for response, _ in responses],
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

    def sliding_windows_batched(
        self,
        retrieved_results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:
        r"""Origianl rank_llm parameters as attributes.
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
            rerank_results = self.permutation_pipeline_batched(rerank_results, start_pos, end_pos, logging)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size
        return rerank_results


    def run_llm_batched(
        self,
        prompt_texts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
        use_logits: bool = False,
    ) -> List[Tuple[str, int]]:
        """Run batched inference with appropriate restrictions for code vs text reranking
            [prompt for prompt, _ in prompts], 
            use_logits=use_logits
            current_window_size=rank_end - rank_start

        params = SamplingParams(
            min_tokens=min_new_tokens,
            max_tokens=max_new_tokens, 
            temperature=temp,
            logprobs=30,
        )
        """
        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size

        # [NOTE] Stream like the arugment calls
        if use_logits:
            max_new_tokens = 2
            min_new_tokens = 2
            assert current_window_size <= 9, "using logits with numerical ordering can only supports window size <= 9"
            params = None
            outputs = self._llm.generate(prompt_texts, prob=True)
            arr = [self._get_logits_single_digit_batched(output, use_alpha=use_alpha) for output in outputs]
            return [(s, len(s)) for s, __ in arr]
        else:
            params = None
            outputs = self._llm.generate(prompt_texts , prob=False)
            return [(output, 0) for output in outputs]
            # return [
            #     (output.outputs[0].text, len(output.outputs[0].token_ids))
            #     for output in outputs
            # ]

    def _replace_number(self, s: str, use_alpha) -> str:
        if use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", s)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", s)

    # def num_output_tokens(self, use_alpha: bool, current_window_size: Optional[int] = None) -> int:
    #     if current_window_size is None:
    #         current_window_size = self._window_size
    #
    #     if self._output_token_estimate and self._window_size == current_window_size:
    #         return self._output_token_estimate
    #
    #     if use_alpha:
    #         token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
    #     else:
    #         token_str = " > ".join([f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)])
    #
    #     _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1
    #
    #     if self._window_size == current_window_size:
    #         self._output_token_estimate = _output_token_estimate
    #
    #     return _output_token_estimate

    # def permutation_pipeline(
    #     self,
    #     result: Result,
    #     use_logits: bool,
    #     use_alpha: bool,
    #     rank_start: int,
    #     rank_end: int,
    #     logging: bool = False,
    # ) -> Result:
    #     """
    #     Runs the permutation pipeline on the passed in result set within the passed in rank range.
    #
    #     Args:
    #         result (Result): The result object to process.
    #         rank_start (int): The start index for ranking.
    #         rank_end (int): The end index for ranking.
    #         logging (bool, optional): Flag to enable logging of operations. Defaults to False.
    #
    #     Returns:
    #         Result: The processed result object after applying permutation.
    #     """
    #     prompt, in_token_count = self.create_prompt(result, use_alpha, rank_start, rank_end)
    #     if logging:
    #         print(f"prompt: {prompt}\n")
    #     permutation, out_token_count = self.run_llm(
    #         prompt, use_logits=use_logits, use_alpha=use_alpha, current_window_size=rank_end - rank_start
    #     )
    #     if logging:
    #         print(f"output: {permutation}")
    #     ranking_exec_info = RankingExecInfo(
    #         prompt, permutation, in_token_count, out_token_count
    #     )
    #     if result.ranking_exec_summary == None:
    #         result.ranking_exec_summary = []
    #     result.ranking_exec_summary.append(ranking_exec_info)
    #     result = self.receive_permutation(result, permutation, rank_start, rank_end, use_alpha)
    #     return result

    # def sliding_windows(
    #     self,
    #     retrieved_result: Result,
    #     use_logits: bool,
    #     use_alpha: bool,
    #     rank_start: int,
    #     rank_end: int,
    #     window_size: int,
    #     step: int,
    #     logging: bool = False,
    # ) -> Result:
    #     """
    #     Applies the sliding window algorithm to the reranking process.
    #
    #     Args:
    #         retrieved_result (Result): The result object to process.
    #         rank_start (int): The start index for ranking.
    #         rank_end (int): The end index for ranking.
    #         window_size (int): The size of each sliding window.
    #         step (int): The step size for moving the window.
    #         logging (bool, optional): Flag to enable logging of operations. Defaults to False.
    #
    #     Returns:
    #         Result: The result object after applying the sliding window technique.
    #     """
    #     rerank_result = copy.deepcopy(retrieved_result)
    #     end_pos = rank_end
    #     start_pos = rank_end - window_size
    #     # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
    #     # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
    #     while end_pos > rank_start and start_pos + step != rank_start:
    #         start_pos = max(start_pos, rank_start)
    #         rerank_result = self.permutation_pipeline(
    #             rerank_result, use_logits, use_alpha, start_pos, end_pos, logging
    #         )
    #         end_pos = end_pos - step
    #         start_pos = start_pos - step
    #     return rerank_result
    #

    # def receive_permutation(
    #     self, result: Result, permutation: str, rank_start: int, rank_end: int, use_alpha: bool
    # ) -> Result:
    #     """
    #     Processes and applies a permutation to the ranking results.
    #
    #     This function takes a permutation string, representing the new order of items,
    #     and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
    #     'result' object based on this permutation.
    #
    #     Args:
    #         result (Result): The result object containing the initial ranking results.
    #         permutation (str): A string representing the new order of items.
    #                         Each item in the string should correspond to a rank in the results.
    #         rank_start (int): The starting index of the range in the results to which the permutation is applied.
    #         rank_end (int): The ending index of the range in the results to which the permutation is applied.
    #
    #     Returns:
    #         Result: The updated result object with the new ranking order applied.
    #
    #     Note:
    #         This function assumes that the permutation string is a sequence of integers separated by spaces.
    #         Each integer in the permutation string corresponds to a 1-based index in the ranking results.
    #         The function first normalizes these to 0-based indices, removes duplicates, and then reorders
    #         the items in the specified range of the 'result.hits' list according to the permutation.
    #         Items not mentioned in the permutation string remain in their original sequence but are moved after
    #         the permuted items.
    #     """
    #     response = self._clean_response(permutation, use_alpha)
    #     response = [int(x) - 1 for x in response.split()]
    #     response = self._remove_duplicate(response)
    #     cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
    #     original_rank = [tt for tt in range(len(cut_range))]
    #     response = [ss for ss in response if ss in original_rank]
    #     response = response + [tt for tt in original_rank if tt not in response] 
    #     # assign the rank to the unappeared document (assuming they are irrelevant)
    #     for j, x in enumerate(response):
    #         result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
    #         if "rank" in result.hits[j + rank_start]:
    #             result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
    #         if "score" in result.hits[j + rank_start]:
    #             result.hits[j + rank_start]["score"] = cut_range[j]["score"]
    #     return result

