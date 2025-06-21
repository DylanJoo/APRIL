import copy
import random
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, List, Dict, Union, Any

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from pairwise.rank_llm.utils import RankingExecInfo, Result
from pairwise.rank_llm.messages import (
    _add_prefix_prompt, 
    _add_post_prompt,
    _add_few_shot_examples_messages,
    _add_few_shot_examples,
)
ALPH_START_IDX = ord('A')-1

from llm.litellm_api import LLM
from ftfy import fix_text
from pprint import pprint

class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    LRL = "LRL"
    APRIL = "APRIL"

    def __str__(self):
        return self.value

class RankPairwiseOSLLM:

    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        batched: bool = False,
        rerank_type: str = "text",
    ) -> None:
        # super().__init__(model, context_size, prompt_mode, num_few_shot_examples)

        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples
        self._device = device

        self._llm = LLM(
            model=model, 
            temperature=0.0,
            top_p=1.0,
            logprobs=20,
            max_tokens=20,
        )
        self._tokenizer = self._llm.get_tokenizer()
        true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
        self._llm.set_classification(true_list, false_list)

        self.system_message_supported = "system" in self._tokenizer.chat_template
        self._batched = batched
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        self._rerank_type = rerank_type

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]

    @abstractmethod
    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.
        """
        pass

    def create_prompt_batched(
        self,
        results: List[Result],
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, use_alpha, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def create_prompt(self, result: Result, use_alpha: bool, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query
        num = len(result.hits[rank_start:rank_end])
        # [TODO] control the doc-max-len in the context
        max_length = 1024 # word length
        while True:
            messages = list()
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})
            messages = _add_few_shot_examples_messages(messages)
            prefix = _add_prefix_prompt(use_alpha, query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                # if self._rerank_type == "code": # remove code reranking for simplicity
                content = hit["content"].replace("Title: Content: ", "").strip()
                content = " ".join(content.split()[:max_length])
                identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                input_context += f"[{identifier}] {self._replace_number(content, use_alpha)}\n"
            input_context += _add_post_prompt(use_alpha, query, num)
            messages.append({"role": "user", "content": input_context})
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(rank_end - rank_start):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens - self.max_tokens() + self.num_output_tokens(rank_end - rank_start)
                    ) // ((rank_end - rank_start) * 4),
                )
        return prompt, num_tokens

    
    def permutation_pipeline_batched(
        self,
        results: List[Result],
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range for a batch of results.
        Args:
            results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The processed list of result objects after applying permutation.
        """

    # generate --> 
    # inference_chat --> 
        prompts = []
        # [DONE] 1. change the prompting template
        prompts = self.create_prompt_batched(results, use_alpha, rank_start, rank_end, batch_size=32)
        # [TODO] 2. Maybe change llm inference pipeline.
        batched_results = self.run_llm_batched(
            [prompt for prompt, _ in prompts], 
            use_logits=use_logits, use_alpha=use_alpha, 
            current_window_size=rank_end - rank_start
        )
        #---------------------------------
        for index, (result, (prompt, in_token_count)) in enumerate(zip(results, prompts)):
            permutation, out_token_count = batched_results[index]
            if logging:
                print(f"output: {permutation}")
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            if result.ranking_exec_summary is None:
                result.ranking_exec_summary = []
            result.ranking_exec_summary.append(ranking_exec_info)
            # [TODO] Recieve permulation
            result = self.receive_permutation(result, permutation, rank_start, rank_end, use_alpha)

        print(prompts[0][0])
        with open("debug.txt", "w") as f:
            f.write(f"Prompt: {prompts[0][0]}\n")
            f.write(f"Result: {result}\n")

        return results

    def sliding_windows_batched(
        self,
        retrieved_results: List[Result],
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ) -> List[Result]:
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            retrieved_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        rerank_results = [copy.deepcopy(result) for result in retrieved_results]

        end_pos = rank_end
        start_pos = rank_end - window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.permutation_pipeline_batched(
                rerank_results, use_logits, use_alpha, start_pos, end_pos, logging
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_results

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - window_size) / step + 1
        input_token_count = (
            num_q * num_promt * (self._context_size - self.num_output_tokens())
        )
        output_token_count = num_q * num_promt * self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def get_ranking_cost(
        self,
        retrieved_results: List[Dict[str, Any]],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Dict[str, Any]]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, start_pos, end_pos)
                input_token_count += self.get_num_tokens(prompt)
                end_pos = end_pos - step
                start_pos = start_pos - step
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _clean_response(self, response: str, use_alpha: bool) -> str:
        new_response = ""
        if use_alpha:
            for c in response:
                if not c.isalpha():
                    new_response += " "
                else:
                    new_response += str(ord(c) - ALPH_START_IDX)
            new_response = new_response.strip()
        else:
            for c in response:
                if not c.isdigit():
                    new_response += " "
                else:
                    new_response += c
            new_response = new_response.strip()
            
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, result: Result, permutation: str, rank_start: int, rank_end: int, use_alpha: bool
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.hits' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation, use_alpha)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] 
        # assign the rank to the unappeared document (assuming they are irrelevant)
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        return result

    def _replace_number(self, s: str, use_alpha) -> str:
        if use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", s)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def max_tokens(self) -> int:
        return self._context_size

    def num_output_tokens(self, use_alpha: bool, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        if use_alpha:
            token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        else:
            token_str = " > ".join([f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)])

        _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
        use_logits: bool = False,
        use_alpha: bool = False,
    ) -> List[Tuple[str, int]]:
        """Run batched inference with appropriate restrictions for code vs text reranking"""
        if self._rerank_type == "code" and (use_logits or use_alpha):
            print("Warning: Code reranking does not support logits or alpha mode. Defaulting to standard mode.")
            use_logits = False
            use_alpha = False

        temp = 0.
        if current_window_size is None:
            current_window_size = self._window_size
        if use_logits:
            max_new_tokens = 2
            min_new_tokens = 2
            if use_alpha:
                params = None
                # params = SamplingParams(
                #     min_tokens=min_new_tokens,
                #     max_tokens=max_new_tokens, 
                #     temperature=temp,
                #     logprobs=30,
                # )
            else:
                assert current_window_size <= 9, "using logits with numerical ordering can only supports window size <= 9"
                params = None
                # params = SamplingParams(
                #     min_tokens=min_new_tokens, 
                #     max_tokens=max_new_tokens, 
                #     temperature=temp,
                #     logprobs=30,
                # )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            arr = [self._get_logits_single_digit_batched(output, use_alpha=use_alpha) for output in outputs]
            return [(s, len(s)) for s, __ in arr]
        else:
            params = None
            # params = SamplingParams(
            #     temperature=temp,
            #     max_tokens=self.num_output_tokens(use_alpha, current_window_size),
            #     min_tokens=self.num_output_tokens(use_alpha, current_window_size),
            # )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            return [(output, 0) for output in outputs]
            # return [
            #     (output.outputs[0].text, len(output.outputs[0].token_ids))
            #     for output in outputs
            # ]

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

