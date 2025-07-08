import json
from typing import List, Optional, Union, Callable, Dict, Tuple
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ftfy import fix_text

from ..utils import RerankMode, Result, batch_iterator
from .formatter.auto import AutoPromptFormatter

class PromptBuilder:
    def __init__(
        self, 
        config,
        include_system_message: Optional[bool] = None,
        system_message: Optional[str] = None,
        **kwargs
    ):
        self.formatter = AutoPromptFormatter.from_config(config, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(config.llm.model_name_or_path)
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self.system_message = system_message

    def get_num_tokens(self, prompt: Union[List[str], str]) -> int:
        if isinstance(prompt, list):
            return [self._tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompt]
        return self._tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:

        all_completed_prompts = []
        with ThreadPoolExecutor() as executor:
            for batch in tqdm(batch_iterator(results, batch_size), desc="Creating prompts"):
                # list of tuples: # [ [prompt set 1], [prompt set 2], ... ]
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end), 
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def create_prompt(
        self, 
        result: Result,
        rank_start: int,
        rank_end: int,
    ) -> Union[Tuple[str, int], List[Tuple[str, int]]]:
        r"""batch processing of results to create prompts using multithreading.
        Returns:
            (str, str): A tuple containing two strings
        """
        # system message (if applicable)
        if self.system_message_supported and self.system_message:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": None}
            ]
        else:
            messages = [{"role": "user", "content": None}]

        # user message
        # [NOTE] doc1 and doc2 are not used in this mode, but kept for compatibility
        query = result.query
        doc_list = [hit['content_dict'] for hit in result.hits[rank_start:rank_end]]

        prefix = self.formatter.prefix(query=query, doc_list=doc_list)
        postfix = self.formatter.postfix(query=query, doc_list=doc_list)
        body = self.formatter.body(query=query, doc_list=doc_list, max_length=None)

        if isinstance(postfix, str) and isinstance(body, str):
            # [NOTE] Sacrifice consistency for simplicity
            # prefix, body, postfix = [prefix], [body], [postfix]
            prompt, token_count = self._convert_message_to_prompt(messages, prefix, body, postfix)
            return prompt
        elif isinstance(body, list) and isinstance(postfix, str):
            prefix = [prefix] * len(body)
            postfix = [postfix] * len(body)
        elif isinstance(postfix, list) and isinstance(body, str):
            prefix = [prefix] * len(postfix)
            body = [body] * len(postfix)
        else:
            raise ValueError(f"Incorrect input types for prefix, body, or postfix, got: {type(prefix)}, {type(body)}, {type(postfix)}")

        outputs = [
            self._convert_message_to_prompt(messages, pre, b, post)
            for pre, b, post in zip(prefix, body, postfix)
        ]
        prompts, token_counts = zip(*outputs)
        return list(prompts)

    def _convert_message_to_prompt(
        self, 
        messages: List[Dict[str, str]], 
        prefix: str, 
        body: str,
        postfix: str
    ) -> Union[Tuple[str, str], Tuple[List, List]]:

        if self.system_message_supported:
            messages_ = messages.copy()
            messages_[1]['content'] = prefix + body + postfix
            prompt = self._tokenizer.apply_chat_template(
                messages_,
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = prefix + body + postfix
        prompt = fix_text(prompt)

        num_tokens = self.get_num_tokens(prompt) 
        return prompt, num_tokens

    def __str__(self) -> str:
        return self.formatter.prefix("") + self.formatter.body("", []) + self.formatter.postfix("")

# [NOTE] consider this if flatten the prompting 
# this is for compatibility the list of list
# # For the scenario that the output is a list of tuples
# if isinstance(completed_prompts[0], list):
# completed_prompts = [item for sublist in completed_prompts for item in sublist]

