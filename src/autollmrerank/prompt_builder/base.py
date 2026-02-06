import json
from typing import List, Optional, Union, Callable, Dict, Tuple
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ftfy import fix_text

from ..utils import Result, batch_iterator
from .formatter.auto import AutoPromptFormatter
import pdb

class PromptBuilder:
    def __init__(self, config):
        self.formatter = AutoPromptFormatter.from_config(config)
        self._tokenizer = AutoTokenizer.from_pretrained(config.llm.model_name_or_path)
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self.system_message = config.system_message

    def get_num_tokens(self, prompt: Union[List[str], str]) -> int:
        if isinstance(prompt, list):
            return [self._tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompt]
        return self._tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    # remove the kwargs later
    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: int = 64,
        **kwargs
    ) -> List[Tuple[str, int]]:

        all_completed_prompts = []
        with ThreadPoolExecutor() as executor:
            for batch in batch_iterator(results, batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end, **kwargs), 
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    # NOTE: move reverse as kwargs as it is not commonly used. or remove becuase we already have reverse
    def create_prompt(
        self, 
        result: Result,
        rank_start: int = 0,
        rank_end: int = None,
        idx_pairs: Optional[List[Tuple[int, int]]] = None,
        **kwargs
    ) -> Union[Tuple[str, int], List[Tuple[str, int]]]:
        """
        Only consider the result in the range of [rank_start, rank_end].
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
        query = result.query
        doc_list = [hit['content_dict'] for hit in result.hits[rank_start:rank_end]]
        inputs = {
            "query": query,
            "doc_list": doc_list,
            "rank_start": rank_start,
            "rank_end": rank_end,
            "idx_pairs": idx_pairs,
        }
        prefix = self.formatter.prefix(**inputs, **kwargs)
        postfix = self.formatter.postfix(**inputs, **kwargs)
        body = self.formatter.body(**inputs, **kwargs)

        # organize the prompts with reranking methods 
        # NOTE Sacrifice consistency for simplicity
        # Case1: postfix and body are single string --> listwise method
        if isinstance(postfix, str) and isinstance(body, str): 
            prompt, token_count = self._convert_message_to_prompt(messages, prefix, body, postfix)
            return prompt
        # Case2: body is a list of string --> pointwise method
        elif isinstance(body, list) and isinstance(postfix, str):
            prefix = [prefix] * len(body)
            postfix = [postfix] * len(body)
        # Case2: postfix is a list of string --> prompt caching (dev)
        elif isinstance(postfix, list) and isinstance(body, str):
            prefix = [prefix] * len(postfix)
            body = [body] * len(postfix)
        else:
            raise ValueError(f"Incorrect input types for prefix, body, or postfix, \
                    got: {type(prefix)}, {type(body)}, {type(postfix)}")

        outputs = [
            self._convert_message_to_prompt(messages, pre, b, post)
            for pre, b, post in zip(prefix, body, postfix)
        ]
        prompts, token_counts = zip(*outputs)
        return list(prompts)

    def _convert_message_to_prompt(
        self, 
        message: List[Dict[str, str]], 
        prefix: str, 
        body: str,
        postfix: str
    ) -> Union[Tuple[str, str], Tuple[List, List]]:

        if self.system_message_supported:
            message_ = message.copy()
            message_[1]['content'] = prefix + body + postfix
            prompt = self._tokenizer.apply_chat_template(
                message_,
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = prefix + body + postfix

        prompt = fix_text(prompt)
        num_tokens = self.get_num_tokens(prompt) 
        return prompt, num_tokens

    def __str__(self) -> str:
        inputs = {
            "query": "{QUERY}", 
            "doc_list": [
                {"title": "{TITLE1}", "contents": "{DOCUMENT1}"}, 
                {"title": "{TITLE2}", "contents": "{DOCUMENT2}"}
            ]
        }
        prefix = self.formatter.prefix(**inputs)
        body = self.formatter.body(**inputs)
        body = body if isinstance(body, str) else body[0]
        postfix = self.formatter.postfix(**inputs)
        postfix = postfix if isinstance(postfix, str) else postfix[0]
        return prefix + body + postfix

# [NOTE] consider this if flatten the prompting 
# this is for compatibility the list of list
# # For the scenario that the output is a list of tuples
# if isinstance(completed_prompts[0], list):
# completed_prompts = [item for sublist in completed_prompts for item in sublist]
