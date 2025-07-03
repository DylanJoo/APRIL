import json
from typing import List, Optional, Union, Callable, Dict, Tuple
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils import RerankMode, Result
from ._rank_gpt import RankGPTFormatter

class PromptBuilder:
    def __init__(
        self, 
        model_name_or_path: str,
        rerank_mode: RerankMode = RerankMode.RANK_GPT,
        include_system_message: Optional[bool] = None,
        system_message: Optional[str] = None,
        **kwargs
    ):
        self.rerank_mode = rerank_mode
        self.formatter = self._get_formatter(rerank_mode, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if model_name_or_path else None
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self.system_message = system_message

    def get_num_tokens(self, prompt: str) -> int:
        if self._tokenizer is None: # switch to use the word tokenizer
            raise ValueError("Tokenizer is not initialized.")

        return self._tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    def _get_formatter(self, rerank_mode: RerankMode, **kwargs) -> Callable:
        formatter_map: Dict[RerankMode, Callable] = {
            RerankMode.RANK_GPT: RankGPTFormatter,
        }
        if rerank_mode not in formatter_map:
            raise ValueError(f"Unsupported prompt mode: {rerank_mode}")
        return formatter_map[rerank_mode](**kwargs)

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        r"""batch processing of results to create prompts using multithreading.

        Returns:
            (List, List): A tuple containing two lists
        """
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []
        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Creating prompts"):
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
        batch_size: int = 32,
        ) -> str:
        r"""batch processing of results to create prompts using multithreading.

        Returns:
            (str, str): A tuple containing two strings
        [TODO] adding length truncation based on the `context_size` parameter
        """
        # system message (if applicable)
        if self.system_message_supported and self.system_message:
            messages = [{"role": "system", "content": self.system_message}]
        else:
            messages = []

        # user message
        ## collect text input data
        query = result.query
        # [NOTE] doc1 and doc2 are not used in this mode, but kept for compatibility
        doc_list = [hit['content'] for hit in result.hits[rank_start:rank_end]]

        prefix = self.formatter.prefix(query=query, doc_list=doc_list)
        postfix = self.formatter.postfix(query=query, doc_list=doc_list)
        body = self.formatter.body(query=query, doc_list=doc_list, max_length=None)
        ## [NOTE] dynamically shrink the body size via length of documents?


        if self.system_message_supported:
            messages.append({"role": "user", "content": prefix + body + postfix})
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = prefix + body + postfix

        # maybe calculate different types
        num_tokens = self.get_num_tokens(prompt) 
        return prompt, num_tokens

