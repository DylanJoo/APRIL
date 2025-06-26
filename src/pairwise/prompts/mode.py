from typing import List, Optional, Union, Callable, Dict, Tuple
from enum import Enum
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class PromptMode(Enum):
    RANK_GPT = "rank_GPT"
    LRL = "LRL"
    APRIL = "APRIL"

    def __str__(self):
        return self.value

class Result:
    def __init__(
        self,
        qid: str,
        query: str,
        hits = None,
        ranking_exec_summary = None,
    ):
        self.qid = qid
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)

class RankGPTFormatter:
    r""" A formatter for the RankGPT prompt mode with textual inputs (instead of Result objects)
    Attributes: TBD
    Args:
        query (str): The search query.
        doc_list (Optional[List[str]]): List of documents to be included in the prompt.
        doc1 (Optional[Union[int, str]]): Identifier for the first document.
        doc2 (Optional[Union[int, str]]): Identifier for the second document.
    """
    def __init__(self, use_alpha=False, variable_passages=False):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if not variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if not variable_passages else "[4] > [2]"

        self.max_doc_lenth = 1024

    def _document_format(self, doc: Union[Dict, str]) -> str:
        if isinstance(doc, dict):
            if 'content' in doc:
                content = doc['content'].replace("Title: Content: ", "").strip()
            else:
                raise ValueError("Incorrect document dictionary format. ")
        elif isinstance(doc, str):
            content = doc.strip()
        else:
            raise ValueError("Document must be a string or a dictionary with 'content' key.")

        return " ".join(content.split()[:self.max_doc_lenth])  

    def prefix(self, query: str, doc_list: Optional[List[str]] = None, **kwargs) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages, "
            f"each indicated by a {self.id_type} identifier []. "
            f"Rank the passages based on their relevance to the search query: {query}.\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[str]] = None, **kwargs) -> str:
        return (
            f"Search Query: {kwargs.get('query', '')}.\n"
            f"Rank the passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, "
            f"in descending order of relevance. The output format should be [] > [], "
            f"e.g., {self.example_ordering}, "
            f"Only respond with the ranking results, do not say any word or explain."
        )

    def body(self, query: str, doc_list: Optional[List[str]], **kwargs) -> str:
        prompt_body = ""
        for i, doc in enumerate(doc_list, start=1): # chr(65) is 'A'
            identifier = f"[{chr(64 + i)}]" if self._use_alpha else f"[{i}]"
            prompt_body += f"{identifier} {doc}\n"
        return prompt_body

class PromptFormatter:

    def __init__(
        self, 
        model_name_or_path: str,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        **kwargs
    ):
        self.prompt_mode = prompt_mode
        self.formatter = self._get_formatter(prompt_mode)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if model_name_or_path else None
        self.system_message_supported = "system" in self._tokenizer.chat_template

    def get_num_tokens(self, prompt: str) -> int:
        if self._tokenizer is None: # switch to use the word tokenizer
            raise ValueError("Tokenizer is not initialized.")

        return self._tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    def _get_formatter(self, prompt_mode: PromptMode) -> Callable:
        formatter_map: Dict[PromptMode, Callable] = {
            PromptMode.RANK_GPT: RankGPTFormatter,
        }
        if prompt_mode not in formatter_map:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        return formatter_map[prompt_mode]()

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        r"""
        Args:
            results (List[Result]): List of search results to be processed.
            rank_start (int): The starting rank for the results.
            rank_end (int): The ending rank for the results.
        Returns:
            (List, List): A tuple containing two lists
        """
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]


        all_completed_prompts = []
        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end), 
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)

        return map(list, zip(*all_completed_prompts))

    def create_prompt(
        self, 
        result: Result,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
        ) -> str:

        messages = []

        # system message
        messages.append(
            {"role": "system", 
             "content": f"You are a helpful assistant for the {self.prompt_mode} task."}
        )

        # user message
        ## collect text input data
        query = result.query
        doc_list = [hit['content'] for hit in result.hits[rank_start:rank_end]]
        # [NOTE] doc1 and doc2 are not used in this mode, but kept for compatibility

        prefix = self.formatter.prefix(query=query, doc_list=doc_list)
        postfix = self.formatter.postfix(query=query, doc_list=doc_list)
        body = self.formatter.body(query=query, doc_list=doc_list, max_length=None)
        ## [NOTE] dynamically shrink the body size via length of documents?

        messages.append(
            {"role": "user", 
             "content": prefix + body + postfix}
        )

        # compiling prompt
        prompt = self._tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # maybe calculate different types
        num_tokens = self.get_num_tokens(prompt) 
        return prompt, num_tokens

