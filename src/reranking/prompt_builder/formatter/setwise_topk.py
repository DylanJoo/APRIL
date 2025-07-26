from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class SetwiseTopKFormatter(BaseFormatter):

    def prefix(self, query, idx_pairs, **kwargs) -> str:
        n_pairs = len(idx_pairs[0])
        return (
            f"I will provide you with {n_pairs} passages. Read and memorize both carefully. "
            f"Your task is to determine which passage is the most relevant to the query: {query}\n\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return (
            "Based on the query, which passage is the most relevant one.\n"
            "Only respond with the passage number in the format of [1], [2], etc. Do not explain."
        )

    def body(self, query: str, doc_list: List[Union[Dict, str]], idx_pairs = None, **kwargs) -> str:
        doc_list = [self._document_format(doc) for doc in doc_list]

        prompts = []
        for idx_pair in idx_pairs:
            prompt_body = ""
            for index, idx in enumerate(idx_pair):
                identifier = f"[{chr(64 + index + 1)}]" if self._use_alpha else f"[{index + 1}]"
                doc_text = self.replace_number(doc_list[idx])
                prompt_body += f"{identifier} {doc_text}\n"

            prompts.append(prompt_body)

        return prompts
