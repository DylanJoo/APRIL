# TODO: use either alphabetical or numerical identifiers based on a flag
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class SetwiseFormatter(BaseFormatter):

    paradigm = 'setwise'

    def prefix(self, query, idx_pairs, **kwargs) -> str:
        n_pairs = len(idx_pairs[0])
        # NOTE: Examples are not supported for setwise paradigm yet
        # as it involves selecting from multiple documents
        return (
            f"I will provide you with {n_pairs} passages. Read and memorize all carefully. "
            f"Your task is to determine which passage is the most relevant to the query: {query}\n\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, idx_pairs = None, **kwargs) -> str:
        return (
            f"Search Query: {query}\nWhich passage is the most relevant one to the query? "
            f"Only respond with the {self.id_type} identifier with bracket (e.g., [A]). Do not say any word or explain."
        )

    def body(self, query: str, doc_list: List[Union[Dict, str]], idx_pairs = None, **kwargs) -> str:
        doc_list = [self._document_format(doc) for doc in doc_list]

        prompt_body = ""
        for order, idx in enumerate(idx_pairs[0], start=1):
            identifier = f"[{chr(64 + order)}]" if self._use_alpha else f"[{order}]"
            doc_text = self.replace_number(doc_list[idx])
            prompt_body += f"{identifier} {doc_text}\n"

        return prompt_body
