from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class PairwiseAllFormatter(BaseFormatter):

    def prefix(self, query: str, **kwargs, doc_list: Optional[List[Dict]] = None) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages. Read and memorize them carefully. "
            f"Your task is to determine which passage is more relevant to the query.\n\n"
            f"Query: {query}\n\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return "Please answer 'Yes' or 'No'.\nAnswer: "

    # def body(self, query: str, doc1: str, doc2: str, **kwargs) -> str:
    def body(self, query: str, doc_list: List[Union[Dict, str]], **kwargs) -> str:
        template = (
            "Passage 1: {doc1}\n"
            "Passage 2: {doc2}\n\n"
            "Based on the given query, is Passage 1 more relevant than Passage 2? "
        )

        doc_list = [self._document_format(doc) for doc in doc_list]
        prompt_body = ""
        idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]
        prompts = []
        for i, j in idx_pairs:
            prompt = template.format(query=query, doc1=doc_list[i], doc2=doc_list[j])
            prompts.append(prompt)
        return prompts
