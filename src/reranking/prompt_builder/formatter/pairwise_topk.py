from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class PairwiseTopKFormatter(BaseFormatter):

    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, *kwargs) -> str:
        return (
            f"I will provide you with two passages. Read and memorize both carefully. "
            f"Your task is to determine which passage is more relevant to the query: {query}\n\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return (
            "Based on the query, is the Passage [1] more relevant than Passage [2]?\n"
            "Only respond with Yes or No, do not explain.\nAnswer: "
        )

    def body(self, query: str, doc_list: List[Union[Dict, str]], rank_end: int, reverse: bool = False, **kwargs) -> str:
        template = "Passages\n[1] {doc1}\n[2] {doc2}\nQuery: {query}\n\n"

        doc1 = self._document_format(doc_list[rank_end-1]) 
        doc2 = self._document_format(doc_list[rank_end-2])
        if reverse:
            prompt = template.format(query=query, doc1=doc2, doc2=doc1)
        else:
            prompt = template.format(query=query, doc1=doc1, doc2=doc2)
        return prompt
