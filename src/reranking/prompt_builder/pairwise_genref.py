from typing import List, Optional, Union, Callable, Dict, Tuple
from .formatter_base import BaseFormatter

class PairwiseGenRefFormatter(BaseFormatter):

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

    def body(self, query, doc_list, idx_pairs, reference, **kwargs) -> str:
        template = "Passages\n[1] {doc1}\n[2] {doc2}\nQuery: {query}\n\n"

        doc_list = [self._document_format(doc) for doc in doc_list]

        prompts = []
        for i, j in idx_pairs:
            if j == -1:
                prompt = template.format(query=query, doc1=doc_list[i], doc2=reference)
            if i == -1:
                prompt = template.format(query=query, doc1=reference, doc2=doc_list[j])
            prompts.append(prompt)
        return prompts
