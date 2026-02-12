from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

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
            # Skip if indices are out of bounds (except -1 which means use reference)
            # Also skip invalid negative indices (other than -1)
            if i != -1 and (i < 0 or i >= len(doc_list)):
                continue
            if j != -1 and (j < 0 or j >= len(doc_list)):
                continue
            if j == -1:
                prompt = template.format(query=query, doc1=doc_list[i], doc2=reference)
            elif i == -1:
                prompt = template.format(query=query, doc1=reference, doc2=doc_list[j])
            else:
                # Both i and j are valid doc indices
                prompt = template.format(query=query, doc1=doc_list[i], doc2=doc_list[j])
            prompts.append(prompt)
        return prompts
