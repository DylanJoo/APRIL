""" Make it more flexible to support alpabetical choices """
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class PairwiseFormatter(BaseFormatter):

    def prefix(self, query, **kwargs) -> str:
        return (
            f"I will provide you with two passages. Read and memorize both carefully. "
            f"Your task is to determine which passage is more relevant to the query: {query}\n\n"
        )

    def postfix(self, **kwargs) -> str:
        return (
            "Search Query: {query}.\n"
            "Based on the search query, is the Passage 1 more relevant than the Passage 2?\n"
            "Only respond with Yes or No, do not explain.\nAnswer: "
        )

    def body(self, query, doc_list, idx_pairs=None, **kwargs) -> str:
        template = "Passage 1: {doc1}\nPassage 2: {doc2}\n\n"
        doc_list = [self._document_format(doc) for doc in doc_list]

        if idx_pairs is None: # all indices
            idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]

        prompts = []
        for i, j in idx_pairs:
            prompt = template.format(query=query, doc1=doc_list[i], doc2=doc_list[j])
            prompts.append(prompt)

        return prompts[0] if len(idx_pairs) == 1 else prompts 

    """
    For PairAll, create the prompt for each pair of document in query-batch
    """

# class PairwiseFormatter(BaseFormatter):
#
#     def prefix(self, query, **kwargs) -> str:
#         return (
#             f"I will provide you with two passages. Read and memorize both carefully. "
#             f"Your task is to determine which passage is more relevant to the query: {query}\n\n"
#         )
#
#     def postfix(self, **kwargs) -> str:
#         return (
#             "Based on the query, is the Passage [1] more relevant than Passage [2]?\n"
#             "Only respond with Yes or No, do not explain.\nAnswer: "
#         )
#
#     def body(self, query, doc_list, idx_pairs=None, **kwargs) -> str:
#         template = "Passages\n[1] {doc1}\n[2] {doc2}\nQuery: {query}\n\n"
#         doc_list = [self._document_format(doc) for doc in doc_list]
#
#         if idx_pairs is None: # all indices
#             idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]
#
#         prompts = []
#         for i, j in idx_pairs:
#             prompt = template.format(query=query, doc1=doc_list[i], doc2=doc_list[j])
#             prompts.append(prompt)
#
#         return prompts[0] if len(idx_pairs) == 1 else prompts 
