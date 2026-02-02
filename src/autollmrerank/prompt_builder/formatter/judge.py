""" Judge formatter for pointwise reranking with 0-5 rating scale """
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class JudgeFormatter(BaseFormatter):

    def prefix(self, **kwargs) -> str:
        return ""

    def postfix(self, **kwargs) -> str:
        return (
            "Rate the relevance of the passage to the query on a scale from 0 to 5:\n"
            "0: Completely irrelevant\n"
            "1: Slightly relevant\n"
            "2: Moderately relevant\n"
            "3: Relevant\n"
            "4: Highly relevant\n"
            "5: Perfectly relevant\n\n"
            "Only respond with a single number (0-5), do not explain.\nRating: "
        )

    def body(self, query, doc_list, **kwargs) -> str:
        prompts = []
        doc_list = [self._document_format(doc) for doc in doc_list]
        for doc in doc_list:
            prompt = f"Passage: {doc}\nQuery: {query}\n\n"
            prompts.append(prompt)
        return prompts
