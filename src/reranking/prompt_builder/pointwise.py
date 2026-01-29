""" Make it more flexible to support alpabetical choices """
from typing import List, Optional, Union, Callable, Dict, Tuple
from .formatter_base import BaseFormatter

class PointwiseFormatter(BaseFormatter):

    def prefix(self, **kwargs) -> str:
        return ""

    def postfix(self, **kwargs) -> str:
        return (
            "Is this passage relevant to the query?\n"
            "Only respond with Yes or No, do not explain.\nAnswer: "
        )

    def body(self, query, doc_list, **kwargs) -> str:
        prompts = []
        doc_list = [self._document_format(doc) for doc in doc_list]
        for doc in doc_list:
            prompt = f"Passage: {doc}\nQuery: {query}\n\n"
            prompts.append(prompt)
        return prompts
