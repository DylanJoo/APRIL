""" Make it more flexible to support alpabetical choices """
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class PointwiseFormatter(BaseFormatter):

    paradigm = 'pointwise'

    def prefix(self, **kwargs) -> str:
        examples_text = self.examples()
        return f"{examples_text}"

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
