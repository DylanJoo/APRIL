# Answerability judgments
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class JudgeBinaryFormatter(BaseFormatter):

    def prefix(self, **kwargs) -> str:
        return (
            "Instruction: Determine whether the question can be answered based on the provided context?"
            "Rate the context on a scale from 0 to 5 according to the guideline below. "
            "Do not write anything except the rating.\n\n"
            "Guideline: \n"
            "- 5: The context is highly relevant, complete, and accurate to the question.\n"
            "- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the question.\n"
            "- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the question.\n"
            "- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the question.\n"
            "- 1: The context is minimally relevant or complete, with substantial shortcomings to the question.\n"
            "- 0: The context is not relevant or complete at all.\n\n"
        )

    def postfix(self, **kwargs) -> str:
        return "Rating: \n"

    def body(self, query, doc_list, **kwargs) -> str:
        prompts = []
        doc_list = [self._document_format(doc) for doc in doc_list]
        for doc in doc_list:
            prompt = f"Question: {query}\n\nContext: {doc}\n\n"
            prompts.append(prompt)
        return prompts
