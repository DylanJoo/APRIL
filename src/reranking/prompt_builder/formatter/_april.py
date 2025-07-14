from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class AprilFormatter(BaseFormatter):

    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages, "
            f"each indicated by a {self.id_type} identifier []. "
            "Read the passages carefully and remember their content. "
            f"Your task is to use these passages to perform pairwise comparisons, "
            "based on their relevance to the query: {query}.\n\n"
        )

    # [UNUSED] compare the following pair of two passages: {idx_i} and {idx_j}. "
    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        template = (
            "Baesd on the query, is the Passage {idx_i} more relevant than Passage {idx_j}?\n"
            "Only respond with Yes or No, do not exaplain.\nAnswer: "
        )

        idx_pairs = [(i, j) for i in range(len(doc_list)) for j in range(len(doc_list)) if i != j]
        prompts = []
        for i, j in idx_pairs:
            prompt = template.format(
                idx_i=f"[{chr(64 + i + 1)}]" if self._use_alpha else f"[{i + 1}]",
                idx_j=f"[{chr(64 + j + 1)}]" if self._use_alpha else f"[{j + 1}]"
            )
            prompts.append(prompt)
        return prompts

    def body(self, query: str, doc_list: Optional[List[Dict]], **kwargs) -> str:
        prompt_body = "Passages:\n"
        doc_list = [self._document_format(doc) for doc in doc_list]
        for i, doc in enumerate(doc_list, start=1): # chr(65) is 'A'
            identifier = f"[{chr(64 + i)}]" if self._use_alpha else f"[{i}]"
            doc_text = self.replace_number(doc)
            prompt_body += f"{identifier} {doc_text}\n"
        prompt_body += f"\nQuery: {query}\n"
        return prompt_body
