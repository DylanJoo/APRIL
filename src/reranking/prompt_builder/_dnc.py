from typing import List, Optional, Union, Callable, Dict, Tuple
from .formatter_base import BaseFormatter

class DnCFormatter(BaseFormatter):

    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, filtering_postfix=False, **kwargs) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages, "
            f"each indicated by a {self.id_type} identifier []. "
            f"Rank the passages based on their relevance to the search query: {query}.\n\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, filtering_postfix=False, **kwargs) -> str:
        if filtering_postfix:
            return (
                f"Search Query: {query}.\n"
                f"Rank the {len(doc_list)} passages above based on their relevance to the search query. "
                f"All the passages should be included and listed using identifiers, "
                f"in descending order of relevance. "
                "In addition, add a vertical bar in the ranking to indicate the boundary of relevance. " 
                "The output format should be [] > [] > | > [] > []. "
                "The first part (before |) is the ranking for relevant passages; the second part is for irrelevant ones. "
                f"Only respond with the ranking results, do not say any word or explain."
            )
        return (
            f"Search Query: {query}.\n"
            f"Rank the {len(doc_list)} passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, "
            f"in descending order of relevance. The output format should be [] > [], "
            f"e.g., {self.example_ordering}, "
            f"Only respond with the ranking results, do not say any word or explain."
        )

    def body(self, query: str, doc_list: Optional[List[Dict]], **kwargs) -> str:
        prompt_body = ""
        doc_list = [self._document_format(doc) for doc in doc_list]
        for i, doc in enumerate(doc_list, start=1): # chr(65) is 'A'
            identifier = f"[{chr(64 + i)}]" if self._use_alpha else f"[{i}]"
            doc_text = self.replace_number(doc)
            prompt_body += f"{identifier} {doc_text}\n"
        return prompt_body
