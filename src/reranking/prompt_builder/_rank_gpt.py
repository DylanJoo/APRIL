from typing import List, Optional, Union, Callable, Dict, Tuple

class RankGPTFormatter:
    r""" A formatter for the RankGPT prompt mode with textual inputs (instead of Result objects)
    Attributes: TBD

    Args:
        query (str): The search query.
        doc_list (Optional[List[str]]): List of documents to be included in the prompt.
        doc1 (Optional[Union[int, str]]): Identifier for the first document.
        doc2 (Optional[Union[int, str]]): Identifier for the second document.
    """
    def __init__(
        self, 
        use_alpha=False, 
        variable_passages=False,
    ):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if not variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if not variable_passages else "[4] > [2]"

        self.max_doc_length = 1024

    def _document_format(self, doc: Union[Dict, str]) -> str:
        if isinstance(doc, dict):
            if 'content' in doc:
                content = doc['content'].replace("Title: Content: ", "").strip()
            else:
                raise ValueError("Incorrect document dictionary format. ")
        elif isinstance(doc, str):
            content = doc.strip()
        else:
            raise ValueError("Document must be a string or a dictionary with 'content' key.")

        return " ".join(content.split()[:self.max_doc_length])  

    def prefix(self, query: str, doc_list: Optional[List[str]] = None, **kwargs) -> str:
        return (
            f"I will provide you with {len(doc_list)} passages, "
            f"each indicated by a {self.id_type} identifier []. "
            f"Rank the passages based on their relevance to the search query: {query}.\n"
        )

    def postfix(self, query: str, doc_list: Optional[List[str]] = None, **kwargs) -> str:
        return (
            f"Search Query: {kwargs.get('query', '')}.\n"
            f"Rank the passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, "
            f"in descending order of relevance. The output format should be [] > [], "
            f"e.g., {self.example_ordering}, "
            f"Only respond with the ranking results, do not say any word or explain."
        )

    def body(self, query: str, doc_list: Optional[List[str]], **kwargs) -> str:
        prompt_body = ""
        for i, doc in enumerate(doc_list, start=1): # chr(65) is 'A'
            identifier = f"[{chr(64 + i)}]" if self._use_alpha else f"[{i}]"
            prompt_body += f"{identifier} {doc}\n"
        return prompt_body

