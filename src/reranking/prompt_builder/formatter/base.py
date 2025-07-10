import re
from typing import List, Optional, Union, Callable, Dict, Tuple
from abc import ABC, abstractmethod

class BaseFormatter(ABC):
    """Base class for all formatters."""

    def __init__(
        self, 
        use_alpha=False, 
        variable_passages=False,
        max_doc_length=1024
    ):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if variable_passages else "[4] > [2]"

        self.max_doc_length = max_doc_length

    @abstractmethod
    def prefix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        """Returns the prefix of the prompt."""
        pass

    @abstractmethod
    def postfix(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        """Returns the postfix of the prompt."""
        pass

    @abstractmethod
    def body(self, query: str, doc_list: Optional[List[Dict]] = None, **kwargs) -> str:
        """Returns the body of the prompt."""
        pass

    # Unified preprocessing function for documents
    # [TODO] Equalize the max length
    def _document_format(self, doc: Union[str, Dict]) -> str:
        if isinstance(doc, dict):
            title = doc.get('title', False)
            if 'contents' in doc:
                text = doc['contents'].strip()
            else:
                raise ValueError(f"Incorrect document dictionary format. Expected keys: 'title', 'contents': got {doc}")
        elif isinstance(doc, str):
            text = doc.strip()
        else:
            raise ValueError(f"Document must be a string or a dictionary with 'content' key: got {doc}")

        return " ".join(text.split()[:self.max_doc_length])  

    def replace_number(self, text: str) -> str:
        if self._use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", text)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", text)
