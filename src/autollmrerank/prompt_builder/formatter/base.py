import re
from typing import List, Optional, Union, Callable, Dict, Tuple
from abc import ABC, abstractmethod

class BaseFormatter(ABC):
    """Base class for all formatters."""

    def __init__(self, config=None):
        self._use_alpha = config.use_alphabetical
        self._variable_passages = config.variable_passages
        self.max_doc_length = config.max_doc_length
        if hasattr(config, 'examples'):
            self.examples = config.examples
        else:
            self.examples = None

        if self._use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"

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

    # @abstractmethod
    # def example(self, **kwargs) -> str:
    #     pass

    # TODO: See if we need to have separate formuation of title
    # TODO: Equalize the max length
    def _document_format(self, doc: Union[str, Dict]) -> str:
        if isinstance(doc, dict):
            if 'contents' in doc:
                text = doc['contents'].strip()
            elif 'text' in doc:
                text = doc['text'].strip()
            else:
                raise ValueError(f"Incorrect document dictionary format. Expected keys: 'contents' or 'text', got {doc.keys()}")
            title = doc.get('title', "")
            text = (title + ' ' + text).strip()
        elif isinstance(doc, str):
            text = doc.strip()
        else:
            raise ValueError(f"Document must be a string or a dictionary with 'content' key: got {doc}")

        if self.max_doc_length is not None:
            return " ".join(text.split(" ")[:self.max_doc_length])  
        else:
            return text

    def replace_number(self, text: str) -> str:
        if self._use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", text)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", text)
