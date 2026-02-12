import re
from typing import List, Optional, Union, Callable, Dict, Tuple
from abc import ABC, abstractmethod

from .example import ExampleFormatter


class BaseFormatter(ABC):
    """Base class for all formatters."""

    # Subclasses should override this to specify their paradigm type
    paradigm = 'base'

    def __init__(self, config=None):
        self._use_alpha = config.use_alphabetical
        self._variable_passages = config.variable_passages
        self.max_doc_length = config.max_doc_length

        if self._use_alpha: 
            self.id_type = "alphabetical"
            self.example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            self.id_type = "numerical"
            self.example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"

        # Initialize example formatter if examples are configured
        # Simplified config: examples is just a list of dicts directly (or None)
        examples_config = getattr(config, 'examples', None)
        
        # Handle both list format and legacy dict format
        if examples_config is None:
            self._example_formatter = ExampleFormatter(examples=None)
        elif isinstance(examples_config, list):
            # New simplified format: examples is a list of dicts directly
            self._example_formatter = ExampleFormatter(
                examples=examples_config,
                max_doc_length=self.max_doc_length
            )
        else:
            # Legacy format: examples is a namespace/dict with 'data' key
            # This provides backward compatibility
            if hasattr(examples_config, 'data'):
                examples_data = examples_config.data
            elif isinstance(examples_config, dict):
                examples_data = examples_config.get('data', None)
            else:
                examples_data = None
            self._example_formatter = ExampleFormatter(
                examples=examples_data,
                max_doc_length=self.max_doc_length
            )

    def examples(self, **kwargs) -> str:
        """
        Returns formatted examples based on the paradigm type.
        
        This method can be called by subclasses to include examples in their prompts.
        The formatting is determined automatically based on the paradigm type.
        """
        return self._example_formatter.format(paradigm=self.paradigm)

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
