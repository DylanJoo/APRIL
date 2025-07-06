from typing import List, Optional, Tuple, Callable, Dict, Union
from abc import ABC, abstractmethod
from ..utils import RerankMode, Result

class BaseResultParser(ABC):
    """Base class for all parser"""

    def __init__(
        self, 
        use_alpha=False, 
        variable_passages=False,
    ):
        self._use_alpha = use_alpha
        self._variable_passages = variable_passages 

        if use_alpha: 
            self.id_type = "alphabetical"
        else:
            self.id_type = "numerical"

        self.max_doc_length = 1024

    @abstractmethod
    def parse_and_update(
        self, 
        permutation: str, 
        result, 
        rank_start: int, 
        rank_end: int
    ) -> Result:
        """Parse the response and update the result object."""
        pass

    @abstractmethod
    def parse_response(
        self, 
        response_texts: List[str], 
        response_texts: List[str], 
        results: List[Result], 
        rank_start: int, 
        rank_end: int, 
    ) -> str:
        assert len(response_texts) == len(results), "Response texts and results must have the same length."

        for index, (response, result) in enumerate(zip(response_texts, results)):
            parsed_result = self.parser.parse_and_update(response, result, rank_start, rank_end)
            results[index] = parsed_result
        return results

    # [NOTE] parse: response is the input, number as output 
    # [NOTE] update: number to the result object
    # [NOTE] also consider additional info: permutation, out_token_count. Use the origianl for loop
