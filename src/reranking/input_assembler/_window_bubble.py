"""
end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
"""
import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import RerankMode, Result
from ..prompt_builder import PromptBuilder
from ..result_parser import ResultParser
from .base import RerankStrategy

class WindowBubble(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:
        r"""Given a list of result files, return a list of reranked results.
        Args:
            init_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
        """
        rerank_results = [copy.deepcopy(result) for result in init_results]

        end_pos = rank_end
        start_pos = rank_end - self._window_size

        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.run_pass(rerank_results, start_pos, end_pos)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size
        return rerank_results
