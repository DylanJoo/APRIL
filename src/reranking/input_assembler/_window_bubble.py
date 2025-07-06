import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result
from .base import RerankStrategy

class WindowBubble(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:

        rerank_results = [copy.deepcopy(result) for result in init_results]

        end_pos = rank_end
        start_pos = rank_end - self._window_size

        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.run_pass(rerank_results, start_pos, end_pos)
            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size
        return rerank_results
