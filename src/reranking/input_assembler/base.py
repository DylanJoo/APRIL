from abc import ABC, abstractmethod
from typing import Optional, List, Any
import time

from ..utils import Result
from ..prompt_builder import PromptBuilder
from ..result_parser import ResultParser

class RerankStrategy(ABC):
    def __init__(
        self,
        config,
        prompt_builder: PromptBuilder,
        llm_provider: Any,
        result_parser: ResultParser,
    ):
        self.config = config
        self._prompt_builder = prompt_builder
        self._llm = llm_provider
        self._result_parser = result_parser

        self._window_size = self.config.window_size
        self._step_size = self.config.step_size

    @abstractmethod
    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:
        """
        Run the full reranking process.
        Strategy-specific and may depend on use_logits or other config flags.
        """
        pass

    @abstractmethod
    def run_pass(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 8,
    ) -> List[Result]:
        """
        Run a single pass of reranking.
        This method is generally shared across strategies, but can be overridden.
        """
        pass
