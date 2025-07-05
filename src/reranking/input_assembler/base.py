from abc import ABC, abstractmethod
from typing import Optional, List, Any

from ..utils import RerankMode, Result
from ..prompt_builder import PromptBuilder
from ..result_parser import ResultParser


class RerankStrategy(ABC):
    def __init__(
        self,
        config,
        rerank_mode: RerankMode,
        prompt_builder: PromptBuilder,
        llm_provider: Any,
        result_parser: ResultParser,
    ):
        self.config = config
        self._prompt_builder = prompt_builder
        self._llm = llm_provider
        self._result_parser = result_parser

        self._rerank_mode = rerank_mode
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

    ## [TODO] maybe we need to set the batch size if one query requires huge amount of prompts/inference. 
    # [NOTE] window size for using logits might have limited to 9, this is not used for now
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
        prompts = self._prompt_builder.create_prompt_batched(results, rank_start, rank_end)
        responses = self._llm.generate(
            prompts=[prompt for prompt, _ in prompts],
            prob=self._rerank_mode.use_logits
        )

        assert len(responses) == len(prompts), "Mismatch between prompts and responses"

        reranked_results = self._result_parser.parse_response(
            response_texts=responses,
            results=results,
            rank_start=rank_start,
            rank_end=rank_end,
        )
        return reranked_results

