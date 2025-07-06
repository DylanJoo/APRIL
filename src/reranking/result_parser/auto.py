from typing import List, Optional, Tuple, Callable, Dict, Union
from ..utils import RerankMode, Result

class ResultParser:
    def __init__(
        self, 
        rerank_mode: RerankMode,
        **kwargs
    ):
        self.rerank_mode = rerank_mode
        self.parser = self._get_parser(rerank_mode, **kwargs)

    def _get_parser(self, rerank_mode: RerankMode, **kwargs) -> Callable:
        parser_map: Dict[RerankMode, Callable] = {
            RerankMode.RANK_GPT: TextListParser,
        }
        if rerank_mode not in parser_map:
            raise ValueError(f"Unsupported prompt mode: {rerank_mode}")
        return parser_map[rerank_mode](**kwargs)
