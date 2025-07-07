from typing import List, Optional, Tuple, Callable, Dict, Union
from ..utils import RerankMode, Result
from ..text_list import TextListParser
from ..scores import ScoreParser

class ResultParser:
    _builder_map = {
        RerankMode.RANK_GPT: TextListParser,
        RerankMode.PAIRWISE: ScoreParser,
    }
    @classmethod
    def from_config(cls, config=None, rerank_mode=None, **kwargs):
        rerank_mode = rerank_mode or RerankMode(config.rerank_mode)
        builder_cls = cls._builder_map.get(rerank_mode)
        if builder_cls is None:
            raise ValueError(
                f"No result parser found for mode: {rerank_mode}\n" 
                f"available modes: {list(cls._builder_map.keys())}"
            )
        return builder_cls(**kwargs)
