## [NOTE] dynamically shrink the body size via length of documents?
from ..utils import RerankMode
from .window_bubble import WindowBubble
from .pair_all import PairAll
from ._pair_bubble_topk import PairBubbleTopK

class AutoAssembler:
    _builder_map = {
        RerankMode.RANK_GPT: WindowBubble,
        RerankMode.PAIRWISE_ALL: PairAll,
        RerankMode.PAIRWISE_TOPK: PairBubbleTopK,
    }
    @classmethod
    def from_config(cls, config=None, rerank_mode=None, **kwargs):
        rerank_mode = rerank_mode or RerankMode(config.rerank_mode)
        builder_cls = cls._builder_map.get(rerank_mode)
        if builder_cls is None:
            raise ValueError(
                f"No prompt builder found for mode: {rerank_mode}\n" 
                f"available modes: {list(cls._builder_map.keys())}"
            )
        return builder_cls(config=config, rerank_mode=rerank_mode, **kwargs)
