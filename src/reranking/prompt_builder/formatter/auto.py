## [NOTE] dynamically shrink the body size via length of documents?
from ...utils import RerankMode
from .listwise import ListwiseFormatter
# from .pairwise_all import PairwiseAllFormatter
# from .pairwise_topk import PairwiseTopKFormatter
# from .pairwise_ref import PairwiseRefFormatter
# from .pairwise_genref import PairwiseGenRefFormatter
from .setwise import SetwiseFormatter
from ._april import AprilFormatter
from ._dev import DevFormatter

from .pairwise import PairwiseFormatter

class AutoPromptFormatter:
    _builder_map = {
        RerankMode.RANK_GPT: ListwiseFormatter,
        RerankMode.PAIRWISE_ALL: PairwiseFormatter,
        RerankMode.PAIRWISE_TOPK: PairwiseFormatter,
        RerankMode.SETWISE_TOPK: SetwiseFormatter,
    }
        # RerankMode.DEV: DevFormatter,
        # RerankMode.APRIL: AprilFormatter,
        # RerankMode.PAIRWISE_REF: PairwiseRefFormatter,
    @classmethod
    def from_config(cls, config=None, rerank_mode=None, **kwargs):
        rerank_mode = rerank_mode or RerankMode(config.rerank_mode)
        builder_cls = cls._builder_map.get(rerank_mode)
        if builder_cls is None:
            raise ValueError(
                f"No prompt builder found for mode: {rerank_mode}\n" 
                f"available modes: {list(cls._builder_map.keys())}"
            )
        return builder_cls(**kwargs)
