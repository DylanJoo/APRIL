## [NOTE] dynamically shrink the body size via length of documents?
from ...utils import RerankMode
from .listwise import ListwiseFormatter
from .pairwise import PairwiseFormatter
from .setwise import SetwiseFormatter
# from .pairwise_all import PairwiseAllFormatter
# from .pairwise_topk import PairwiseTopKFormatter
# from .pairwise_ref import PairwiseRefFormatter
# from .pairwise_genref import PairwiseGenRefFormatter
from ._april import AprilFormatter
from ._dev import DevFormatter
        # RerankMode.RANK_GPT: ListwiseFormatter,
        # RerankMode.PAIRWISE_ALL: PairwiseFormatter,
        # RerankMode.PAIRWISE_TOPK: PairwiseFormatter,
        # RerankMode.SETWISE_TOPK: SetwiseFormatter,

class AutoPromptFormatter:
    _builder_map = {
        'RankGPT': ListwiseFormatter,
        'PairAll': PairwiseFormatter,
        'PairTopK': PairwiseFormatter,
        'SetTopK': SetwiseFormatter,
    }
        # RerankMode.DEV: DevFormatter,
        # RerankMode.APRIL: AprilFormatter,
        # RerankMode.PAIRWISE_REF: PairwiseRefFormatter,
    @classmethod
    def from_config(cls, config):
        rerank_mode = config.rerank_mode
        builder_cls = cls._builder_map.get(rerank_mode)
        if builder_cls is None:
            raise ValueError(
                f"No prompt builder found for mode: {rerank_mode}\n" 
                f"available modes: {list(cls._builder_map.keys())}"
            )
        return builder_cls(config)
