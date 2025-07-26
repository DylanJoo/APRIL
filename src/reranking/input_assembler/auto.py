# TODO Adapt to utils rerank mode directly.
from ..utils import RerankMode
from .window_bubble import WindowBubble # NOTE: rename it 
from .pair_all import PairAll
from .pair_bubble_topk import PairBubbleTopK
from .set_bubble_topk import SetBubbleTopK
from ._april import April
from ._dev import Dev

# from .pair_quick_topk import PairQuickTopK
# from .set_tournament import SetTouranment
# from .ref_rerank import RefRerank
# from .genref_rerank import GenRefRerank

class AutoAssembler:
    _builder_map = {
        RerankMode.RANK_GPT: WindowBubble,
        RerankMode.PAIRWISE_ALL: PairAll,
        RerankMode.PAIRWISE_TOPK: PairBubbleTopK,
        RerankMode.SETWISE_TOPK: SetBubbleTopK,
    }
        # RerankMode.PAIRWISE_REF: RefRerank,
        # RerankMode.DEV: Dev,
        # RerankMode.PAIRWISE_GENREF: GenRefRerank,
        # RerankMode.APRIL: April,
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
