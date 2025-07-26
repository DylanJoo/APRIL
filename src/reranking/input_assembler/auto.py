# TODO Adapt to utils rerank mode directly.
from ..utils import RerankMode
from .list_bubble import SlidingWindow
from .pair_all import PairAll
from .pair_bubble_topk import PairBubbleTopK
from .set_bubble_topk import SetBubbleTopK
# from .pair_quick_topk import PairQuickTopK
# from .set_tournament import SetTouranment
# from .ref_rerank import RefRerank
# from .genref_rerank import GenRefRerank
from ._april import April
from ._dev import Dev

class AutoAssembler:
    _builder_map = {
        'RankGPT': SlidingWindow,
        'PariAll': PairAll,
        'PairTopK': PairBubbleTopK,
        'SetTopK': SetBubbleTopK,
    }

    @classmethod
    def from_config(cls, config, **kwargs):
        rerank_mode = config.rerank_mode
        builder_cls = cls._builder_map.get(rerank_mode)
        if builder_cls is None:
            raise ValueError(
                f"No prompt builder found for mode: {rerank_mode}\n" 
                f"available modes: {list(cls._builder_map.keys())}"
            )
        return builder_cls(config=config, **kwargs)
