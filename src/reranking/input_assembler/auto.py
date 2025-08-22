from .list_bubble import SlidingWindow, SlidingWindowFIRST
from .list_bubble import SlidingWindowPlus
from .pair_all import PairAll
from .pair_bubble_topk import PairBubbleTopK
from .pair_maxheap_topk import PairMaxHeapTopK
from .set_bubble_topk import SetBubbleTopK
from .set_maxheap_topk import SetMaxHeapTopK

# Developement
from ._april import April
from ._dnc import DnC
from ._dev import Dev
# from .set_tournament import SetTouranment

class AutoAssembler:
    _builder_map = {
        'RankZephyr': SlidingWindow,
        'RankGPT': SlidingWindow,
        'RankGPT+': SlidingWindowPlus,
        'RankFirst': SlidingWindowFIRST,
        'PariAll': PairAll,
        'PairTopK': PairBubbleTopK,
        'PairMaxHeapTopK': PairMaxHeapTopK,
        'SetTopK': SetBubbleTopK,
        'SetMaxHeapTopK': SetMaxHeapTopK,
        'Dev': Dev,
        'DnC': DnC,
        'April': April,
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
