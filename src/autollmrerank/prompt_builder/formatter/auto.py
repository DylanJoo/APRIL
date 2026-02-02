## TODO: dynamically shrink the body size via length of documents?
from .listwise import ListwiseFormatter
from .pairwise import PairwiseFormatter
from .setwise import SetwiseFormatter
from .pointwise import PointwiseFormatter
from ._dev import DevFormatter

class AutoPromptFormatter:
    _builder_map = {
        'RankZephyr': ListwiseFormatter,
        'RankGPT': ListwiseFormatter,
        'RankGPT+': ListwiseFormatter,
        'RankFirst': ListwiseFormatter,
        'PairAll': PairwiseFormatter,
        'PairTopK': PairwiseFormatter,
        'PairMaxHeapTopK': PairwiseFormatter,
        'SetTopK': SetwiseFormatter,
        'SetMaxHeapTopK': SetwiseFormatter,
        'Point': PointwiseFormatter,
        'Dev': DevFormatter,
    }
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
