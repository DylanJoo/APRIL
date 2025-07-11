from copy import deepcopy
from enum import Enum

class Result:

    def __init__(
        self,
        qid: str,
        query: str,
        hits = None,
        ranking_exec_summary = None,
    ):
        self.qid = qid
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)

    def sort_by(self, field: str = 'score'):
        hits = deepcopy(self.hits)
        hits.sort(key=lambda x: x[field], reverse=True)
        for i, hit in enumerate(hits):
            hit['rank'] = i + 1
        self.hits = hits
        return hits

class RerankMode(Enum):
    RANK_GPT = "RankGPT"
    PAIRWISE_ALL = "AllPair"
    APRIL = "April"

    @property
    def prompt_builder_name(self):
        return {
            "RankGPT": "listwise", 
            "AllPair": "pairwise",
            "APRIL": "listwise"
        }[self.value]

    @property
    def use_logits(self):
        return {
            "RankGPT": False, 
            "AllPair": True, 
            "April": True, 
        }[self.value]

    @property
    def result_parser_name(self):
        return {
            "RankGPT": "text_list", 
            "AllPair": "prob", 
            "April": "prob", 
        }[self.value]

    def __str__(self):
        return f"""
        [{self.value}]
        - PROMPT_BUILDER FORMATTER: {self.prompt_builder_name}
        - RESULT_PARSER: {self.result_parser_name}
        """

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

