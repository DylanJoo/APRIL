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

class RerankMode(Enum):
    RANK_GPT = "RankGPT"
    PAIRWISE = "Pairwise"

    @property
    def prompt_builder_name(self):
        return {
            "RankGPT": "listwise", 
            "Pairwise": "pairwise"
        }[self.value]

    @property
    def use_logits(self):
        return {
            "RankGPT": False, 
            "Pairwise": True, 
        }[self.value]

    @property
    def result_parser_name(self):
        return {
            "RankGPT": "text_list", 
            "Pairwise": "prob", 
        }[self.value]

    def __str__(self):
        return f"""[{self.value}]
          - prompt_builder_name: {self.prompt_builder_name})
          - use logits: {self.use_logits}
          - tresult_parser_name: {self.result_parser_name}
        """

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

