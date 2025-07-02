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

    # @classmethod
    # def from_string(cls, line: str):
    #     """ the string line of trec run file """
    #     parts = line.strip().split()
    #     qid = parts[0]
    #     query = parts[1]
    #     hits = []
    #     return cls(qid=qid, query=query, hits=hits)

    # run_dict = defaultdict(list)
    # with open(path, 'r') as f:
    #     for line in f:
    #         qid, _, docid, rank, score, _ = line.strip().split()
    #         if int(rank) <= (topk or 9999):
    #             run_dict[str(qid)] += [(docid, float(rank), float(score))]
    #
    # # sort by score and return static dictionary
    # sorted_run_dict = OrderedDict()
    # for qid, docid_ranks in run_dict.items():
    #     sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
    #     if output_score:
    #         sorted_run_dict[qid] = {docid: rel_score for docid, rel_rank, rel_score in sorted_docid_ranks}
    #     else:
    #         sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]
    # return sorted_run_dict

class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)

class RerankMode(Enum):
    RANK_GPT = "rank_GPT"
    LRL = "LRL"
    APRIL = "APRIL"

    @property
    def prompt_builder_name(self):
        return {
            RerankMode.RANK_GPT: "listwise",
            RerankMode.LRL: "listwise",
            RerankMode.APRIL: "listwise"
        }[self]

    @property
    def use_logits(self):
        return {
            RerankMode.RANK_GPT: False,
            RerankMode.LRL: False,
            RerankMode.APRIL: True
        }[self]

    @property
    def result_parser_name(self):
        return {
            RerankMode.RANK_GPT: "text_list",
            RerankMode.LRL: "prob_list",
            RerankMode.APRIL: "prob_list"
        }[self]

    def __str__(self):
        return f"[{self.value}]: (prompt_builder_name: {self.prompt_builder_name}) | (result_parser_name: {self.result_parser_name})"
    #     return self.value

