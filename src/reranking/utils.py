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

    # @staticmethod
    # def covert_to_trec_run(
    #     results: List[Result], 
    #     file_path: str = 'temp.run'
    # ) -> str:
    #     trec_run = ""
    #     for result in results:
    #         for i, hit in enumerate(result.hits):
    #             trec_run += f"{result.qid} Q0 {hit['docid']} {i+1} {hit['score']} reranking\n"
    #     return trec_run

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

