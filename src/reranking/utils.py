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

class PromptMode(Enum):
    RANK_GPT = "rank_GPT"
    LRL = "LRL"
    APRIL = "APRIL"

    def __str__(self):
        return self.value

