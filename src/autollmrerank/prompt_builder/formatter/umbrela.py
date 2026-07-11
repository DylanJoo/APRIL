# UMBRELA zero-shot relevance judgments (bing-style prompt).
# Ported verbatim from castorini/umbrela's qrel_zeroshot_bing.yaml template.
from typing import List, Optional, Union, Callable, Dict, Tuple
from .base import BaseFormatter

class UmbrelaFormatter(BaseFormatter):

    def prefix(self, **kwargs) -> str:
        return (
            "Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n"
            "0 = represent that the passage has nothing to do with the query, \n"
            "1 = represents that the passage seems related to the query but does not answer it, \n"
            "2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and \n"
            "3 = represents that the passage is dedicated to the query and contains the exact answer.\n\n"
            "Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.\n\n"
        )

    def postfix(self, **kwargs) -> str:
        return (
            "Split this problem into steps:\n"
            "Consider the underlying intent of the search.\n"
            "Measure how well the content matches a likely intent of the query (M).\n"
            "Measure how trustworthy the passage is (T).\n"
            "Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.\n"
            "Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.\n"
        )

    def body(self, query, doc_list, **kwargs) -> str:
        prompts = []
        doc_list = [self._document_format(doc) for doc in doc_list]
        for doc in doc_list:
            prompt = f"Query: {query}\nPassage: {doc}\n\n"
            prompts.append(prompt)
        return prompts
