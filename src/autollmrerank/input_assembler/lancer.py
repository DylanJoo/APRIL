import re
import math
import copy
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import Result, batch_iterator
from .base import RerankStrategy

TEMPLATE = """
Instruction: Given the following request, write {NUM} diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be short (ideally under 20 words) and should focus on a single aspect or dimension of the report.

Here are examples of sub-questions for a request about the mysteries of Machu Picchu's architecture:
- Where is Machu Picchu located?
- How high is the mountain ridge on which Machu Picchu sits?
- What make Machu Picchu one of the world's most visited sites?
- What are the most remarkable aspects of the construction structure of Machu Picchu?

Request:
{query}

Output format:
- List each sub-question on a new line. Do not number the sub-questions.
- Do not add any comment or explanation.
- Output without adding additional questions after the specified {NUM}. Begin with "<START OF LIST>" and, when you are finished, output "<END OF LIST>". Never ever add anything else after "<END OF LIST>", my life depends on it!!!

Now, generate the {NUM} sub-questions:
"""

class Lancer(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        num_runs: int = 2,
        **kwargs
    ) -> List[Result]:

        num_subquestions = num_runs

        # Copy and reset
        results = copy.deepcopy(init_results)
        for r in results:
            r.reset()

        # 1. Generate n sub-questions for all queries
        queries = [r.query for r in init_results]
        prompts = [self.preprocess(q, num_subquestions) for q in queries]
        with self._llm.default(): 
            outputs = self._llm.generate(prompts)
        subquestions = [self.postprocess(o, num_subquestions) for o in outputs]

        # 2. Answerability judgment for each question and add the score
        for i in range(num_subquestions):
            for j, r in enumerate(results):
                # r.query = r.query + "\n" + subquestions[j][i]
                r.query = subquestions[j][i]

            subresults = self.run_pass(
                results,
                rank_start=rank_start,
                rank_end=rank_end,
                batch_size=batch_size,
                **kwargs
            )

            for j, r in enumerate(results):
                r.append_subresult(subquestions[j][i], subresults[j])

        return results


    def run_pass(
        self, 
        init_results: List[Result],
        rank_start: int = 0,
        rank_end: int = None,
        batch_size: Optional[int] = 32,
        **kwargs
    ):
        results = [copy.deepcopy(result) for result in init_results]
        all_scores = {}
        
        for index, result in enumerate(results):

            ## Placeholder for scores
            result.hits = [hit for hit in result.hits[:rank_end]]
            all_scores[result.qid] = [0 for _ in result.hits]

            ## Create prompts for all pairs
            prompts = self._prompt_builder.create_prompt(result, rank_start=0, rank_end=rank_end)

            ## Iterate over pairs
            scores = []
            for batch_prompts in batch_iterator(prompts, batch_size):
                batch_scores = self._llm.generate(
                    batch_prompts, 
                    rating_logp=(self.config.result_parser_name == "rating_logp"),
                    expected_rating=(self.config.result_parser_name == "expected_rating"),
                )
                scores.extend(batch_scores)

            ## Score aggregation
            for i, score in enumerate(scores):
                all_scores[result.qid][i] = score

        ## Update results with scores
        reranked_results = self._result_parser.parse(
            [all_scores[result.qid] for result in results], 
            init_results
        )
        return reranked_results

    # NOTE: ad-hoc adoption of LANCER method
    def preprocess(self, query, num_subquestions=2):
        template = self._prompt_builder._tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": TEMPLATE}],
            tokenize=False,
            add_generation_prompt=True
        )
        return template.format(NUM=num_subquestions, query=query)

    def postprocess(self, llm_output, n):
        pattern = r'<START OF LIST>(.*?)<END OF LIST>'
        match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = llm_output

        subquestions = extracted.split("\n")
        subquestions = [s.strip() for s in subquestions if (s and s not in ["START OF LIST", "END OF LIST"])]
        subquestions = [re.sub(r'^[\-\*\d\.\)\s]+', '', s) for s in subquestions]
        subquestions = [s for s in subquestions if s != ""]
        return subquestions[:n]

