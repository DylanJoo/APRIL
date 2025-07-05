import copy
from typing import Optional, Tuple, List, Dict, Union, Any

from ..utils import RerankMode, Result
from ..prompt_builder import PromptBuilder
from ..result_parser import ResultParser
from .base import RerankStrategy

class PairAll(RerankStrategy):

    def run(
        self,
        init_results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: Optional[int] = 128,
    ) -> List[Result]:
        prompts = []
        id_pairs = []

        # Build pairwise prompts
        for result in init_results:
            qid = result.qid
            docids = [doc.docid for doc in result.hits]
            query = result.query
            pairs = [(i, j) for i in range(len(docids)) for j in range(len(docids)) if i != j]

            for i, j in pairs:
                prompt = self.template.format(
                    cand1=result.doc_map[docids[i]]["contents"],
                    cand2=result.doc_map[docids[j]]["contents"],
                    query=query
                )
                prompts.append(prompt)
                id_pairs.append((qid, i, j))

        logger.info(f"Number of prompts: {len(prompts)}")

        # Set up tokenizer and label mapping
        true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
        self._llm.set_classification(true_list, false_list)

        # Batch inference
        scores = []
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_scores = self._llm.inference_chat(self.system_prompt, batch_prompts)
            scores.extend(batch_scores)

        # Aggregate pairwise scores
        all_scores = {result.qid: [0 for _ in result.hits] for result in init_results}
        for (qid, i, j), score in zip(id_pairs, scores):
            all_scores[qid][i] += score
            all_scores[qid][j] += (1 - score)

        # Update and return reranked results
        reranked_results = []
        for result in init_results:
            qid = result.qid
            docids = [doc.docid for doc in result.hits]
            scores = all_scores[qid]
            doc_score_pairs = list(zip(docids, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            result.hits = result.update_hits_from_scores(doc_score_pairs)
            reranked_results.append(result)

        return reranked_results

