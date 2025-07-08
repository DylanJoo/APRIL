from typing import Optional, Tuple, List, Dict, Union, Any
from pprint import pprint
from tqdm import tqdm

from .utils import RerankMode, Result, batch_iterator
from .config_manager import ConfigManager
from .input_assembler import AutoAssembler
from .prompt_builder import PromptBuilder
# from .llm_provider.vllm_api import LLM # use a llm wrapper to handle
from .llm_provider.litellm_api import LLM
from .result_parser import ResultParser

class ModularReranker:

    def __init__(self, 
        config, 
        include_system_message: Optional[bool] = True,
        system_message: Optional[str] = None,
    ) -> None:
        self.config = config

        # initialize method
        rerank_mode = RerankMode(config.rerank_mode)
        print(f"[Model] {config.llm.model_name_or_path}") 
        print(f"{rerank_mode}")

        # initlaize instances 
        prompt_builder = PromptBuilder(
            config=config,
            rerank_mode=rerank_mode,
            include_system_message=include_system_message,
            system_message=system_message,
            use_alpha=False, 
            variable_passages=True,
        )
        agent = LLM( 
            model_name_or_path=config.llm.model_name_or_path,
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
            logprobs=20 if rerank_mode.use_logits else None,
            max_tokens=128 if 'list' in rerank_mode.result_parser_name else 3,
            max_model_len=config.llm.max_model_len,
        )

        if rerank_mode.use_logits:
            agent.set_classification()
        result_parser = ResultParser()

        # initialize the algorithm module
        self.assembler = AutoAssembler.from_config(
            config, 
            rerank_mode=rerank_mode,
            prompt_builder=prompt_builder,
            llm_provider=agent,
            result_parser=result_parser,
        )

    @staticmethod
    def convert_run_to_result(run, queries=None, corpus=None):
        results = []
        for qid, hits in run.items():
            query = queries[qid]
            hit_docs = []
            for docid, score in hits.items():
                hit_docs.append({'docid': docid, 'score': float(score), 'content_dict': corpus[docid]})
            results.append(Result(qid=qid, query=query, hits=hit_docs))
        return results

    def rerank(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        query_batch_size: int = 16,
    ) -> Dict[str, Dict[str, float]]:
        """
        Args
            run (Dict[str, Dict[str, float]]): The initial run to be reranked.
            queries: (Dict[str, str]): A dictionary mapping query IDs to query strings.
            corpus (Dict[str, Dict[str, str]]): A dictionary mapping document IDs to their content and title (if applicable).
            batch_size (int): The number of query (with their results) to process in each batch.
        """
        init_results = self.convert_run_to_result(run, queries, corpus)

        reranked_results = []
        for batch_results in tqdm(batch_iterator(init_results, size=query_batch_size), desc="Batch reranking"):
            batch_reranked_results = self.assembler.run(
                init_results=batch_results, 
                rank_start=0,
                rank_end=min(self.config.rank_end, self.config.top_k),
                batch_size=query_batch_size,
            )
            reranked_results.extend(batch_reranked_results)

        # sort 
        reranked_results.sort_by(field='score')

        # covert back to run
        reranked_run = {}
        for result in reranked_results:
            reranked_run[result.qid] = {}
            for rank, hit in enumerate(result.hits, start=1):
                hit['rank'] = rank
                if 'score' in hit:
                    reranked_run[result.qid].update({ hit['docid']: hit['score'] })
                else:
                    reranked_run[result.qid].update({ hit['docid']: 1/rank })

        return reranked_run
