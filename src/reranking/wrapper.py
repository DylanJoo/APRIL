from typing import Optional, Tuple, List, Dict, Union, Any
from pprint import pprint

from reranking.utils import RerankMode, Result
from reranking.config_manager import ConfigManager
from reranking.input_assembler import BubbleSort

from reranking.prompt_builder import PromptBuilder
from reranking.llm_provider.vllm_api import LLM
from reranking.result_parser import ResultParser

class ModularReranker:

    def __init__(self, 
        config, 
        include_system_message: Optional[bool] = True,
        system_message: Optional[str] = None,
    ) -> None:

        # initialize method
        rerank_mode = RerankMode(config.rerank_mode)
        print(f"[Model] {config.llm.model_name_or_path}") 
        print(f"{rerank_mode}")

        # initlaize instances 
        formatter = PromptBuilder(
            model_name_or_path=config.llm.model_name_or_path,
            rerank_mode=rerank_mode,
            include_system_message=include_system_message,
            system_message=system_message,
        )
        llm = LLM( # [NOTE] assume the backend is vllm, change it later
            model_name_or_path=config.llm.model_name_or_path,
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
            logprobs=None if config.rerank_mode == RerankMode.RANK_GPT else 30,
            max_tokens=100 if config.rerank_mode == RerankMode.RANK_GPT else 2,
        )
        llm.set_classification()

        processor = ResultParser(
            rerank_mode=RerankMode(config.rerank_mode),
            formatter=formatter,
        )

        self.assembler = BubbleSort(
            config=config, 
            formatter=formatter,
            llm_provider=llm,
            processor=processor,
        )

        ## [Attibutes]
        # self._context_size = context_size
        # self._window_size = window_size
        # self._step_size = step_size
        # self._batched = batched

    @staticmethod
    def convert_run_to_result(run, queries=None, corpus=None):
        results = []
        for qid, hits in run.items():
            query = queries[qid]
            hits = []
            for docid, score in hits.items():
                hits.append({'docid': docid, 'score': float(score), 'content': corpus[docid]['contents']})
            results.append(Result(qid=qid, query=query, hits=hits))
        return results

    def rerank(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """
        Args
            run (Dict[str, Dict[str, float]]): The initial run to be reranked.
            queries: (Dict[str, str]): A dictionary mapping query IDs to query strings.
            corpus (Dict[str, Dict[str, str]]): A dictionary mapping document IDs to their content and title (if applicable).
        """
        init_results = self.convert_run_to_result(run, queries, corpus)

        reranked_results = self.assembler.run(
            retrieved_run=init_results,
            batch_size=batch_size,
        )
        return 0

if __name__ == '__main__':
    config = ConfigManager().get_config()

    # data loading
    modurlar_reranker = ModularReranker(config=config)
    modular_reranker.rerank(
        run={},
        queries={},
        corpus={},
        batch_size=64
    )
