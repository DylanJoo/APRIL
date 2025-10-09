import os
from typing import Optional, Tuple, List, Dict, Union, Any
from pprint import pprint
from tqdm import tqdm
import torch
from functools import wraps
import time

from .utils import Result, batch_iterator
from .input_assembler import AutoAssembler
from .prompt_builder import PromptBuilder
from .result_parser import ResultParser
from .config_manager import ConfigManager

class AutoLLMReranker:

    @classmethod
    def from_prebuilt(cls, method_name, model_name_or_path, **kwargs) -> "AutoLLMReranker":
        import importlib.resources as pkg_resources
        default_path = pkg_resources.files("reranking.configs").joinpath(f"{method_name}.yaml")
        path = pkg_resources.files("reranking.configs").joinpath(f"{method_name}.yaml")
        path = path if path.exists() else default_path

        # TODO: figure out what else
        llmconfig = {'model_name_or_path': model_name_or_path}
        llmconfig.update(kwargs.pop('llm', {}))
        config = ConfigManager(path=path, llm=llmconfig, **kwargs).get_config()
        return cls(config, **kwargs)

    @staticmethod
    def timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"\n\n{func.__qualname__} took {end - start:.6f} seconds")
            return result
        return wrapper

    def __init__(self, config, **kwargs) -> None:

        self.config = config
        prompt_builder = PromptBuilder(config=config)

        # NOTE: do we want to make it clearer in the wrapper?
        if config.llm.backend == 'vllm':
            from .llm_provider.vllm_api import LLM  # for v100
        if (config.llm.backend == 'openai') or (config.llm.backend == 'request'):
            from .llm_provider.request import LLM

        agent = LLM( 
            model_name_or_path=config.llm.model_name_or_path,
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
            logprobs=20 if config.llm.use_logits else None,
            max_model_len=config.llm.max_model_len,
            max_tokens=5 if config.llm.use_logits else 128,
            dtype='half' if config.llm.dtype == 'float16' else 'float32',
            num_gpus=max(1, int(torch.cuda.device_count())), # NOTE: do we want it to be specified?
            base_url='http://localhost:8000/v1',
            api_key='EMPTY'
        )
        # TODO: Make this more flexible in the future
        if config.llm.use_logits:
            agent.set_classification(id_strings=[chr(i) for i in range(65, 91)])

        result_parser = ResultParser(use_alpha=config.use_alphabetical)

        # initialize the algorithm module
        self.assembler = AutoAssembler.from_config(
            config, 
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

    # TODO: Figure out another input format called `text_pairs=[(q1, [d1, d2, ...]), (q2, ...)]`. 
    # This is more friendly for users who only have texts.
    @timer
    def rerank(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        query_batch_size: int = 32,
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
        for batch_results in tqdm(
            batch_iterator(init_results, size=query_batch_size), 
            desc=f"Reranking with query batch size {query_batch_size}",
            total=len(init_results) // query_batch_size + 1
        ):
            batch_reranked_results = self.assembler.run(
                init_results=batch_results, 
                rank_start=0,
                rank_end=min(self.config.rank_end, self.config.top_k),
                batch_size=query_batch_size,
                num_runs=self.config.num_runs,
                references=self.references if hasattr(self, 'references') else None,
                anchor_index=self.config.anchor_index if hasattr(self.config, 'anchor_index') else None,
            )
            reranked_results.extend(batch_reranked_results)

        # sort 
        for r in reranked_results:
            r.sort_by(field='score')

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

if __name__ == "__main__":
    import ir_measures
    from ir_measures import *
    from reranking import loader

    # init config with CLI commands
    config = ConfigManager().get_config()
    config_dict = ConfigManager().get_config(return_dict=True)
    pprint(config_dict)

    results = {}

    # init reranker
    rankllm = AutoLLMReranker(config, system_message=config.system_message)

    # load data
    run = loader.load_run(config.data.input_run)
    corpus, queries, qrels = loader.load(config.data.ir_datasets_name, query_fields=None, doc_fields=None)
    run = {qid: hit for qid, hit in run.items() if qid in qrels} # filter

    # reranking
    reranked_run = rankllm.rerank(run=run, queries=queries, corpus=corpus, query_batch_size=config.data.batch_size)

    # output reranked result
    output_path = os.path.join(config.data.input_run.replace('runs', f'runs/{config.rerank_mode}'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for qid in reranked_run:
            for i, (docid, score) in enumerate(reranked_run[qid].items()):
                f.write(f"{qid} Q0 {docid} {i+1} {score} {config.rerank_mode}\n")

    # evaluation
    r1 = ir_measures.calc_aggregate([nDCG@10], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)

    # print logs
    eval_log = {
        'rerank_mode': config.rerank_mode,
        'model_name_or_path': config.llm.model_name_or_path, 
        'ir_datasets_name': config.data.ir_datasets_name,
        'run_path': config.data.input_run,
        'original': r1, 
        'reranked': r2
    }
    pprint(eval_log)
