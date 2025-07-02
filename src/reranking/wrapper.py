""" [NOTE] add configuration control. """
from typing import Optional, Tuple, List, Dict, Union, Any
from reranking.utils import RerankMode

class ModularReranker:

    def __init__(
        self,
        model_name_or_path: str,
        rerank_mode: RerankMode = RerankMode.RANK_GPT,
        include_system_message: bool = False,
        system_message: str = None,
        context_size: int = 4096,
        window_size: int = 20,
        step_size: int = 10,
        batched: bool = False,
        backend: str = 'vllm',
    ) -> None:
        pass

    def rerank(
        self,
        run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, str],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """
        Rerank the given run using the model.
        
        Args:
            run (Dict[str, Dict[str, float]]): The run to rerank.
            queries (Dict[str, str]): The queries.
            corpus (Dict[str, str]): The corpus.
            batch_size (int): The batch size for processing.

        Returns:
            Dict[str, Dict[str, float]]: The reranked run.
        """
        return 0

if __main__ == '__main__':
    import argparse
    import os
    from pathlib import Path
    from typing import Optional
    import ir_measures
    from ir_measures import *
    import loader
    from utils.tools import load_runs
    home_dir=str(Path.home())
    ALPH_START_IDX = ord('A')-1

    parser = argparse.ArgumentParser(description="Modular Reranker")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--rerank_mode', type=str, default='RANK_GPT', help='Rerank mode')
    parser.add_argument('--include_system_message', action='store_true', help='Include system message')
    parser.add_argument('--system_message', type=str, default=None, help='System message')
    parser.add_argument('--context_size', type=int, default=4096, help='Context size')
    parser.add_argument('--window_size', type=int, default=20, help='Window size')
    parser.add_argument('--step_size', type=int, default=10, help='Step size')
    parser.add_argument('--batched', action='store_true', help='Use batched processing')
    parser.add_argument('--backend', type=str, default='vllm', help='Backend to use')
    args = parser.parse_args()

    modurlar_reranker = ModularReranker(
        model_name_or_path=args.model_name_or_path,
        rerank_mode=args.rerank_mode,
        include_system_message=args.include_system_message,
        system_message=args.system_message,
        context_size=args.context_size,
        window_size=args.window_size,
        step_size=args.step_size,
        batched=args.batched,
        backend=args.backend
    )

    modular_reranker.rerank(
        run={},
        queries={},
        corpus={},
        batch_size=64
    )

#     run = load_runs(run_path, topk=topk, output_score=True)
#     corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)
#     run = {k: v for k, v in run.items() if k in qrels}
#
#     if kwargs.get('vllm_backend', False):
#         from llm.vllm_back import LLM
#         model = LLM(model=model_name_or_path, temperature=0, top_p=1, logprobs=20)
#
#     if kwargs.get('litellm_backend', False):
#         from llm.litellm_api import LLM
#         model = LLM(temperature=0, top_p=1.0, logprobs=20, max_tokens=3)
#         model_name_or_path = 'llama3.3-70b-instruct'
#
#     reranked_run = april_rerank(
#         model=model,
#         run=run,
#         queries=queries,
#         corpus=corpus,
#         batch_size=64
#     )
#
#     if topk >= 50:
#         with open(run_path.replace('runs', 'pa_reranked_runs'), 'w') as f:
#             for qid in reranked_run:
#                 for i, (docid, score) in enumerate(reranked_run[qid].items()):
#                     f.write(f"{qid} Q0 {docid} {i+1} {score} pa_rerank\n")
#
#     # evaluation
#     r1 = ir_measures.calc_aggregate([nDCG@10], qrels, run)
#     r2 = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)
#     return {
#         'model_name_or_path': model_name_or_path, 
#         'ir_datasets_name': ir_datasets_name,
#         'run_path': run_path,
#         'original': r1, 
#         'reranked': r2
#     }
#
# # starting experiments
# os.makedirs(f"{home_dir}/APRIL/pa_reranked_runs", exist_ok=True)
# model_name_or_path='Qwen/Qwen2.5-7B-Instruct'
#
# results = {}
# for dataset in ['trec-dl-2019']:
#     results[dataset] = {}
#     run_path = f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"
#
#     from llm.litellm_api import LLM
#     model = LLM(temperature=0, top_p=1.0, logprobs=20, max_tokens=3)
#     model_name_or_path = 'llama3.3-70b-instruct'
#
#     results[dataset] = main(
#         model_name_or_path=model_name_or_path,
#         run_path=run_path,
#         topk=100,
#         ir_datasets_name=f'msmarco-passage/{dataset}/judged',
#         use_logits=True, use_alpha=True,
#         variable_passages=False,
#         vllm_backend=False,
#         litellm_backend=True
#     )
#
# print(results)
