import os
from typing import Optional
import ir_measures
from ir_measures import *

import loader
from pointwise import pt_rerank
from llm.vllm_back import LLM
from utils.tools import load_runs

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    **kwargs,
):
    run = load_runs(run_path, topk=100, output_score=True)
    corpus, queries, qrels = loader.load(
        ir_datasets_name, query_fields, doc_fields
    )

    model = LLM(
        model=model_name_or_path,
        temperature=0, 
        top_p=1.0,
        gpu_memory_utilization=0.9, 
        logprobs=20,
        prompt_logprobs=None,
    )

    reranked_run = pt_rerank(
        model=model,
        run=run,
        queries=queries,
        corpus=corpus,
        batch_size=16
    )

    with open(run_path.replace('runs', 'pt_reranked_runs'), 'w') as f:
        for qid in reranked_run:
            for i, (docid, score) in enumerate(reranked_run[qid].items()):
                f.write(f"{qid} Q0 {docid} {i+1} {score} pt_rerank\n")

    # evaluation
    r1 = ir_measures.calc_aggregate([nDCG@10, MRR@10], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10, MRR@10], qrels, reranked_run)
    print(r1)
    print(r2)

os.makedirs("../pt_reranked_runs", exist_ok=True)
main(
    model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    run_path="/home/dju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt",
    ir_datasets_name='msmarco-passage/trec-dl-2019',
)
main(
    model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    run_path="/home/dju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl20.txt",
    ir_datasets_name='msmarco-passage/trec-dl-2020',
)
