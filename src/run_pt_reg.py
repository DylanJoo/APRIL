import os
from pathlib import Path
from typing import Optional
import ir_measures
from ir_measures import *
import loader
from pointwise import pt_rerank
from llm.hf_encode import LLM
from utils.tools import load_runs
home_dir=str(Path.home())

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

    model = LLM(model=model_name_or_path, model_class='clm', temperature=0) 

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
    r1 = ir_measures.calc_aggregate([nDCG@10, nDCG@20], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10, nDCG@20], qrels, reranked_run)
    print(r1)
    print(r2)

# starting experiments
os.makedirs(f"{home_dir}/APRIL/li_reranked_runs", exist_ok=True)

# model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
# model_name_or_path='meta-llama/Llama-3.1-8B-Instruct'

main(
    model_name_or_path='allenai/Llama-3.1-Tulu-3.1-8B',
    run_path=f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt",
    ir_datasets_name='msmarco-passage/trec-dl-2019/judged',
)
# main(
#     model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
#     run_path=f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl20.txt",
#     ir_datasets_name='msmarco-passage/trec-dl-2020',
# )
