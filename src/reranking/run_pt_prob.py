import os
from pathlib import Path
from typing import Optional
import ir_measures
from ir_measures import *
import loader
from pointwise import pt_rerank
from utils.tools import load_runs
home_dir=str(Path.home())

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    topk: int = 100,
    **kwargs,
):
    run = load_runs(run_path, topk=topk, output_score=True)
    corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)
    run = {k: v for k, v in run.items() if k in qrels}

    if kwargs.get('vllm_backend', False):
        from llm.vllm_back import LLM
        model = LLM(model=model_name_or_path, logprobs=20)

    if kwargs.get('litellm_backend', False):
        from llm.litellm_api import LLM
        model = LLM(temperature=0, top_p=1.0, logprobs=20, max_tokens=3)
        model_name_or_path = 'llama3.3-70b-instruct'

    if kwargs.get('hf_backend', False):
        from llm.hf_encode import LLM
        model = LLM(model=model_name_or_path, model_class='clm', temperature=0) 

    reranked_run = pt_rerank(
        model=model,
        run=run,
        queries=queries,
        corpus=corpus,
        batch_size=32
    )

    if topk >= 100:
        with open(run_path.replace('runs', 'pt_reranked_runs'), 'w') as f:
            for qid in reranked_run:
                for i, (docid, score) in enumerate(reranked_run[qid].items()):
                    f.write(f"{qid} Q0 {docid} {i+1} {score} pt_rerank\n")

    # evaluation
    r1 = ir_measures.calc_aggregate([nDCG@10], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)
    return {
        'model_name_or_path': model_name_or_path, 
        'ir_datasets_name': ir_datasets_name,
        'run_path': run_path,
        'original': r1, 
        'reranked': r2
    }

# starting experiments
os.makedirs(f"{home_dir}/APRIL/pt_reranked_runs", exist_ok=True)
model_name_or_path='Qwen/Qwen2.5-7B-Instruct'

results = {}
for dataset in ['trec-dl-2019', 'trec-dl-2020']:
    results[dataset] = {}
    run_path = f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"
    results[dataset] = main(
        model_name_or_path=model_name_or_path,
        run_path=run_path,
        topk=100,
        ir_datasets_name=f'msmarco-passage/{dataset}/judged',
        use_logits=True, use_alpha=True,
        variable_passages=False,
        vllm_backend=False,
        litellm_backend=True
    )

print(results)
