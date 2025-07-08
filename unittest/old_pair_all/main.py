import os
from pathlib import Path
from typing import Optional
import ir_measures
from ir_measures import *
import reranking.loader as loader
from model import rerank
home_dir=str(Path.home())

from reranking.llm_provider.vllm_api import LLM

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    topk: int = 100,
    **kwargs,
):
    run = loader.load_run(run_path, topk=topk)
    corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)
    run = {k: v for k, v in run.items() if k in qrels}

    model = LLM(model_name_or_path, temperature=0, top_p=1, logprobs=20, max_tokens=3, max_model_len=8196)

    system_prompt = """You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"""
    user_prompt = """I will provide you with two passages. Read and memorize both carefully. Your task is to determine which passage is more relevant to the query: {query}\n\n"""
    user_prompt += """"Passage 1: {cand1}\nPassage 2: {cand2}\nQuery: {query}\nBased on the query, is the Passage 1 more relevant than Passage 2?\nPlease answer 'Yes' or 'No'.\nAnswer: """
    template = model.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tokenize=False, 
        add_generation_prompt=True
    )

    reranked_run = rerank(
        model=model,
        run=run,
        queries=queries,
        corpus=corpus,
        batch_size=64,
        template=template,
    )

    if topk >= 50:
        with open(run_path.replace('runs', 'pa_reranked_runs'), 'w') as f:
            for qid in reranked_run:
                for i, (docid, score) in enumerate(reranked_run[qid].items()):
                    f.write(f"{qid} Q0 {docid} {i+1} {score} pa_rerank\n")

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
os.makedirs(f"{home_dir}/APRIL/pa_reranked_runs", exist_ok=True)
model_name_or_path='Qwen/Qwen2.5-7B-Instruct'

results = {}
for dataset in ['trec-dl-2019']:
    results[dataset] = {}
    run_path = f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"
    results[dataset] = main(
        model_name_or_path=model_name_or_path,
        run_path=run_path,
        topk=10,
        ir_datasets_name=f'msmarco-passage/{dataset}/judged',
        use_logits=True, use_alpha=True,
        variable_passages=False,
        vllm_backend=False,
        litellm_backend=True
    )
    print(results)
