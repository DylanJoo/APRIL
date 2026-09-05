import os
import numpy as np
from pathlib import Path
from pprint import pprint
import ir_measures
from ir_measures import *
home_dir=str(Path.home())

# Initialize the reranker with the configuration
from autollmrerank.config_manager import ConfigManager
config = ConfigManager(
    rerank_mode='Lancer',
    top_k=100,
    rank_start=0,
    rank_end=100,
    step_size=10,
    window_size=20,
    num_runs=2,
    llm={'max_model_len': 8196, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'},
    system_message="You are a helpful, honest, and harmless assistant.",
    result_parser_name='text'
).get_config()

from autollmrerank.wrapper_dev import AutoLLMReranker
rankllm = AutoLLMReranker(config)

# start reranking
config.data.loader_type = 'cruxmds'
config.data.dataset_name = 'crux-mds-duc04'
config.data.input_run = f"{home_dir}/APRIL/runs/run.bm25.crux-mds-duc04.txt"
config.data.input_diversity_qrels = f"{home_dir}/datasets/crux/crux-mds-duc04/qrels/div_qrels-tau3.txt"
config.data.input_ratings = f"{home_dir}/datasets/crux/crux-mds-duc04/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl"
from crux.tools import load_diversity_qrel, load_ratings
div_qrels = load_diversity_qrel(config.data.input_diversity_qrels)
ratings = load_ratings(config.data.input_ratings)

from autollmrerank.loader_dev import cruxmds as loader
run = loader.load_run(config.data.input_run)
corpus, queries, qrels = loader.load(config.data.dataset_name, query_fields=None, doc_fields=None)
run = {qid: hit for qid, hit in run.items() if qid in qrels} # filter

reranked_run = rankllm.rerank(
    run=run,
    queries=queries,
    corpus=corpus,
    query_batch_size=128,
)

# prepare output run
output_path = os.path.join(config.data.input_run.replace('runs', f'runs/{config.rerank_mode}'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for qid in reranked_run:
        for i, (docid, score) in enumerate(reranked_run[qid].items()):
            f.write(f"{qid} Q0 {docid} {i+1} {score} rerank\n")

# evaluation
from crux.evaluation.rac_eval import rac_eval
r1 = rac_eval(
    run=run, 
    qrel=qrels, div_qrel=div_qrels,
    run_b=None, 
    tau=3,
    cutoff=10,
    judge=ratings, 
    filter_by_oracle=True
)
r2 = rac_eval(
    run=reranked_run, 
    qrel=qrels, div_qrel=div_qrels, 
    run_b=None, 
    tau=3,
    cutoff=10,
    judge=ratings, 
    filter_by_oracle=True
)
for key, values in r1.items():
    r1[key] = np.mean(values).item()
for key, values in r2.items():
    r2[key] = np.mean(values).item()

# print logs
eval_log = {
    'rerank_mode': config.rerank_mode,
    'model_name_or_path': config.llm.model_name_or_path, 
    'dataset_name': f"{config.data.loader_type}:{config.data.dataset_name}",
    'run_path': config.data.input_run,
    'original': r1, 
    'reranked': r2
}
pprint(eval_log)
