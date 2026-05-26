import os
from pathlib import Path
from pprint import pprint
import ir_measures
from ir_measures import *
home_dir=str(Path.home())
crux_dir=os.environ['CRUX_ROOT']

# Initialize the reranker with the configuration
from autollmrerank.config_manager import ConfigManager
config = ConfigManager(
    rerank_mode='Judge',
    top_k=100,
    rank_start=0,
    rank_end=100,
    step_size=10,
    window_size=20,
    num_runs=1,
    llm={'max_model_len': 8196, 'backend': 'vllm_dev', 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct', 'use_logits': False},
    result_parser_name='text',
).get_config()

# additional configs
config.data.dataset_name = 'crux-mds-duc04'
config.data.input_diversity_qrel = f"{crux_dir}/{config.data.dataset_name}/qrels/div_qrels-tau3.txt"
config.data.input_ratings = f"{crux_dir}/{config.data.dataset_name}/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl"
config.data.input_run = f"{home_dir}/APRIL/runs/run.bm25.{config.data.dataset_name}.txt"
config.data.loader_type = 'cruxmds'

import importlib
loader = importlib.import_module(f"autollmrerank.loader_dev.{config.data.loader_type}")

from autollmrerank.wrapper import AutoLLMReranker
rankllm = AutoLLMReranker(config, 
    system_message= "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query"
)

# start reranking
results = {}
corpus, queries, qrels = loader.load(
    config.data.dataset_name, 
    query_fields=None, 
    doc_fields=None
)
run = loader.load_run(config.data.input_run)
# for testing, only run the first 5
qrels = {k: v for i, (k, v) in enumerate(qrels.items()) if i < 5}
run = {qid: hit for qid, hit in run.items() if qid in qrels} # filter

from crux.tools import load_diversity_qrel, load_ratings
div_qrels = load_diversity_qrel(config.data.input_diversity_qrel)
ratings = load_ratings(config.data.input_ratings)

reranked_run = rankllm.rerank(
    run=run,
    queries=queries,
    corpus=corpus,
    query_batch_size=64,
)

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


eval_log = {
    'model_name_or_path': config.llm.model_name_or_path, 
    'ir_datasets_name': config.data.ir_datasets_name,
    'run_path': config.data.input_run,
    'original': r1, 
    'reranked': r2
}
pprint(eval_log)

