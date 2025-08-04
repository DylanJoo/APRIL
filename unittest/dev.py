import os
from pathlib import Path
from reranking import loader
from pprint import pprint
import ir_measures
from ir_measures import *
home_dir=str(Path.home())

# Initialize the reranker with the configuration
from reranking.config_manager import ConfigManager
config = ConfigManager(
    rerank_mode='Dev',
    top_k=100,
    rank_start=0,
    rank_end=100,
    step_size=10,
    window_size=20,
    num_runs=1,
    llm={'max_model_len': 8196, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
).get_config()

from reranking.wrapper import ModularReranker
rankllm = ModularReranker(config, 
    system_message= "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
)

# start reranking
results = {}
for dataset in ['trec-dl-2019', 'trec-dl-2020']:
    results[dataset] = {}

    if ('2019' in dataset) or ('2020' in dataset):
        config.data.ir_datasets_name = f'msmarco-passage/{dataset}/judged'
        config.data.input_run = f"{home_dir}/APRIL/runs/run.msmarco-passage.bm25.{dataset}.txt"

    if ('2021' in dataset) or ('2022' in dataset):
        config.data.ir_datasets_name = f'msmarco-passage-v2/{dataset}/judged'
        config.data.input_run = f"{home_dir}/APRIL/runs/run.msmarco-passage-v2.bm25.{dataset}.txt"

    run = loader.load_run(config.data.input_run)
    corpus, queries, qrels = loader.load(
        config.data.ir_datasets_name, 
        query_fields=None, 
        doc_fields=None
    )
    run = {qid: hit for qid, hit in run.items() if qid in qrels} # filter

    reranked_run = rankllm.rerank(
        run=run,
        queries=queries,
        corpus=corpus,
        query_batch_size=64,
    )

    # prepare output run
    output_path = os.path.join(config.data.input_run.replace('runs', f'runs/{config.rerank_mode}'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for qid in reranked_run:
            for i, (docid, score) in enumerate(reranked_run[qid].items()):
                f.write(f"{qid} Q0 {docid} {i+1} {score} li_rerank\n")

    # evaluation
    r1 = ir_measures.calc_aggregate([nDCG@10], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)

    eval_log = {
        'model_name_or_path': config.llm.model_name_or_path, 
        'ir_datasets_name': config.data.ir_datasets_name,
        'run_path': config.data.input_run,
        'original': r1, 
        'reranked': r2
    }
    results[dataset] = eval_log
    pprint(eval_log)

