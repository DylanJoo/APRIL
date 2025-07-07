import os
from pathlib import Path
from reranking import loader
from pprint import pprint
home_dir=str(Path.home())

# Prepare data (inout and output)
os.makedirs(f"{home_dir}/APRIL/pa_reranked_runs", exist_ok=True)

# Prepare reranker
from reranking.wrapper import ModularReranker
rankllm = ModularReranker(config, 
    system_message= "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
)


# start reranking
import ir_measures
from ir_measures import *

results = {}
for dataset in ['trec-dl-2019', 'trec-dl-2020']:
    results[dataset] = {}

    from reranking.config_manager import ConfigManager
    config = ConfigManager().get_config()
    config.data.ir_datasets_name = f'msmarco-passage/{dataset}/judged'
    config.data.input_run = f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"

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
        query_batch_size=32
    )

    # prepare output run
    output_path = os.path.join(run_path.replace('runs', f'runs/{config.rerank_mode}'))
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

