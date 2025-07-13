import os
from pathlib import Path
from reranking import loader
from pprint import pprint
import ir_measures
from ir_measures import *
home_dir=str(Path.home())

# Prepare data (inout and output)
os.makedirs(f"{home_dir}/APRIL/pa_reranked_runs", exist_ok=True)

results = {}
for dataset in ['trec-dl-2019', 'trec-dl-2020']:
    results[dataset] = {}

    from reranking.config_manager import ConfigManager
    config = ConfigManager(
        data={'ir_datasets_name': f'msmarco-passage/{dataset}/judged',
              'input_run': f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"},
        rerank_mode='oracle_top100',
    ).get_config()

    run = loader.load_run(config.data.input_run)
    corpus, queries, qrels = loader.load(
        config.data.ir_datasets_name, 
        query_fields=None, 
        doc_fields=None
    )
    run = {qid: hit for qid, hit in run.items() if qid in qrels} # filter

    reranked_run = {}
    for qid, qrel in qrels.items():
        reranked_run[qid] = {}
        for rank, docid in enumerate(run[qid], start=1):
            if rank > 100:
                break
            if docid in qrel:
                reranked_run[qid].update({docid: qrel[docid]})

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
