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
for dataset in ['trec-dl-2019']:
    results[dataset] = {}

    from reranking.config_manager import ConfigManager
    config = ConfigManager(
        data={'ir_datasets_name': f'msmarco-passage/{dataset}/judged',
              'input_run': f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"},
        rerank_mode='Pairwise',
        top_k=10,
        llm={'max_model_len': 4096, 'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}
    ).get_config()

    from reranking.wrapper import ModularReranker
    rankllm = ModularReranker(config, 
        system_message= "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
    )

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

