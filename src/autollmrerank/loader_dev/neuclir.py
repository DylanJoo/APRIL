import csv
import json
import logging
import os
from collections import defaultdict, OrderedDict
from typing import Optional 

from crux.tools.neuclir.ir_utils import load_topic, get_qrel
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load(
    dataset_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    ignore_corpus: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

    queries = load_topic()
    logger.info("Query Example: %s", list(queries.values())[0])

    qrels = get_qrel()
    logger.info("Qrel Example: %s", list(qrels.values())[0])

    if ignore_corpus:
        return None, queries, qrels

    # [TODO] revise this to fit all the document format 
    # CoveR's top1000: /home/dju/trec2026/data/neuclir/neuclir24-relevant-docs.jsonl
    # All: /home/dju/scratch/neuclir1/*.processed.jsonl.gz
    ds = load_dataset('json', 
            data_files='/home/dju/trec2026/data/neuclir/neuclir24-relevant-docs.jsonl.gz',
            num_proc=3, 
            split='train'
    )
    corpus = {example["id"]: {"contents": example["title"] + " " + example["text"]} \
            for example in ds}
    logger.info("Doc Example: %s", list(corpus.values())[0])

    return corpus, queries, qrels

# [deprecated] will use the function above instead
def load_run(path, topk=100):
    run_dict = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= (topk or 9999):
                run_dict[str(qid)] += [(docid, float(rank), float(score))]

    # sort by score and return static dictionary
    sorted_run_dict = OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        sorted_run_dict[qid] = {docid: rel_score for docid, rel_rank, rel_score in sorted_docid_ranks}

    return sorted_run_dict
