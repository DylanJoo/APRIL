import csv
import json
import logging
import os
from collections import defaultdict, OrderedDict
from typing import Optional 

from crux.tools.mds.ir_utils import load_topic, get_qrel
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load(
    crux_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    ignore_corpus: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

    subset = crux_datasets_name.replace('crux-mds-', '')
    queries = load_topic(subset)
    logger.info("Query Example: %s", list(queries.values())[0])

    qrels = get_qrel(subset)
    logger.info("Qrel Example: %s", list(qrels.values())[0])

    if ignore_corpus:
        return None, queries, qrels

    # [TODO] revise this to fit all the document format 
    train_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='train')
    test_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='test')
    corpus = {example["id"]: {"contents": example["contents"]} for example in train_corpus}
    corpus.update({example["id"]: {"contents": example["contents"]} for example in test_corpus})
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
