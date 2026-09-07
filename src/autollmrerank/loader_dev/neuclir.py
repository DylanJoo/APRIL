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
