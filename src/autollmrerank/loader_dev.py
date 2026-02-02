import csv
import json
import logging
import os
from collections import defaultdict, OrderedDict
from typing import Optional 

logger = logging.getLogger(__name__)

def load_crux(
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    ignore_corpus: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

    dataset = ir_datasets.load(ir_datasets_name)
    corpus, queries, qrels = {}, {}, {}

    # ids
    query_fields = ['text'] if query_fields is None else query_fields
    doc_fields = ['text'] if doc_fields is None else doc_fields

    logger.info("Loading Queries...")
    for query in dataset.queries_iter():
        query_contents = [getattr(query, f) for f in query_fields]
        query_contents = " ".join(query_contents)
        queries[query.query_id] = query_contents
    logger.info("Query Example: %s", list(queries.values())[0])

    logger.info("Loading Qrels...")
    n = 0
    for qrel in dataset.qrels_iter():
        n += 1
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {qrel.doc_id: qrel.relevance}
        else:
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    logger.info("Qrel Example: %s (%s)", n, list(qrels.values())[0])

    if ignore_corpus:
        return None, queries, qrels

    # [TODO] revise this to fit all the document format 
    logger.info("Loading Corpus...")
    for doc in dataset.docs_iter():
        contents = [getattr(doc, f) for f in doc_fields]
        contents = " ".join(contents)
        corpus[doc.doc_id] = {"contents": contents}
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
