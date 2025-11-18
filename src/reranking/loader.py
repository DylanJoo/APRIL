import csv
import json
import logging
import os
import ir_datasets
from collections import defaultdict, OrderedDict
from typing import Optional 

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets library not available. Install with: pip install datasets")

def load(
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

def load_hf(
    dataset_name_queries: str,
    dataset_name_corpus: str,
    subset: str,
    query_split: str = 'test',
    corpus_split: str = 'corpus',
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    ignore_corpus: bool = False,
    qrels_split: Optional[str] = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """
    Load queries and corpus from HuggingFace datasets.
    
    Args:
        dataset_name_queries: HuggingFace dataset name for queries (e.g., 'DylanJHJ/nano-beir')
        dataset_name_corpus: HuggingFace dataset name for corpus (e.g., 'DylanJHJ/nano-beir-corpus')
        subset: Dataset subset/configuration name (e.g., 'nfcorpus')
        query_split: Split name for queries (default: 'test')
        corpus_split: Split name for corpus (default: 'corpus')
        query_fields: List of query fields to concatenate (default: ['query_texts'])
        doc_fields: List of document fields to concatenate (default: ['title', 'text'])
        ignore_corpus: If True, skip loading corpus (default: False)
        qrels_split: Optional split name for qrels (e.g., 'qrels'). If None, qrels are not loaded.
    
    Returns:
        Tuple of (corpus, queries, qrels) where:
        - corpus: Dict mapping doc_id to dict with 'contents' key
        - queries: Dict mapping query_id to query text
        - qrels: Dict mapping query_id to dict of doc_id to relevance score
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace datasets library is required. Install with: pip install datasets")
    
    corpus, queries, qrels = {}, {}, {}
    
    # Set default fields
    query_fields = ['query_texts'] if query_fields is None else query_fields
    doc_fields = ['title', 'text'] if doc_fields is None else doc_fields
    
    # Load queries
    logger.info(f"Loading Queries from {dataset_name_queries} (subset: {subset}, split: {query_split})...")
    query_dataset = load_dataset(dataset_name_queries, subset, split=query_split)
    
    for item in query_dataset:
        query_id = str(item['query_id'])
        # Concatenate specified query fields
        query_contents = []
        for field in query_fields:
            if field in item and item[field]:
                query_contents.append(str(item[field]))
        query_text = " ".join(query_contents)
        queries[query_id] = query_text
    
    logger.info(f"Loaded {len(queries)} queries")
    if queries:
        logger.info("Query Example: %s", list(queries.values())[0])
    
    # Load qrels if available
    if qrels_split:
        logger.info(f"Loading Qrels from {dataset_name_queries} (subset: {subset}, split: {qrels_split})...")
        try:
            qrels_dataset = load_dataset(dataset_name_queries, subset, split=qrels_split)
            for item in qrels_dataset:
                query_id = str(item.get('query_id', item.get('qid', '')))
                doc_id = str(item.get('doc_id', item.get('docid', '')))
                relevance = int(item.get('relevance', item.get('score', 1)))
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
            
            logger.info(f"Loaded {len(qrels)} qrels")
            if qrels:
                logger.info("Qrel Example: %s", list(qrels.values())[0])
        except Exception as e:
            logger.warning(f"Could not load qrels: {e}")
    
    if ignore_corpus:
        return None, queries, qrels
    
    # Load corpus
    logger.info(f"Loading Corpus from {dataset_name_corpus} (subset: {subset}, split: {corpus_split})...")
    corpus_dataset = load_dataset(dataset_name_corpus, subset, split=corpus_split)
    
    for item in corpus_dataset:
        doc_id = str(item['docid'])
        # Concatenate specified document fields
        doc_contents = []
        for field in doc_fields:
            if field in item and item[field]:
                doc_contents.append(str(item[field]))
        content_text = " ".join(doc_contents)
        corpus[doc_id] = {"contents": content_text}
    
    logger.info(f"Loaded {len(corpus)} documents")
    if corpus:
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
