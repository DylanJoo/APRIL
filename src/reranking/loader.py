import csv
import json
import logging
import os
import ir_datasets
from collections import defaultdict, OrderedDict
from typing import Optional 

logger = logging.getLogger(__name__)

def load_hf(
    queries_dataset_name: str,
    corpus_dataset_name: str,
    subset: str,
    query_split: str = 'test',
    corpus_split: str = 'train',
    qrels_split: Optional[str] = None,
    ignore_corpus: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """
    Load queries and corpus from HuggingFace datasets.
    
    Args:
        queries_dataset_name: HuggingFace dataset name for queries (e.g., 'DylanJHJ/nano-beir')
        corpus_dataset_name: HuggingFace dataset name for corpus (e.g., 'DylanJHJ/nano-beir-corpus')
        subset: Dataset subset/config name (e.g., 'nq', 'msmarco')
        query_split: Split name for queries dataset (default: 'test')
        corpus_split: Split name for corpus dataset (default: 'train')
        qrels_split: Split name for qrels in queries dataset (default: None, uses query_split)
        ignore_corpus: If True, skip loading corpus
        
    Returns:
        tuple: (corpus, queries, qrels) where:
            - corpus: Dict[str, Dict[str, str]] mapping docid to {"contents": text}
            - queries: Dict[str, str] mapping query_id to query text
            - qrels: Dict[str, Dict[str, int]] mapping query_id to {docid: relevance}
            
    Expected query dataset fields: 'query_id', 'query_texts'
    Expected corpus dataset fields: 'docid', 'title', 'text'
    """
    from datasets import load_dataset
    
    corpus, queries, qrels = {}, {}, {}
    
    # Load queries
    logger.info(f"Loading Queries from HuggingFace: {queries_dataset_name}/{subset}...")
    try:
        queries_dataset = load_dataset(queries_dataset_name, subset, split=query_split)
        
        for item in queries_dataset:
            query_id = str(item['query_id'])
            query_text = item['query_texts']
            queries[query_id] = query_text
            
            # Check if qrels are included in the query dataset
            if 'positive_passages' in item and item['positive_passages']:
                if query_id not in qrels:
                    qrels[query_id] = {}
                # Handle positive passages (relevant documents)
                for doc_info in item['positive_passages']:
                    if isinstance(doc_info, dict) and 'docid' in doc_info:
                        doc_id = str(doc_info['docid'])
                        qrels[query_id][doc_id] = 1
                    elif isinstance(doc_info, str):
                        qrels[query_id][doc_info] = 1
            
            # Alternative field names for qrels
            if 'relevant_docs' in item and item['relevant_docs']:
                if query_id not in qrels:
                    qrels[query_id] = {}
                for doc_id in item['relevant_docs']:
                    qrels[query_id][str(doc_id)] = 1
                    
        logger.info(f"Loaded {len(queries)} queries")
        if len(queries) > 0:
            logger.info("Query Example: %s", list(queries.values())[0])
        if qrels:
            logger.info(f"Loaded qrels for {len(qrels)} queries")
    except Exception as e:
        logger.error(f"Error loading queries from HuggingFace: {e}")
        raise
    
    # Try to load qrels from separate split if specified and not already loaded
    if qrels_split and not qrels:
        logger.info(f"Loading Qrels from separate split: {qrels_split}...")
        try:
            qrels_dataset = load_dataset(queries_dataset_name, subset, split=qrels_split)
            for item in qrels_dataset:
                query_id = str(item.get('query_id', item.get('qid', '')))
                doc_id = str(item.get('docid', item.get('doc_id', '')))
                relevance = int(item.get('relevance', item.get('score', 1)))
                
                if query_id and doc_id:
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance
            logger.info(f"Loaded qrels for {len(qrels)} queries from {qrels_split} split")
        except Exception as e:
            logger.warning(f"Could not load qrels from separate split: {e}")
    
    if not qrels:
        logger.info("No qrels found in dataset. Returning empty qrels dict.")
    
    if ignore_corpus:
        return None, queries, qrels
    
    # Load corpus
    logger.info(f"Loading Corpus from HuggingFace: {corpus_dataset_name}/{subset}...")
    try:
        corpus_dataset = load_dataset(corpus_dataset_name, subset, split=corpus_split)
        
        for item in corpus_dataset:
            doc_id = str(item['docid'])
            title = item.get('title', '')
            text = item.get('text', '')
            
            # Combine title and text similar to how ir_datasets does it
            if title and text:
                contents = f"{title} {text}"
            elif title:
                contents = title
            else:
                contents = text
                
            corpus[doc_id] = {"contents": contents}
            
        logger.info(f"Loaded {len(corpus)} documents")
        if len(corpus) > 0:
            logger.info("Doc Example: %s", list(corpus.values())[0])
    except Exception as e:
        logger.error(f"Error loading corpus from HuggingFace: {e}")
        raise
    
    return corpus, queries, qrels

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
