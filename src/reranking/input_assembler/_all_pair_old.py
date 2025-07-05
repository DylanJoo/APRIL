import re
import math
from tqdm import tqdm
from typing import List
from utils.tools import batch_iterator
import logging
from itertools import combinations
from llm import hf_back, vllm_back, litellm_api

logger = logging.getLogger(__name__)

system_prompt = \
"""You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."""
user_prompt = \
"""I will provide you with two passages. Read and memorize both carefully. Your task is to determine which passage is more relevant to the query.

Query: {query}

Passage 1: {cand1}
Passage 2: {cand2}

Based on the given query, is the Passage 1 more relevant than Passage 2?
Please answer 'Yes' or 'No'.
Answer: """
template = user_prompt

def rerank(
    model: str,
    run: dict, queries: dict, corpus: dict,
    batch_size: int = 128,
    **kwargs,
):
    # prompt preparation
    id_pairs, prompts = [], []
    for qid in run:
        query = queries[qid]
        candidates = [docid for docid in run[qid]]
        pairs = [(i, j) for i in range(len(candidates)) for j in range(len(candidates)) if i != j]
        for i, j in pairs:
            prompts.append(template.format(
                cand1=corpus[candidates[i]]["contents"], 
                cand2=corpus[candidates[j]]["contents"], 
                query=query
            ))
            id_pairs.append((qid, i, j))

    # token identifier
    tokenizer = model.tokenizer
    true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
    false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']

    # batch inference
    logger.info('Number of prompts: {len(prompts)}')
    scores = []
    for start, end in tqdm(
        batch_iterator(prompts, size=batch_size, return_index=True),
        total=len(prompts) // batch_size + 1
    ):
        batch_prompts = prompts[start:end]

        # [TODO] put them togethe. loading all llm classes is unecessary.
        model.set_classification(true_list, false_list)
        batch_scores = model.inference_chat(system_prompt, batch_prompts)

        scores += batch_scores

    # aggregate scores
    all_scores = {}
    for qid in run:
        all_scores[qid] = [0 for _ in range(len(run[qid]))]

    for (qid, i, j), score in zip(id_pairs, scores):
        all_scores[qid][i] += score
        all_scores[qid][j] += (1 - score)

    # update pairwise scores
    reranked_run = {}
    for qid in all_scores:
        docids = [k for k in run[qid]]
        for i, s in enumerate(all_scores[qid]):
            if qid not in reranked_run:
                reranked_run[qid] = {docids[i]: s}
            else:
                reranked_run[qid][docids[i]] = s

    # sorting
    sorted_run_dict = {}
    for qid, hit in reranked_run.items():
        sorted_hit = sorted(hit.items(), key=lambda x: x[1], reverse=True) 
        sorted_run_dict[qid] = {docid: rel_score for docid, rel_score in sorted_hit}

    return sorted_run_dict
