import re
import math
from tqdm import tqdm
from typing import List
from reranking.utils import batch_iterator
import logging
from itertools import combinations
from ftfy import fix_text

logger = logging.getLogger(__name__)


def rerank(
    model: str,
    run: dict, queries: dict, corpus: dict,
    batch_size: int = 128,
    **kwargs,
):

    template = kwargs.pop('template', None)

    # prompt preparation
    id_pairs, prompts = [], []
    for qid in run:
        query = queries[qid]
        candidates = [docid for docid in run[qid]]
        pairs = [(i, j) for i in range(len(candidates)) for j in range(len(candidates)) if i != j]
        for i, j in pairs:
            prompts.append(
                    fix_text(template.format(
                    cand1=" ".join(corpus[candidates[i]]["contents"].split()), 
                    cand2=" ".join(corpus[candidates[j]]["contents"].split()),
                    query=query)
            ))
            id_pairs.append((qid, i, j))

    with open('rerank_prompts-old.txt', 'w') as f:
        for p in prompts:
            f.write(f"{p}\n")

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
        batch_scores = model.generate(batch_prompts, prob=True)

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
