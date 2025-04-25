import re
import math
from tqdm import tqdm
from typing import List
from utils.tools import batch_iterator
import logging

logger = logging.getLogger(__name__)

# template of tokens
# [('No', 2822), ('NO', 9173), ('no', 2201), ('ĠNo', 2360), (Ġno', 912)] 
# [('Yes', 9642), ('YES', 14331), ('yes', 9891), ('ĠYes', 7566), ('Ġyes', 10035)] 

# template of prompts. Recommend to use ':' as the end of the prompt token. It is more stable.
template = "Passage: {doc}\nQuery: {query}\nIs this passage relevant to the query?\nPlease answer 'Yes' or 'No'.\nAnswer: "
# template = "### Instruction:\nDetermine whether the passage is relevant to the given query. Answer only with 'Yes' or 'No'.\n\n" + \
#         "### Input:\nPassage: {doc}\nQuery: {query}\n\n" + \
#         "### Response:\n"

def extract_scores(
    batch_logits, 
    yes_tokens, 
    no_tokens
):
    scores = []
    for logits in batch_logits: # (B, L, N)
        yes_ = math.exp(max( [logits[-1, i] for i in yes_tokens] ))
        no_ = math.exp(max( [logits[-1, i] for i in no_tokens] ))
        scores.append( (yes_) / (no_ + yes_) )
    return scores

def rerank(
    model: str,
    run: dict, 
    queries: dict, 
    corpus: dict,
    batch_size: int = 16,
    **kwargs,
):

    # prompt preparation
    id_pairs, prompts = [], []
    for qid in run:
        for docid in run[qid]:
            prompts.append(template.format(doc=corpus[docid]["contents"], query=queries[qid]))
            id_pairs.append((qid, docid))

    # token identifier
    tokenizer = model.tokenizer
    true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
    false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
    yes_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in true_list]
    no_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in false_list]

    # batch inference
    logger.info('Number of prompts: {len(prompts)}')
    scores = []
    for start, end in tqdm(
        batch_iterator(prompts, size=batch_size, return_index=True),
        total=len(prompts) // batch_size + 1
    ):
        batch_prompts = prompts[start:end]

        batch_logits = model.inference(batch_prompts)
        batch_scores = extract_scores(batch_logits, yes_tokens, no_tokens)
        scores += batch_scores

    # update scores
    reranked_run = {}
    for i in range(len(scores)):
        qid, docid = id_pairs[i]
        if qid not in reranked_run:
            reranked_run[qid] = {docid: scores[i]}
        else:
            reranked_run[qid][docid] = scores[i]

    # sorting
    sorted_run_dict = {}
    for qid, hit in reranked_run.items():
        sorted_hit = sorted(hit.items(), key=lambda x: x[1], reverse=True) 
        sorted_run_dict[qid] = {docid: rel_score for docid, rel_score in sorted_hit}

    return sorted_run_dict
