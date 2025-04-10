import re
from typing import List
from utils.tools import batch_iterator

def preprocess(query, doc):
    template = \
    """Passage: {doc}\nQuery: {query}\nIs this passage relevant to the query? Please answer 'true' or 'false'. Answer: 
    """.strip()
    return x

def postprocess(x):
    x = x.lower()
    pattern = re.findall('true')
    pattern = re.findall('false')
    return rating

def rerank(
    model: str,
    run: dict, 
    queries: dict, 
    corpus: dict,
    batch_size: int = 16,
    **kwargs,
):
    tokenizer = model.model.get_tokenizer()

    # prompt preparation
    id_pairs, prompts = [], []
    for qid in run:
        for docid in run[qid]:
            prompts.append(postprocess(doc=corpus[docid]["contents"], query=queries[qid]))
            id_pairs.append((qid, docid))

    # batch inference
    scores = []
    for start, end in batch_iterator(prompts, size=batch_size, return_index=True):
        batch_prompts = prompts[start:end]
        batch_query_lengths = query_lengths[start:end]

        batch_outputs = model.generate(batch_prompts, max_tokens=5)
        # batch_scores = postprocess(batch_outputs)
        scores += batch_scores

    # sorting
    reranked_run = {}
    for i in range(len(scores)):
        qid, docid, _ = id_pairs[i]
        reranked_run[qid][docid] = scores[i]

    return reranked_run
