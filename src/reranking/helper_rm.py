from collections import defaultdict, OrderedDict
from reranking.utils import Result

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def convert_run_to_result(run, queries=None, corpus=None):
    results = []
    for qid, hits in run.items():
        query = queries[qid]
        hits = []
        for docid, score in hits.items():
            hits.append({'docid': docid, 'score': float(score), 'content': corpus[docid]['contents']})
        results.append(Result(qid=qid, query=query, hits=hits))
    return results
