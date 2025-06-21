from collections import defaultdict, OrderedDict
import collections

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_runs(path, topk=None, output_score=False): 
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
        if output_score:
            sorted_run_dict[qid] = {docid: rel_score for docid, rel_rank, rel_score in sorted_docid_ranks}
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict

"""
run: {
    "<qid1>": {"<docid>": score, "<docid2>": score, ...}, 
    "<qid2>": ...
}
results: [
    Result{query=<query1>, hits=[{"docid": <docid>, "score": <score>, "content": <content>},...], 
    Result{query=<query2>, hits=[...]}
]
"""
def convert_run_to_result(run, queries=None, corpus=None):
    results = []
    for qid, hits in run.items():
        query = queries[qid]
        pairs = []
        for docid, score in hits.items():
            pairs.append({'docid': docid, 'score': float(score), 'content': corpus[docid]['contents']})
        results.append(Result(qid=qid, query=query, hits=pairs))
    return results

