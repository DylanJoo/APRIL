from copy import deepcopy
from typing import List

class Result:

    def __init__(
        self,
        qid: str,
        query: str,
        hits = None,
        ranking_exec_summary = None,
    ):
        self.qid = qid
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

        self.subquestions = []

    def __repr__(self):
        return str(self.__dict__)

    def sort_by(self, field: str = 'score'):
        hits = deepcopy(self.hits)
        hits.sort(key=lambda x: x[field], reverse=True)
        for i, hit in enumerate(hits):
            hit['rank'] = i + 1
        self.hits = hits
        return hits

    # NOTE: not an ideal way to handle subresults 
    # TODO: think about judging path of subquestion and document pair.
    def append_subresult(self, subquestion, result):
        # hi: [{'docid': docid, 'score': float(score), 'content_dict': corpus[docid]}, ...]

        # first convert to input hit into dict format
        hits_dict = {}
        for hit in result.hits:
            hits_dict[hit['docid']] = hit['score']

        # directly add the scores
        self.subquestions.append(subquestion)
        for hit in self.hits:
            docid = hit['docid']
            hit['score'] += hits_dict.get(docid, 0.0)

        self.sort_by('score')

    def reset(self):
        for hit in self.hits:
            hit['score'] = 0.0
            hit['rank'] = -1

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]
