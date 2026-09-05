from typing import List
import argparse
import ir_measures
from ir_measures import nDCG
import ir_datasets
from collections import defaultdict, OrderedDict

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrapping for preliminary evaluation")
    parser.add_argument("--irds_tag", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # filtered qrels
    run = load_run(args.path)
    qrels = {}
    for qrel in ir_datasets.load(args.irds_tag).qrels_iter():
        if qrel.query_id in run:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {qrel.doc_id: qrel.relevance}
            else:
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance


    # evaluation
    r = ir_measures.calc_aggregate([nDCG@10], qrels, run)[nDCG@10]
    print(r)
