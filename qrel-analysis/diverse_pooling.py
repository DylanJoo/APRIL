import argparse
import os
import random
from collections import defaultdict


def parse_trec_run(filepath, topk):
    """Return {qid: [(docid, rank), ...]} keeping only top-k per query."""
    results = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank = parts[0], parts[1], parts[2], int(parts[3])
            results[qid].append((docid, rank))
    return {qid: sorted(docs, key=lambda x: x[1])[:topk] for qid, docs in results.items()}


def rrf_fuse(system_results, rrf_k=60):
    """
    system_results: list of {qid: [(docid, rank), ...]}
    Returns {qid: [(docid, rrf_score), ...]} sorted by score descending.
    """
    all_qids = set()
    for r in system_results:
        all_qids.update(r.keys())

    fused = {}
    counts = []
    for qid in all_qids:
        scores = defaultdict(float)
        for i, r in enumerate(system_results):
            for docid, rank in r.get(qid, []):
                scores[docid] += 1.0 / (rrf_k + rank)
            if i == len(system_results) - 1:
                counts.append(len(scores))
        fused[qid] = sorted(scores.items(), key=lambda x: -x[1])
    print("Average pool size", sum(counts) / len(counts))
    return fused


def write_trec_run(fused, filepath, run_name="diverse-pool"):
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(filepath, "w") as f:
        for qid in sorted(fused.keys()):
            for rank, (docid, score) in enumerate(fused[qid], start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_files", nargs="+", required=True, help="TREC run files to fuse")
    parser.add_argument("--output", required=True, help="Output file path for the fused run")
    parser.add_argument("--topk", type=int, default=10, help="Top-K docs per system per query")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF smoothing constant")
    parser.add_argument("--seed", type=int, default=None, help="If set, randomly shuffle the RRF ranking with this seed")
    args = parser.parse_args()

    print(f"Fusing {len(args.run_files)} systems (top-{args.topk} each) ...")
    for fp in args.run_files:
        print(f"  {fp}")
    system_results = [parse_trec_run(fp, args.topk) for fp in args.run_files]
    fused = rrf_fuse(system_results, rrf_k=args.rrf_k)

    if args.seed is not None:
        rng = random.Random(args.seed)
        for qid in fused:
            rng.shuffle(fused[qid])
        print(f"Shuffled ranking with seed={args.seed}")

    write_trec_run(fused, args.output)
    total_docs = sum(len(v) for v in fused.values())
    print(f"-> {args.output}  ({len(fused)} queries, {total_docs} total docs)")


if __name__ == "__main__":
    main()
