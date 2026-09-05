"""
ndcg_table.py

nDCG@10 of each candidate run (bm25, splade-v3, ..., and each retriever's
rerankers) evaluated against different judgment sources (LLM-judge-derived
auto-qrels), for the top10-pool (pool-40-systems-top10) results already
computed under eval_results/. This does not re-run any evaluation; it
parses the precomputed "run \t nDCG@10" files that eval_autoqrels.py /
output_autoqrel.py produced, one per (dataset, judge method, strategy)
combination:

    eval_results/{dataset}/pool-40-systems-top10-rerank-{method}.autollmqrel.{strategy}.txt

One table is printed per dataset: rows are candidate runs, columns are
judge methods, cells are that candidate's nDCG@10 under that judge
method's auto-qrel. No averaging across candidates.

Usage
-----
# Default: rank@10 strategy, all 6 judge methods, all candidates, one table per dataset.
python ndcg_table.py

# Just the 5 base (non-reranked) retrievers.
python ndcg_table.py --base-only

# A specific set of candidates.
python ndcg_table.py --runs bm25 splade-v3

# Different strategy, narrower dataset sweep.
python ndcg_table.py --strategy thresholding@0.5 --datasets nfcorpus scidocs

# JSON output.
python ndcg_table.py --json
"""

import argparse
import json
import os

DEFAULT_EVAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

DEFAULT_DATASETS = ["dbpedia-entity", "nfcorpus", "scidocs", "trec-covid", "webis-touche2020",
                     "trec-dl-2019", "trec-dl-2020"]
DEFAULT_METHODS = ["judge", "judge_expr", "point", "rankgpt", "setmaxheaptopk", "umbrela"]
DEFAULT_POOL = "pool-40-systems-top10"
DEFAULT_STRATEGY = "rank@10"

BASE_RUNS = ["bm25", "splade-v3", "nomicai-modernbert-embed", "qwen3-embed-600m", "colbert-small"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_eval_file(root, dataset, pool, method, strategy):
    path = os.path.join(root, dataset, f"{pool}-rerank-{method}.autollmqrel.{strategy}.txt")
    return path if os.path.isfile(path) else None


def load_ndcg_file(path):
    """Parse a 'run \\t nDCG@10' file (header row + one row per run),
    preserving the run order as it appears in the file."""
    scores = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 2:
            continue
        run, ndcg = parts[0], parts[-1]
        try:
            scores[run] = float(ndcg)
        except ValueError:
            continue
    return scores


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def load_dataset_table(args, dataset):
    """Returns (candidate_order, {run: {method: ndcg10}}) for one dataset."""
    candidate_order = []
    table = {}
    for method in args.methods:
        path = find_eval_file(args.eval_root, dataset, args.pool, method, args.strategy)
        if path is None:
            continue
        scores = load_ndcg_file(path)
        for run in scores:
            if run not in table:
                table[run] = {}
                candidate_order.append(run)
        for run, score in scores.items():
            table[run][method] = score

    if args.runs:
        candidate_order = [r for r in args.runs if r in table]
    elif args.base_only:
        candidate_order = [r for r in BASE_RUNS if r in table]

    return candidate_order, table


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(v):
    return "-" if v is None else "{:.4f}".format(v)


def print_dataset_table(dataset, candidate_order, table, methods):
    print(f"\n=== {dataset} ===")
    if not candidate_order:
        print("  No candidates found.")
        return

    run_w = max(18, max((len(r) for r in candidate_order), default=0) + 2)
    col_w = max(12, max((len(m) for m in methods), default=0) + 2)
    header = "  " + "{:<{}}".format("run", run_w) + \
        "".join("{:>{}}".format(m, col_w) for m in methods)
    print(header)
    print("-" * len(header))
    for run in candidate_order:
        cells = [_fmt(table[run].get(m)) for m in methods]
        print("  " + "{:<{}}".format(run, run_w) +
              "".join("{:>{}}".format(c, col_w) for c in cells))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-candidate nDCG@10 across judgment sources, using precomputed top10-pool eval_results/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--eval-root", type=str, default=DEFAULT_EVAL_ROOT,
                         help="Root dir containing {dataset}/pool-40-systems-top10-rerank-{method}.autollmqrel.{strategy}.txt files.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--pool", type=str, default=DEFAULT_POOL,
                         help="Pool tag in the eval_results filenames (default: top10 pool).")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY,
                         help="Auto-qrel strategy label, e.g. rank@10, thresholding@0.5, direct.")
    parser.add_argument("--runs", nargs="+", default=None,
                         help="Only show these candidate runs (default: all found).")
    parser.add_argument("--base-only", action="store_true",
                         help=f"Only show the 5 non-reranked base retrievers ({', '.join(BASE_RUNS)}).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = {}
    for dataset in args.datasets:
        candidate_order, table = load_dataset_table(args, dataset)
        results[dataset] = (candidate_order, table)

    if args.json:
        out = {
            dataset: {run: table[run] for run in candidate_order}
            for dataset, (candidate_order, table) in results.items()
        }
        print(json.dumps(out, indent=2))
    else:
        for dataset in args.datasets:
            candidate_order, table = results[dataset]
            print_dataset_table(dataset, candidate_order, table, args.methods)


if __name__ == "__main__":
    main()
