import argparse
import importlib
import os
import sys

import ir_measures
import pandas as pd
from ir_measures import nDCG

from eval_judge_qrel import AutoQrel

METHODS = ["point", "judge", "judge_expr", "setmaxheaptopk", "rankgpt"]
DEFAULT_STRATEGIES = ["direct", "binarize", "rank_cutoff"]


def main():
    parser = argparse.ArgumentParser(
        description="Build a qrel x run nDCG@10 matrix for a given dataset and retriever."
    )
    parser.add_argument("--dataset_name", required=True, help="e.g. beir/nfcorpus/test")
    parser.add_argument("--retriever", required=True, help="e.g. bm25")
    parser.add_argument("--runs_dir", required=True, help="directory containing reranked run files")
    parser.add_argument("--base_runs_dir", default=None, help="directory containing baseline (pre-reranking) runs")
    parser.add_argument("--strategies", action="append", default=None, choices=AutoQrel.STRATEGIES, help="auto-qrel strategies (repeatable); defaults to direct binarize rank_cutoff")
    parser.add_argument("--loader_type", default="irds")
    parser.add_argument("--output", default=None, help="path to save output CSV")
    args = parser.parse_args()

    strategies = args.strategies or DEFAULT_STRATEGIES

    loader = importlib.import_module(f"autollmrerank.loader_dev.{args.loader_type}")

    # Parse names for file lookup
    parts = args.dataset_name.split("/")
    benchmark = parts[0]
    subset_name = parts[1]  # first path segment, e.g. "nfcorpus", "trec-dl-2019"

    # Load human qrel
    _, _, human_qrel = loader.load(args.dataset_name)

    # --- Collect evaluation runs (matrix columns) ---
    all_eval_runs = {}

    # Baseline (pre-reranking)
    if args.base_runs_dir:
        path = os.path.join(
            args.base_runs_dir, benchmark,
            f"run.{benchmark}.{args.retriever}.{subset_name}.txt"
        )
        if os.path.exists(path):
            all_eval_runs[args.retriever] = loader.load_run(path)
        else:
            print(f"[warn] baseline not found: {path}", file=sys.stderr)

    # Reranked runs
    reranker_runs = {}
    for method in METHODS:
        path = os.path.join(
            args.runs_dir,
            f"run.{benchmark}.{args.retriever}-rerank-{method}.{subset_name}.txt"
        )
        if os.path.exists(path):
            run = loader.load_run(path)
            name = f"{args.retriever}-rerank-{method}"
            all_eval_runs[name] = run
            reranker_runs[name] = run
        else:
            print(f"[warn] run not found: {path}", file=sys.stderr)

    # Human qrel converted to a run (reference ceiling)
    tiebreak_run = next(iter(reranker_runs.values()), None)
    all_eval_runs["human_qrel_as_run"] = AutoQrel.qrel_to_run(human_qrel, tiebreak_run)

    # --- Build qrel sources (matrix rows) ---
    qrel_sources = {"human": human_qrel}

    for run_name, run in reranker_runs.items():
        aq = AutoQrel(qrel=human_qrel, judge_run=run, strategies=strategies)
        for strategy, llm_qrel in aq.llm_qrels.items():
            qrel_sources[f"{run_name}[{strategy}]"] = llm_qrel

    # --- Fill matrix ---
    metric = nDCG @ 10
    rows = {}
    for qrel_name, qrel in qrel_sources.items():
        row = {}
        for run_name, run in all_eval_runs.items():
            score = ir_measures.calc_aggregate([metric], qrel, run)[metric]
            row[run_name] = round(score, 4)
        rows[qrel_name] = row

    df = pd.DataFrame(rows).T
    df.index.name = "qrel \\ run"

    print(df.to_string())

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output)
        print(f"\n[info] saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
