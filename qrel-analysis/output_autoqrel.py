"""
output_autoqrel.py

Output auto-generated qrel files in TREC format from an LLM judge run,
using the same thresholding strategies as eval_autoqrels.py.

TREC qrel format: <qid> 0 <docid> <relevance>

Examples
--------
# All strategies, one file per strategy:
python qrel-analysis/output_autoqrel.py \
    --dataset_name msmarco-passage/trec-dl-2019/judged \
    --loader_type irds \
    --judge_run runs/Llama-3.3-70B-Instruct/run.msmarco-passage.qwen3-embed-600m-rerank-setmaxheaptopk.trec-dl-2019.txt \
    --strategies all \
    --output_dir qrel-analysis/autoqrels/qwen3-embed-600m-rerank-setmaxheaptopk/trec-dl-2019/

# Single strategy with a specific parameter:
python qrel-analysis/output_autoqrel.py \
    --judge_run ... \
    --strategies rank --rank_cutoff 20 \
    --output_dir ...
"""

import argparse
import importlib
import os
import sys

from eval_autoqrels import AutoQrel


def write_qrel_trec(qrel: dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for qid in sorted(qrel, key=lambda x: str(x)):
            for docid, rel in sorted(qrel[qid].items()):
                f.write(f"{qid}\t0\t{docid}\t{rel}\n")


def strategy_label(strategy: str, args) -> str:
    if strategy == "thresholding":
        return f"thresholding@{args.threshold}"
    elif strategy == "rank":
        return f"rank@{args.rank_cutoff}"
    elif strategy == "largest_gap":
        return f"largest_gap@{args.gap_k}"
    elif strategy == "quantile_binary":
        return f"quantile_binary@{args.quantile_cutoff}"
    elif strategy == "quantile_bucket":
        return f"quantile_bucket"
    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write LLM-judge-derived qrels in TREC format."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--loader_type", type=str, default="irds")
    parser.add_argument("--judge_run", type=str, required=True)

    parser.add_argument("--strategies", action="append", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rank_cutoff", type=int, default=10)
    parser.add_argument("--gap_k", type=int, default=1)
    parser.add_argument("--quantile_cutoff", type=float, default=0.75)
    parser.add_argument("--min_relevance", type=int, default=1)

    # Output: either a directory (one file per strategy) or a single file
    # (only valid when exactly one strategy is requested)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write one qrel file per strategy.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Single output file (only when one strategy is requested).",
    )
    args = parser.parse_args()

    if args.output_dir is None and args.output is None:
        parser.error("Provide --output_dir (multi-strategy) or --output (single file).")

    loader = importlib.import_module(f"autollmrerank.loader_dev.{args.loader_type}")
    _, _, human_qrel = loader.load(args.dataset_name)
    judge_run = loader.load_run(args.judge_run)

    strategies_requested = args.strategies or ["all"]

    autoqrel = AutoQrel(
        qrel=human_qrel,
        judge_run=judge_run,
        strategies=strategies_requested,
        threshold=args.threshold,
        rank_cutoff=args.rank_cutoff,
        gap_k=args.gap_k,
        quantile_cutoff=args.quantile_cutoff,
        min_relevance=args.min_relevance,
    )

    if args.output and len(autoqrel.llm_qrels) > 1:
        sys.exit(
            f"--output accepts exactly one strategy; got {list(autoqrel.llm_qrels)}. "
            "Use --output_dir instead."
        )

    for strategy, qrel in autoqrel.llm_qrels.items():
        label = strategy_label(strategy, args)
        if args.output:
            path = args.output
        else:
            path = os.path.join(args.output_dir, f"autollmqrel.{label}.txt")

        write_qrel_trec(qrel, path)

        n_queries = len(qrel)
        n_docs = sum(len(v) for v in qrel.values())
        n_relevant = sum(1 for docs in qrel.values() for r in docs.values() if r > 0)
        print(
            f"[{label}] queries={n_queries}  judged_docs={n_docs}  relevant={n_relevant}"
            f"  -> {path}"
        )
