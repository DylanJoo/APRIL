import argparse
import json
import glob
import os
import pandas as pd

def parse_result(results_dir: str) -> pd.DataFrame:
    rows = []

    for file in sorted(glob.glob(results_dir + "/*.jsonl")):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                exp = json.loads(line.strip())

                dataset = exp["dataset"].split('/')[1]
                judge, prediction = exp["exp"].split(":")
                strategy = exp["strategy"]
                ndcg = exp["nDCG@10"]
                is_qrel_eval = strategy.endswith(":qrel")
                base_strategy = strategy.removesuffix(":qrel")

                ## other metadata of qrel
                judge_retriever, judge_reranker = judge.split("-rerank-")
                pred_retriever, pred_reranker = prediction.split("-rerank-")

                rows.append({
                    "dataset":         dataset,
                    "strategy":        base_strategy,
                    "judge_retriever": judge_retriever,
                    "judge_reranker":  judge_reranker,
                    "pred_retriever":  pred_retriever,
                    "pred_reranker":   pred_reranker,
                    "is_qrel_eval":    is_qrel_eval,
                    "ndcg10":          ndcg,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the result JSONL files.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the aggregated results as a CSV file.")
    args = parser.parse_args()
    df = parse_result(args.results_dir)
    df.to_csv(args.output_csv, index=False)
