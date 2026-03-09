import argparse
import json
import glob
import os
import pandas as pd


def parse_result(results_base_dir: str) -> pd.DataFrame:
    rows = []

    for file in sorted(glob.glob(results_base_dir + "/*/raaj*.jsonl")):
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = json.loads(line)
                except:
                    print(f"Skipping malformed line in {file}: {line}")
                    continue

                dataset = exp["dataset"].split('/')[1]
                judge, prediction = exp["exp"].split(":")
                strategy = exp["strategy"].removesuffix(":qrel")
                ndcg = exp["nDCG@10"]

                judge_retriever, judge_reranker = judge.split("-rerank-")

                prediction = prediction.replace("-rerank-", "+")

                rows.append({
                    "dataset":         dataset,
                    "strategy":        strategy,
                    "judge_retriever": judge_retriever,
                    "judge_reranker":  judge_reranker,
                    "prediction":      prediction,
                    "ndcg10":          ndcg,
                })

    return pd.DataFrame(rows)


def save_grouped_csvs(df: pd.DataFrame, output_dir: str):
    """Save one CSV per pred_retriever, pivoted so judge_reranker types are columns.

    Rows: (dataset, pred_reranker, strategy)
    Columns: (judge_retriever, judge_reranker) — one per qrel source
    """
    os.makedirs(output_dir, exist_ok=True)

    for retriever, group in df.groupby("judge_retriever"):
        pivot = group.pivot_table(
            index=["dataset", "prediction", "strategy", "judge_reranker"],
            values="ndcg10",
            aggfunc="sum"
        )

        pivot = pivot.reset_index()
        out_path = os.path.join(output_dir, f"{retriever}.csv")
        pivot.to_csv(out_path, index=False, sep="|")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Base directory containing per-judge result subdirs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save one CSV per retrieval system.")
    args = parser.parse_args()

    df = parse_result(args.results_dir)
    save_grouped_csvs(df, args.output_dir)
