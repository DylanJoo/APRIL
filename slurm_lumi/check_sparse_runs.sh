#!/bin/bash
RUNS_DIR="$HOME/APRIL/runs"
MODEL_DIR="$RUNS_DIR/Llama-3.3-70B-Instruct"
# MODEL_DIR="$RUNS_DIR/Qwen/Qwen2.5-7B-Instruct"
DATASETS=(
    "beir@arguana"
    "beir@climate-fever"
    "beir@fever/test"
    "beir@fiqa/test"
    "beir@hotpotqa/test"
    "beir@nq"
    "beir@quora/test"
    "beir@scifact/test"
)

RETRIEVERS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
RERANKING=(judge judge_expr point setmaxheaptopk rankgpt)
SEEDS=$(seq 1 5)

echo "========================================"
echo "SPARSE LLAMA RERANKING CHECK"
echo "========================================"

total=0
present=0
missing=0

for dataset in "${DATASETS[@]}"; do
    benchmark="${dataset%%@*}"
    full_subset="${dataset##*@}"
    subset_key="${full_subset%%/*}"   # e.g. arguana, climate-fever, fiqa

    for r in "${RETRIEVERS[@]}"; do
        for method in "${RERANKING[@]}"; do
            for seed in $SEEDS; do
                fname="run.${benchmark}.${r}-rerank-${method}.${subset_key}.txt"
                fpath="$MODEL_DIR/sample-${seed}/${fname}"
                total=$((total + 1))
                if [ -f "$fpath" ]; then
                    present=$((present + 1))
                else
                    echo "  MISSING: sample-${seed}/${fname}"
                    missing=$((missing + 1))
                fi
            done
        done
    done
done

echo ""
echo "Summary: $present / $total present  ($missing missing)"
echo ""
