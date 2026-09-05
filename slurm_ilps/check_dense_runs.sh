#!/bin/bash
RUNS_DIR="$HOME/APRIL/runs"
DATASETS=(
    "msmarco-passage@trec-dl-2019/judged"
    "msmarco-passage@trec-dl-2020/judged"
    "beir@dbpedia-entity/test"
    "beir@nfcorpus/test"
    "beir@scidocs"
    "beir@trec-covid"
    "beir@webis-touche2020/v2"
)

RETRIEVERS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
MODEL_DIR="$RUNS_DIR/Qwen2.5-7B-Instruct"
RERANKING=(judge judge_expr point setmaxheaptopk rankgpt)
# MODEL_DIR="$RUNS_DIR/supervised"
# RERANKING=(rankzephyr rankfirst)

echo "========================================"
echo "RERANKING: $1"
echo "========================================"

total=0
present=0
missing=0

for dataset in "${DATASETS[@]}"; do
    benchmark="${dataset%%@*}"
    full_subset="${dataset##*@}"
    subset_key="${full_subset%%/*}"   # e.g. trec-dl-2019, dbpedia-entity

    for r in "${RETRIEVERS[@]}"; do
        for method in "${RERANKING[@]}"; do
            fname="run.${benchmark}.${r}-rerank-${method}.${subset_key}.txt"
            fpath="$MODEL_DIR/$fname"
            total=$((total + 1))
            if [ -f "$fpath" ]; then
                present=$((present + 1))
            else
                echo "  MISSING: $fpath"
                missing=$((missing + 1))
            fi
        done
    done
done

echo ""
echo "Summary: $present / $total present  ($missing missing)"
echo ""
