#!/bin/bash
RUNS_DIR="$HOME/APRIL/runs"
MODEL_DIR="$RUNS_DIR/Llama-3.3-70B-Instruct"
SCRIPT="$HOME/APRIL/qrel-analysis/sanity_check_first_stage.py"

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
RERANKING=(judge judge_expr point setmaxheaptopk rankgpt)

echo "========================================"
echo "FIRST-STAGE CONSISTENCY CHECK"
echo "========================================"

all_ok=true

for dataset in "${DATASETS[@]}"; do
    benchmark="${dataset%%@*}"
    subset="${dataset##*@}"
    subset_short="${subset%%/*}"

    echo ""
    echo "--- $subset_short ---"

    RUN_FILES=()
    for r in "${RETRIEVERS[@]}"; do
        for method in "${RERANKING[@]}"; do
            fpath="$MODEL_DIR/run.${benchmark}.${r}-rerank-${method}.${subset_short}.txt"
            if [ -f "$fpath" ]; then
                RUN_FILES+=("$fpath")
            else
                echo "  MISSING: run.${benchmark}.${r}-rerank-${method}.${subset_short}.txt"
            fi
        done
    done

    if [ ${#RUN_FILES[@]} -eq 0 ]; then
        echo "  No run files found — skipping."
        continue
    fi

    result=$(python3 "$SCRIPT" --topk 100 --run_files "${RUN_FILES[@]}" 2>&1 \
        | grep -E "(OK |FAIL|All first-stage|Could not detect)")

    echo "$result"
    if echo "$result" | grep -q "FAIL"; then
        all_ok=false
    fi
done

echo ""
echo "========================================"
if $all_ok; then
    echo "All datasets: first-stage consistent."
else
    echo "FAILURES detected — see above."
fi
echo "========================================"
