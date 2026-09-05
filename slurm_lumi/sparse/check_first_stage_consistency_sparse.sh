#!/bin/bash
RUNS_DIR="$HOME/APRIL/runs"
MODEL_DIR="$RUNS_DIR/Llama-3.3-70B-Instruct"
SCRIPT="$HOME/APRIL/qrel-analysis/sanity_check_first_stage.py"

DATASETS=(
    "beir@arguana"
    "beir@climate-fever"
    "beir@fever"
    "beir@fiqa"
    "beir@hotpotqa"
    "beir@nq"
    "beir@quora"
    "beir@scifact"
)

RETRIEVERS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
RERANKING=(judge judge_expr point setmaxheaptopk rankgpt)
SAMPLES=$(seq 1 5)

echo "========================================"
echo "FIRST-STAGE CONSISTENCY CHECK"
echo "========================================"

all_ok=true

for dataset in "${DATASETS[@]}"; do
    benchmark="${dataset%%@*}"
    subset="${dataset##*@}"

    echo ""
    echo "--- $subset ---"

    RUN_FILES=()
    missing=0
    for seed in $SAMPLES; do
        for r in "${RETRIEVERS[@]}"; do
            for method in "${RERANKING[@]}"; do
                fpath="$MODEL_DIR/sample-${seed}/run.${benchmark}.${r}-rerank-${method}.${subset}.txt"
                if [ -f "$fpath" ]; then
                    RUN_FILES+=("$fpath")
                else
                    echo "  MISSING: sample-${seed}/run.${benchmark}.${r}-rerank-${method}.${subset}.txt"
                    missing=$((missing + 1))
                fi
            done
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
