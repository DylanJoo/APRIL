#!/bin/bash -l
#SBATCH --job-name=autoqrel-test
#SBATCH --partition=small
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --account=project_465002532
#SBATCH --array=1-6
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

cd $HOME/APRIL

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
BENCHMARK=$(echo $dataset | cut -d'@' -f1)
SUBSET=$(echo $dataset | cut -d'@' -f2)
NAME=${SUBSET%%/*}
DATASET=${BENCHMARK}/${SUBSET}
MODEL_DIR=Llama-3.3-70B-Instruct

RETRIEVALS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
RERANKERS=(judge judge_expr point rankgpt setmaxheaptopk)

# All eval runs: retrieval + reranking
EVAL_RUNS=()
for r in "${RETRIEVALS[@]}"; do
    EVAL_RUNS+=("${HOME}/runs-and-qrels/runs/${BENCHMARK}/run.${BENCHMARK}.${r}.${NAME}.txt|${r}")
done
for r1 in "${RETRIEVALS[@]}"; do
for r2 in "${RERANKERS[@]}"; do
    EVAL_RUNS+=("${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
done
for r2 in rankfirst rankzephyr;do
    EVAL_RUNS+=("${HOME}/APRIL/runs/supervised/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt|${r1}-rerank-${r2}")
done
done

# ── Step 1: generate auto-qrel files ──────────────────────────────────────────
echo "=== Generating auto-qrel files for ${NAME} ==="
for r1 in pool-40-systems-top10;do
for r2 in "${RERANKERS[@]}"; do
    judge_run=${HOME}/APRIL/runs/${MODEL_DIR}/run.${BENCHMARK}.${r1}-rerank-${r2}.${NAME}.txt
    if [ ! -f "$judge_run" ]; then
        echo "  [skip] not found: $judge_run"
        continue
    fi
    output_dir=${HOME}/APRIL/qrel-analysis/autoqrels/${r1}-rerank-${r2}/${NAME}/
    mkdir -p "$output_dir"
    echo "  Generating: ${r1}-rerank-${r2}"
    srun singularity exec $SIF python qrel-analysis/output_autoqrel.py \
        --dataset_name $DATASET \
        --loader_type irds \
        --judge_run $judge_run \
        --strategies all \
        --output_dir $output_dir
done
done

# ── Step 2: nDCG@10 with human qrel (baseline) ────────────────────────────────
echo ""
echo "=== nDCG@10 — human qrel baseline ==="
printf "%-45s  %s\n" "run" "nDCG@10"
for entry in "${EVAL_RUNS[@]}"; do
    run_path="${entry%%|*}"
    run_name="${entry##*|}"
    [ -f "$run_path" ] || { echo "  [skip] not found: $run_name"; continue; }
    ndcg=$(srun singularity exec $SIF python -m ir_measures ${BENCHMARK}/${SUBSET} "$run_path" nDCG@10 | cut -f2)
    printf "%-45s  %s\n" "$run_name" "$ndcg"
done

# ── Step 3: nDCG@10 with each auto-qrel ───────────────────────────────────────
echo ""
echo "=== nDCG@10 — auto-qrels ==="

EVAL_RESULTS_DIR=${HOME}/APRIL/qrel-analysis/eval_results/${NAME}
mkdir -p "$EVAL_RESULTS_DIR"

for r1 in pool-40-systems-top10;do
for r2 in "${RERANKERS[@]}"; do
    qrel_dir=${HOME}/APRIL/qrel-analysis/autoqrels/${r1}-rerank-${r2}/${NAME}/
    for qrel_file in "${qrel_dir}"autollmqrel.*.txt; do
        [ -f "$qrel_file" ] || continue
        strategy=$(basename "$qrel_file" .txt | sed 's/^qrel\.//')
        out_file="${EVAL_RESULTS_DIR}/${r1}-rerank-${r2}.${strategy}.txt"
        echo "  Writing: $(basename "$out_file")"
        printf "%-45s  %s\n" "run" "nDCG@10" > "$out_file"
        for entry in "${EVAL_RUNS[@]}"; do
            run_path="${entry%%|*}"
            run_name="${entry##*|}"
            [ -f "$run_path" ] || continue
            ndcg=$(srun singularity exec $SIF python -m ir_measures "$qrel_file" "$run_path" nDCG@10 | cut -f2)
            printf "%-45s  %s\n" "$run_name" "$ndcg" >> "$out_file"
        done
    done
done
done
