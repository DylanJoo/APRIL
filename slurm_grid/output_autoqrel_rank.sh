#!/bin/bash -l
#SBATCH --job-name=autoqrel-rank
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

# Pipeline step 2: turn each judge run into auto-qrel files, one per rank cutoff
# (rank@k: top-k of the judge run -> relevant).
# Judges, pool, cutoffs and output layout are defined in autoqrel_config_rank.sh;
# eval_autoqrel_rank.sh (step 3) scores the candidates against these files.

source $HOME/.bashrc
initconda
conda activate autollmreranker

cd $HOME/APRIL
source slurm_grid/autoqrel_config_rank.sh

# ── Check: every judge run must exist ────────────────────────────────────────
missing=0
for judge in "${JUDGES[@]}"; do
    judge_run=$(judge_run_path $judge)
    [ -s "$judge_run" ] || { echo "  [missing judge] $judge_run"; missing=$((missing + 1)); }
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: ${missing} of ${#JUDGES[@]} judge runs missing for ${NAME}."
    exit 1
fi

# ── Generate auto-qrel files from each judge, one per cutoff ─────────────────
echo "=== ${NAME}: auto-qrels from ${#JUDGES[@]} judges (rank@${RANK_CUTOFFS[*]}) ==="
strategy_args=()
for s in "${STRATEGIES[@]}"; do strategy_args+=(--strategies "$s"); done

for judge in "${JUDGES[@]}"; do
    output_dir=$(autoqrel_dir $judge)
    mkdir -p "$output_dir"
    for cutoff in "${RANK_CUTOFFS[@]}"; do
        if [ -s "${output_dir}autollmqrel.rank@${cutoff}.txt" ]; then
            echo "  [skip] already generated: ${POOL}-rerank-${judge} rank@${cutoff}"
            continue
        fi
        echo "  Generating: ${POOL}-rerank-${judge} rank@${cutoff}"
        python qrel-analysis/output_autoqrel.py \
            --dataset_name $DATASET \
            --loader_type irds \
            --judge_run $(judge_run_path $judge) \
            "${strategy_args[@]}" --rank_cutoff $cutoff \
            --output_dir $output_dir
    done
done
