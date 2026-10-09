#!/bin/bash -l
#SBATCH --job-name=autoqrel-test
#SBATCH --partition=standard-g
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00
#SBATCH --account=project_465002438
#SBATCH --array=0-6
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

# Pipeline step 2: turn each judge run into auto-qrel files.
# Judges, pool and output layout are defined in autoqrel_config.sh;
# eval_autoqrel.sh (step 3) scores the candidates against these files.

cd $HOME/APRIL
source slurm_lumi/autoqrel_config.sh

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

# ── Generate auto-qrel files from each judge ─────────────────────────────────
echo "=== ${NAME}: auto-qrels from ${#JUDGES[@]} judges ==="
strategy_args=()
for s in "${STRATEGIES[@]}"; do strategy_args+=(--strategies "$s"); done

for judge in "${JUDGES[@]}"; do
    output_dir=$(autoqrel_dir $judge)
    if [ -f "${output_dir}autollmqrel.direct.txt" ]; then
        echo "  [skip] already generated: ${POOL}-rerank-${judge}"
        continue
    fi
    mkdir -p "$output_dir"
    echo "  Generating: ${POOL}-rerank-${judge}"
    srun singularity exec $SIF python qrel-analysis/output_autoqrel.py \
        --dataset_name $DATASET \
        --loader_type irds \
        --judge_run $(judge_run_path $judge) \
        "${strategy_args[@]}" \
        --output_dir $output_dir
done
