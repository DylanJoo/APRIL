#!/bin/bash -l
#SBATCH --job-name=autoqrel-sr
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=4
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

# Same-retrieval setting, step 2: turn each <retrieval>-rerank-<judge> run into
# auto-qrel files. Config in autoqrel_config_same_retrieval.sh;
# eval_autoqrel_same_retrieval.sh (step 3) scores the candidates against them.

source $HOME/.bashrc
initconda
conda activate autollmreranker

cd $HOME/APRIL
source slurm_grid/autoqrel_config_same_retrieval.sh

# ── Check: every judge run must exist ────────────────────────────────────────
missing=0
for retrieval in "${RETRIEVALS[@]}"; do
for judge in "${JUDGES[@]}"; do
    judge_run=$(judge_run_path $retrieval $judge)
    [ -s "$judge_run" ] || { echo "  [missing judge] $judge_run"; missing=$((missing + 1)); }
done
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: ${missing} of $(( ${#RETRIEVALS[@]} * ${#JUDGES[@]} )) judge runs missing for ${NAME}."
    exit 1
fi

# ── Generate auto-qrel files from each judge ─────────────────────────────────
echo "=== ${NAME}: auto-qrels from ${#RETRIEVALS[@]} retrievals x ${#JUDGES[@]} judges (strategies: ${STRATEGIES[*]}) ==="
strategy_args=()
for s in "${STRATEGIES[@]}"; do strategy_args+=(--strategies "$s"); done

# A judge is done when every requested strategy has its file
# (autollmqrel.<name>.txt or autollmqrel.<name>@<param>.txt); "all" checks direct.
all_generated() {
    local dir=$1 s
    for s in "${STRATEGIES[@]}"; do
        [ "$s" = all ] && s=direct
        compgen -G "${dir}autollmqrel.${s}.txt" >/dev/null || \
            compgen -G "${dir}autollmqrel.${s}@*.txt" >/dev/null || return 1
    done
}

for retrieval in "${RETRIEVALS[@]}"; do
for judge in "${JUDGES[@]}"; do
    output_dir=$(autoqrel_dir $retrieval $judge)
    if all_generated "$output_dir"; then
        echo "  [skip] already generated: ${retrieval}-rerank-${judge}"
        continue
    fi
    mkdir -p "$output_dir"
    echo "  Generating: ${retrieval}-rerank-${judge}"
    python qrel-analysis/output_autoqrel.py \
        --dataset_name $DATASET \
        --loader_type irds \
        --judge_run $(judge_run_path $retrieval $judge) \
        "${strategy_args[@]}" \
        --output_dir $output_dir
done
done
