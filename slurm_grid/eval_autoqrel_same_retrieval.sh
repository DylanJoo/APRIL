#!/bin/bash -l
#SBATCH --job-name=autoqrel-sr-eval
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

# Same-retrieval setting, step 3: for each <retrieval>-rerank-<judge> auto-qrel,
# score only the candidates built on that same retrieval (the retrieval alone +
# <retrieval>-rerank-*) with METRIC; also score them against the human qrels.
# Auto-qrels come from output_autoqrel_same_retrieval.sh (step 2).

source $HOME/.bashrc
initconda
conda activate autollmreranker

cd $HOME/APRIL
source slurm_grid/autoqrel_config_same_retrieval.sh

# ── Check: every candidate run and every judge's auto-qrels must exist ───────
missing=0
for retrieval in "${RETRIEVALS[@]}"; do
    set_candidates $retrieval
    for entry in "${CANDIDATES[@]}"; do
        run_path="${entry%%|*}"
        [ -s "$run_path" ] || { echo "  [missing candidate] $run_path"; missing=$((missing + 1)); }
    done
    for judge in "${JUDGES[@]}"; do
        qrel_dir=$(autoqrel_dir $retrieval $judge)
        ls "${qrel_dir}"autollmqrel.*.txt >/dev/null 2>&1 || \
            { echo "  [missing auto-qrels] ${qrel_dir} (run output_autoqrel_same_retrieval.sh first)"; missing=$((missing + 1)); }
    done
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: ${missing} input(s) missing for ${NAME}."
    exit 1
fi
echo "=== ${NAME}: ${#RETRIEVALS[@]} retrievals x ${#JUDGES[@]} judges ==="

mkdir -p "$EVAL_RESULTS_DIR"

# Writes "run  METRIC" for every candidate under the given qrels (file or irds id).
score_candidates() {
    local qrels=$1 out_file=$2
    printf "%-45s  %s\n" "run" "$METRIC" > "$out_file"
    for entry in "${CANDIDATES[@]}"; do
        run_path="${entry%%|*}"
        run_name="${entry##*|}"
        score=$(python -m ir_measures "$qrels" "$run_path" $METRIC | cut -f2)
        printf "%-45s  %s\n" "$run_name" "$score" >> "$out_file"
    done
}

# A finished eval file has 1 header + 1 row per candidate.
is_complete() { [ -f "$1" ] && [ "$(wc -l < "$1")" -eq $(( ${#CANDIDATES[@]} + 1 )) ]; }

for retrieval in "${RETRIEVALS[@]}"; do
    set_candidates $retrieval
    echo ""
    echo "=== ${retrieval}: ${#CANDIDATES[@]} candidates ==="

    # ── Candidates vs. human qrels (baseline) ────────────────────────────────
    out_file="${EVAL_RESULTS_DIR}/${retrieval}.human.txt"
    if is_complete "$out_file"; then
        echo "  [skip] already complete: $(basename "$out_file")"
    else
        score_candidates "${BENCHMARK}/${SUBSET}" "$out_file"
        echo "  Wrote: $(basename "$out_file")"
    fi

    # ── Candidates vs. each same-retrieval judge's auto-qrels ────────────────
    for judge in "${JUDGES[@]}"; do
        for qrel_file in "$(autoqrel_dir $retrieval $judge)"autollmqrel.*.txt; do
            strategy=$(basename "$qrel_file" .txt)
            strategy=${strategy#autollmqrel.}
            out_file="${EVAL_RESULTS_DIR}/${retrieval}-rerank-${judge}.autollmqrel.${strategy}.txt"
            if is_complete "$out_file"; then
                echo "  [skip] already complete: $(basename "$out_file")"
                continue
            fi
            score_candidates "$qrel_file" "$out_file"
            echo "  Wrote: $(basename "$out_file")"
        done
    done
done
