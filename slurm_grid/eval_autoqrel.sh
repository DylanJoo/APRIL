#!/bin/bash -l
#SBATCH --job-name=autoqrel-eval
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

# Pipeline step 3: score every candidate with METRIC against the human qrels
# and against each judge's auto-qrels (written by output_autoqrel.sh, step 2).
# Judges, candidates and output layout are defined in autoqrel_config.sh.

source $HOME/.bashrc
initconda
conda activate autollmreranker

cd $HOME/APRIL
source slurm_grid/autoqrel_config.sh

# ── Check: every candidate run and every judge's auto-qrels must exist ───────
missing=0
for entry in "${CANDIDATES[@]}"; do
    run_path="${entry%%|*}"
    [ -s "$run_path" ] || { echo "  [missing candidate] $run_path"; missing=$((missing + 1)); }
done
for judge in "${JUDGES[@]}"; do
    qrel_dir=$(autoqrel_dir $judge)
    ls "${qrel_dir}"autollmqrel.*.txt >/dev/null 2>&1 || \
        { echo "  [missing auto-qrels] ${qrel_dir} (run output_autoqrel.sh first)"; missing=$((missing + 1)); }
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: ${missing} input(s) missing for ${NAME} (${#CANDIDATES[@]} candidates, ${#JUDGES[@]} judges expected)."
    exit 1
fi
echo "=== ${NAME}: ${#CANDIDATES[@]} candidates x ${#JUDGES[@]} judges ==="

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
expected_lines=$(( ${#CANDIDATES[@]} + 1 ))
is_complete() { [ -f "$1" ] && [ "$(wc -l < "$1")" -eq "$expected_lines" ]; }

# ── Candidates vs. human qrels (baseline) ────────────────────────────────────
echo ""
echo "=== ${METRIC} — human qrels ==="
out_file="${EVAL_RESULTS_DIR}/human.txt"
if is_complete "$out_file"; then
    echo "  [skip] already complete: $(basename "$out_file")"
else
    score_candidates "${BENCHMARK}/${SUBSET}" "$out_file"
    echo "  Wrote: $(basename "$out_file")"
fi

# ── Candidates vs. each judge's auto-qrels ───────────────────────────────────
echo ""
echo "=== ${METRIC} — auto-qrels ==="
for judge in "${JUDGES[@]}"; do
    for qrel_file in "$(autoqrel_dir $judge)"autollmqrel.*.txt; do
        strategy=$(basename "$qrel_file" .txt)
        strategy=${strategy#autollmqrel.}
        out_file="${EVAL_RESULTS_DIR}/${POOL}-rerank-${judge}.autollmqrel.${strategy}.txt"
        if is_complete "$out_file"; then
            echo "  [skip] already complete: $(basename "$out_file")"
            continue
        fi
        score_candidates "$qrel_file" "$out_file"
        echo "  Wrote: $(basename "$out_file")"
    done
done
