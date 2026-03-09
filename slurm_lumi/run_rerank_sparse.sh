#!/bin/bash -l
#SBATCH --job-name=sparse
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --array=2
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0

cd $HOME/APRIL
mkdir -p runs/${MODEL##*/}

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

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
benchmark=$(echo $dataset | cut -d'@' -f1)
subset=$(echo $dataset | cut -d'@' -f2)

MODEL=Qwen/Qwen2.5-7B-Instruct
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 10); do
for method in point judge judge_expr setmaxheaptopk rankgpt; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    echo "=== RUNNING: dataset=$dataset r=$r seed=$seed method=$method ==="
    python -m autollmrerank.wrapper_sample \
        --sampling=true --sampling_size=32 --sampling_seed=$seed \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
done
done
