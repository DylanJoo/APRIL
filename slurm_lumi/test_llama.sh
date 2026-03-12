#!/bin/bash -l
#SBATCH --job-name=sparse
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --account=project_465002438
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/APRIL

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

MODEL=meta-llama/Llama-3.3-70B-Instruct
server_log=vllm_server.log.tmp

## POINTWISE
needs_pointwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for method in judge judge_expr point; do
for seed in $(seq 1 10);do
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_pointwise=true
        break 3
    fi
done
done
done

if [ "$needs_pointwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > $server_log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 10); do
for method in point judge judge_expr; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    echo "=== RUNNING: dataset=$dataset r=$r seed=$seed method=$method ==="
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper_sample \
        --sampling=true --sampling_size=32 --sampling_seed=$seed \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
done
done
fi

kill $PID
