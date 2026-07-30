#!/bin/bash -l
#SBATCH --job-name=sparse-llama
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0-7
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=72:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/APRIL
MODEL=meta-llama/Llama-3.3-70B-Instruct
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

## POINTWISE
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 10); do
for method in judge judge_expr point; do
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
kill $PID

## SETWISE
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 20480 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

method=setmaxheaptopk
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 10); do
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

## LISTWISE
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

method=rankgpt
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 10); do
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
