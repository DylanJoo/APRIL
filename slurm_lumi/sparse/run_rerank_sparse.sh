#!/bin/bash -l
#SBATCH --job-name=sparse
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --account=project_465002438
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err
# transformers==4.46.0

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/APRIL
MODEL=meta-llama/Llama-3.3-70B-Instruct
LOG=vllm_server.log
# mkdir -p runs/${MODEL##*/}

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
needs_pointwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for method in judge judge_expr point; do
for seed in $(seq 1 5);do
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
    --enforce-eager \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!

START_TIME=$(date +%s)
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -ge 1200 ]; then
    echo "Timeout: vLLM server did not start within 20 minutes. Exiting."
    kill $PID
    exit 1
  fi
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 5); do
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
        --data.batch_size=128 \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL --max_doc_length=4000
done
done
done
kill $PID
fi

## SETWISE
method=setmaxheaptopk
needs_setwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for seed in $(seq 1 5);do
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_setwise=true
        break 3
    fi
done
done

if [ "$needs_setwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --enforce-eager \
    --max-model-len 20480 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
START_TIME=$(date +%s)
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -ge 1200 ]; then
    echo "Timeout: vLLM server did not start within 20 minutes. Exiting."
    kill $PID
    exit 1
  fi
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 5); do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
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
kill $PID
fi

## LISTWISE
method=rankgpt
needs_listwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for seed in $(seq 1 5);do
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_listwise=true
        break 3
    fi
done
done

if [ "$needs_listwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --port 8000 \
    --enforce-eager \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
START_TIME=$(date +%s)
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
  if [ $(( $(date +%s) - START_TIME )) -ge 1200 ]; then
    echo "Timeout: vLLM server did not start within 20 minutes. Exiting."
    kill $PID
    exit 1
  fi
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for seed in $(seq 1 5); do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/sample-$seed/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
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
kill $PID
fi
