#!/bin/bash -l
#SBATCH --job-name=dense-qwen3
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --time=72:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module --force purge
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
export HIP_VISIBLE_DEVICES=0,1,2,3

cd $HOME/APRIL
MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
mkdir -p runs/${MODEL##*/}

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
benchmark=$(echo $dataset | cut -d'@' -f1)
subset=$(echo $dataset | cut -d'@' -f2)

## POINTWISE
srun singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --max-model-len 10240 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > vllm_server_qwen3.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for method in judge judge_expr point; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF_QWEN \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
done
kill $PID

## SETWISE
srun singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --max-model-len 20480 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server_qwen3.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

method=setmaxheaptopk
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF_QWEN \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done

## LISTWISE
srun singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server_qwen3.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

method=rankgpt
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF_QWEN \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
