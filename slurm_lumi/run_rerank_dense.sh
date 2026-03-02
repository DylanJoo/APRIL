#!/bin/bash -l
#SBATCH --job-name=dense
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=06:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/APRIL
LOGDIR=log.dense-judged
mkdir -p $LOGDIR
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

MODEL=meta-llama/Llama-3.3-70B-Instruct

## POINTWISE
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 8196 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for method in judge judge_expr point; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt \
        --llm.model_name_or_path=$MODEL > $LOGDIR/${method}.${benchmark}-${subset%%/*}.log
done
done
kill $PID

## SETWISE
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 20480 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
method=setmaxheaptopk
srun singularity exec $SIF \
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
    --llm.backend=request \
    --data.dataset_name=${benchmark}/${subset} \
    --data.input_run=${inital_run} \
    --data.output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt \
    --llm.model_name_or_path=$MODEL > $LOGDIR/${method}.${benchmark}-${subset%%/*}.log
done

## LISTWISE
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 4 > vllm_server.log 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
method=rankgpt
srun singularity exec $SIF \
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
    --llm.backend=request \
    --data.dataset_name=${benchmark}/${subset} \
    --data.input_run=${inital_run} \
    --data.output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt \
    --llm.model_name_or_path=$MODEL > $LOGDIR/${method}.${benchmark}-${subset%%/*}.log
done
