#!/bin/bash -l
#SBATCH --job-name=dense
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=2,3,6
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --account=project_465002532
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/APRIL
MODEL=meta-llama/Llama-3.3-70B-Instruct
LOG=vllm_server.log.c
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
needs_pointwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    for method in judge judge_expr point; do
        output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
        if [ ! -f "$output_run" ]; then
            needs_pointwise=true
            break 2
        fi
    done
done

if [ "$needs_pointwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --enforce-eager \
    --max-model-len 10240 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for method in judge judge_expr point; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
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
fi

## SETWISE
method=setmaxheaptopk
needs_setwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_setwise=true
        break
    fi
done

if [ "$needs_setwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --enforce-eager \
    --max-model-len 20480 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
kill $PID
fi

## LISTWISE
method=rankgpt
needs_listwise=false
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_listwise=true
        break
    fi
done

if [ "$needs_listwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --port 8000 \
    --enforce-eager \
    --dtype bfloat16 \
    --tensor-parallel-size 4 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --llm.backend=request \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
kill $PID
fi
