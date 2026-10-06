#!/bin/bash -l
#SBATCH --job-name=pool
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --account=project_465002438
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1 
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/APRIL
MODEL=meta-llama/Llama-3.3-70B-Instruct
LOG=vllm_server.log
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

## 
POOL=pool-55-systems-top10
# Pools can exceed 100 docs/query (pool-55 top10: up to 179, top20: up to 255), but
# load_run (data.topk) and the pointwise assemblers (rank_end/top_k) default to 100.
# Raise all three so every pooled doc is judged (setwise heap only reads data.topk).
# Listwise: rank_end must stay a multiple of step_size (10) or the [0:20] window is
# skipped; empty windows beyond a query's hits are skipped in SlidingWindow.run_pass.
POOL_TOPK=1000

## POINTWISE
needs_pointwise=false
for r in $POOL; do
for method in judge judge_expr point umbrela; do
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
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for r in $POOL;do
for method in judge judge_expr point umbrela; do
    inital_run=$HOME/runs-and-qrels/runs-pool-rankjudge/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.batch_size=512 \
        --data.topk=${POOL_TOPK} \
        --top_k=${POOL_TOPK} \
        --rank_end=${POOL_TOPK} \
        --llm.backend=request \
        --llm.base_url=http://localhost:8000/v1 \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
done
kill $PID
sleep 5
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
waited=0
until ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || [ $waited -ge 60 ]; do
    sleep 5
    waited=$((waited + 5))
done
fi

## SETWISE
needs_setwise=false
for method in setmaxheaptopk umbrela_setmaxheaptopk; do
for r in $POOL; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_setwise=true
        break
    fi
done
done

echo $needs_setwise

if [ "$needs_setwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port 8000 \
    --enforce-eager \
    --max-model-len 20480 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for method in setmaxheaptopk umbrela_setmaxheaptopk; do
for r in $POOL;do
    inital_run=$HOME/runs-and-qrels/runs-pool-rankjudge/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.topk=${POOL_TOPK} \
        --top_k=${POOL_TOPK} \
        --rank_end=${POOL_TOPK} \
        --llm.backend=request \
        --llm.base_url=http://localhost:8000/v1 \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL --max_doc_length=1024
done
done
kill $PID
until ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 5; done
fi

## LISTWISE
needs_listwise=false
for method in rankgpt umbrela_rankgpt; do
for r in $POOL; do
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "$output_run" ]; then
        needs_listwise=true
        break
    fi
done
done
echo $needs_listwise

if [ "$needs_listwise" = true ]; then
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 30720 \
    --port 8000 \
    --enforce-eager \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for method in rankgpt umbrela_rankgpt; do
for r in $POOL;do
    inital_run=$HOME/runs-and-qrels/runs-pool-rankjudge/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt
    if [ -f "$output_run" ]; then
        echo "Skipping $output_run (already exists)"
        continue
    fi
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.topk=${POOL_TOPK} \
        --top_k=${POOL_TOPK} \
        --rank_end=${POOL_TOPK} \
        --llm.backend=request \
        --llm.base_url=http://localhost:8000/v1 \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL --max_doc_length=1024
done
done
kill $PID
fi
