#!/bin/bash -l
#SBATCH --job-name=cost-report
#SBATCH --partition=standard-g           # partition name
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=2,3,6
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

stop_vllm_server() {
    local pid=$1
    kill "$pid" 2>/dev/null
    for i in $(seq 1 30); do
        curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || break
        sleep 5
    done
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -9 8000/tcp >/dev/null 2>&1
    fi
    for i in $(seq 1 12); do
        curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || break
        sleep 5
    done
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "ERROR: vLLM server on port 8000 would not shut down; refusing to start the next stage against a possibly stale server." >&2
        exit 1
    fi
}

# Cost-report runs write to their own directory so they never collide with,
# or get skipped because of, the production outputs from pool_rerank_dense.sh.
REPORT_DIR=runs/${MODEL##*/}/cost_report
mkdir -p $REPORT_DIR

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

# Compare running cost of every pointwise reranking/judgement method on the
# bm25 top100 pool.
POOL=bm25-top100
METHODS=(judge judge_expr point umbrela)

needs_run=false
for method in "${METHODS[@]}"; do
    output_run=${REPORT_DIR}/run.${benchmark}.${POOL}-rerank-${method}.${subset%%/*}.txt
    if [ ! -f "${output_run}.stats.json" ]; then
        needs_run=true
        break
    fi
done

if [ "$needs_run" = true ]; then
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

for method in "${METHODS[@]}"; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.bm25.${subset%%/*}.txt
    output_run=${REPORT_DIR}/run.${benchmark}.${POOL}-rerank-${method}.${subset%%/*}.txt
    if [ -f "${output_run}.stats.json" ]; then
        echo "Skipping $output_run (cost stats already exist)"
        continue
    fi
    echo "=== RUNNING: dataset=$dataset pool=$POOL method=$method ==="
    srun singularity exec $SIF \
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.batch_size=512 \
        --llm.backend=request \
        --llm.base_url=http://localhost:8000/v1 \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=${output_run} \
        --llm.model_name_or_path=$MODEL
done
stop_vllm_server $PID
else
    echo "Skipping pointwise methods (cost stats already exist)"
fi

# SETWISE
method=setmaxheaptopk
output_run=${REPORT_DIR}/run.${benchmark}.${POOL}-rerank-${method}.${subset%%/*}.txt
if [ ! -f "${output_run}.stats.json" ]; then
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

inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.bm25.${subset%%/*}.txt
echo "=== RUNNING: dataset=$dataset pool=$POOL method=$method ==="
srun singularity exec $SIF \
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
    --llm.backend=request \
    --llm.base_url=http://localhost:8000/v1 \
    --data.dataset_name=${benchmark}/${subset} \
    --data.input_run=${inital_run} \
    --data.output_run=${output_run} \
    --llm.model_name_or_path=$MODEL --max_doc_length=1024

stop_vllm_server $PID
else
    echo "Skipping $output_run (cost stats already exist)"
fi

## LISTWISE
method=rankgpt
output_run=${REPORT_DIR}/run.${benchmark}.${POOL}-rerank-${method}.${subset%%/*}.txt
if [ ! -f "${output_run}.stats.json" ]; then
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

inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.bm25.${subset%%/*}.txt
echo "=== RUNNING: dataset=$dataset pool=$POOL method=$method ==="
srun singularity exec $SIF \
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
    --llm.backend=request \
    --llm.base_url=http://localhost:8000/v1 \
    --data.dataset_name=${benchmark}/${subset} \
    --data.input_run=${inital_run} \
    --data.output_run=${output_run} \
    --llm.model_name_or_path=$MODEL --max_doc_length=1024

stop_vllm_server $PID
else
    echo "Skipping $output_run (cost stats already exist)"
fi

# Every wrapper.py run above already wrote its own <output_run>.stats.json
# (num_calls / prompt_words / completion_words / time_sec) as a side effect
# of inference -- this just collects them into one table for this dataset.
python slurm_lumi/aggregate_cost_report.py \
    --dir ${REPORT_DIR} \
    --benchmark ${benchmark} \
    --subset ${subset%%/*}
