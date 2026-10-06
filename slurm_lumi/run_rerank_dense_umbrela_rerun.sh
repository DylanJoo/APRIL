#!/bin/bash -l
#SBATCH --job-name=u_rankjudge_rerun
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=40:00:00
#SBATCH --account=project_465002532
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
LOG=vllm_server.${SLURM_ARRAY_TASK_ID}.log
mkdir -p runs/${MODEL##*/}

# Only the (benchmark, subset, reranker, method) combos that were left missing
# after run_rerank_dense_umbrela.sh -- either never attempted (setwise/pointwise
# were disabled) or killed by walltime (scidocs listwise).
BENCHMARKS=(
beir
beir
beir
beir
beir
beir
beir
beir
beir
beir
)
SUBSETS=(
scidocs
scidocs
scidocs
scidocs
scidocs
scidocs
scidocs
scidocs
scidocs
nfcorpus/test
)
RERANKERS=(
colbert-small
bm25
splade-v3
nomicai-modernbert-embed
qwen3-embed-600m
colbert-small
nomicai-modernbert-embed
qwen3-embed-600m
colbert-small
colbert-small
)
METHODS=(
umbrela
umbrela_setmaxheaptopk
umbrela_setmaxheaptopk
umbrela_setmaxheaptopk
umbrela_setmaxheaptopk
umbrela_setmaxheaptopk
umbrela_rankgpt
umbrela_rankgpt
umbrela_rankgpt
umbrela_setmaxheaptopk
)

benchmark=${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}
subset=${SUBSETS[$SLURM_ARRAY_TASK_ID]}
r=${RERANKERS[$SLURM_ARRAY_TASK_ID]}
method=${METHODS[$SLURM_ARRAY_TASK_ID]}

case $method in
    umbrela)
        PORT=8000
        MAXLEN=10240
        EXTRA_ARGS="--data.batch_size=512"
        ;;
    umbrela_setmaxheaptopk)
        PORT=8001
        MAXLEN=20480
        EXTRA_ARGS=""
        ;;
    umbrela_rankgpt)
        PORT=8002
        MAXLEN=30720
        EXTRA_ARGS=""
        ;;
esac

inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
output_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt

if [ -f "$output_run" ]; then
    echo "Skipping $output_run (already exists)"
    exit 0
fi

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --enforce-eager \
    --max-model-len $MAXLEN \
    --dtype bfloat16 \
    --tensor-parallel-size 8 > $LOG 2>&1 &
PID=$!
until curl -s http://localhost:$PORT/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

srun singularity exec $SIF \
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
    $EXTRA_ARGS \
    --llm.backend=request \
    --llm.base_url=http://localhost:$PORT/v1 \
    --data.dataset_name=${benchmark}/${subset} \
    --data.input_run=${inital_run} \
    --data.output_run=${output_run} \
    --llm.model_name_or_path=$MODEL

kill $PID
sleep 5
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
waited=0
until ! curl -s http://localhost:$PORT/v1/models >/dev/null 2>&1 || [ $waited -ge 60 ]; do
    sleep 5
    waited=$((waited + 5))
done
