#!/bin/bash -l
#SBATCH --job-name=rerank
#SBATCH --output=rerank.out
#SBATCH --error=rerank.err
#SBATCH --partition=small-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd ${HOME}/APRIL

LOGDIR=log.request.lumi
mkdir -p $LOGDIR

# Initialize vllm server on NVIDIA
MODEL=Qwen/Qwen2.5-7B-Instruct
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 8196 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for year in 2019 2020;do
    # common method
    for method in point pairtopk rankgpt setmaxheaptopk;do
        srun singularity exec $SIF \
            python3 -m reranking.wrapper \
            --config=src/reranking/configs/$method.yaml \
            --llm.backend=request \
            --llm.model_name_or_path=$MODEL \
            --data.ir_datasets_name=msmarco-passage/trec-dl-$year/judged \
            --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-$year.txt > $LOGDIR/${method}_trec-dl-${year}.log 2>&1
    done

    # SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
    srun singularity exec $SIF \
        python -m reranking.wrapper \
        --config=src/reranking/configs/setmaxheaptopk.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --rerank_mode=SetTopK > $LOGDIR/settop10_trec-dl-$year.log 2>&1

    # PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
    srun singularity exec $SIF \
        python -m reranking.wrapper \
        --config=src/reranking/configs/pairtopk.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --rerank_mode=PairAll \
        --score_aggregation=symsum > $LOGDIR/pairall_trec-dl-$year.log 2>&1

done
kill $PID

RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 8196 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
for year in 2019 2020;do
    srun singularity exec $SIF \
    python -m reranking.wrapper \
        --config=src/reranking/configs/rankgpt.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL > $LOGDIR/rankzephyr_trec-dl-${year}.log
done
kill $PID

# RankFirst:dist_logp:castorini/first_mistral
MODEL=castorini/first_mistral
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --max-model-len 8196 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for year in 2019 2020;do
    srun singularity exec $SIF \
    python3 -m reranking.wrapper \
        --config=src/reranking/configs/rankgpt.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --llm.use_logits=true \
        --rerank_mode=RankFirst \
        --use_alphabetical=true \
        --result_parser_name=distribution_logp > $LOGDIR/rankfirst_trec-dl-${year}.log 2>&1
done
kill $PID
