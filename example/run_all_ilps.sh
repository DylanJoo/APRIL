#!/bin/sh
#SBATCH --job-name=autorerank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.out

source ~/.bashrc
initconda
conda activate autollmrerank

LOGDIR=log.request
mkdir -p $LOGDIR

# 1. Initialize vllm server
MODEL=Qwen/Qwen2.5-7B-Instruct
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8196  \
    --port 8000  \
    --dtype float16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 1 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for year in 2019 2020;do
    # common method
    for method in judge point pairtopk rankgpt setmaxheaptopk;do
        python -m autollmrerank.wrapper \
            --config=src/autollmrerank/configs/$method.yaml \
            --llm.backend=request \
            --llm.model_name_or_path=$MODEL \
            --data.dataset_name=msmarco-passage/trec-dl-$year/judged \
            --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-$year.txt > $LOGDIR/${method}_trec-dl-${year}.log 2>&1
    done

    # SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
    python -m autollmrerank.wrapper \
        --config=src/autollmrerank/configs/setmaxheaptopk.yaml \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --rerank_mode=SetTopK > $LOGDIR/settop10_trec-dl-$year.log 2>&1

    # PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
    python -m autollmrerank.wrapper \
        --config=src/autollmrerank/configs/pairtopk.yaml \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --rerank_mode=PairAll \
        --score_aggregation=symsum > $LOGDIR/pairall_trec-dl-$year.log 2>&1

done
kill $PID

# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8196  \
    --port 8000  \
    --dtype float16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 1 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
for year in 2019 2020;do
    python -m autollmrerank.wrapper \
        --config=src/autollmrerank/configs/rankgpt.yaml \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL > $LOGDIR/rankzephyr_trec-dl-${year}.log
done
kill $PID

# RankFirst:dist_logp:castorini/first_mistral
MODEL=castorini/first_mistral
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8196  \
    --port 8000  \
    --dtype float16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 1 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done
echo "vLLM server is up and running."

for year in 2019 2020;do
    python -m autollmrerank.wrapper \
        --config=src/autollmrerank/configs/rankgpt.yaml \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --llm.use_logits=true \
        --rerank_mode=RankFirst \
        --use_alphabetical=true \
        --result_parser_name=distribution_logp > $LOGDIR/rankfirst_trec-dl-${year}.log 2>&1
done
kill $PID
