#!/bin/sh
#SBATCH --job-name=autorerank
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate autollmreranker

LOGDIR=log.request
mkdir -p $LOGDIR

# Initialize vllm server
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
    for method in point pairtopk rankgpt setmaxheaptopk;do
        python -m reranking.wrapper \
            --config=src/reranking/configs/$method.yaml \
            --llm.backend=request \
            --llm.model_name_or_path=$MODEL \
            --data.ir_datasets_name=msmarco-passage/trec-dl-$year/judged \
            --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-$year.txt > $LOGDIR/point_trec-dl-${year}.log 2>&1
    done

    # SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
    python -m reranking.wrapper \
        --config=src/reranking/configs/setmaxheaptopk.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.backend=request \
        --llm.model_name_or_path=$MODEL \
        --rerank_mode=SetTopK > $LOGDIR/settop10_trec-dl-$year.log 2>&1

    # PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
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
    python -m reranking.wrapper \
        --config=src/reranking/configs/rankgpt.yaml \
        --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
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

python -m reranking.wrapper \
    --config=src/reranking/configs/rankgpt.yaml \
    --data.ir_datasets_name=msmarco-passage/trec-dl-${year}/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=RankFirst \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/rankfirst_trec-dl-${year}.log 2>&1
kill $PID
