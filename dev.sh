#!/bin/sh
#SBATCH --job-name=debug
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate april

# Initialize vllm server
MODEL=Qwen/Qwen2.5-7B-Instruct
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 10240  \
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
    --config=src/reranking/configs/point.yaml \
    --llm.backend=request \
    --llm.model_name_or_path=$MODEL \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2020/judged
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2020.txt

kill $PID

# 2025-09-11: testing the reranking pipeline [OK]
# 2025-09-12: testing the request setting [OK] 
# NOTE: but the backend=request is slightly different from backend=vllm
# 2025-09-13: testing the pointwise

