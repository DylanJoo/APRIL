#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --time=02:00:00             # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002532 # Project for billing
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

module --force purge
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ${HOME}/.bashrc
cd ${HOME}/temp

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
singularity exec $SIF_QWEN \
    python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --disable-custom-all-reduce \
    --disable-log-stats \
    --enforce-eager \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 > ${HOME}/APRIL/vllm_server_qwen.log 2>&1 & 
PID=$!

until curl -sf http://localhost:8000/v1/models >/dev/null; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') waiting... server not up yet"
  sleep 10
done
echo "$(date '+%Y-%m-%d %H:%M:%S') start running"
