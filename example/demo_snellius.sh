#!/bin/sh
#SBATCH --job-name=dl19-all
#SBATCH --partition=gpu_a100
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.out
#SBATCH --output=%x.err

source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
module load CUDA/12.6.0/
conda activate rerank

cd $HOME/APRIL
LOGDIR=log.vllm
mkdir -p $LOGDIR

MODEL=castorini/rank_zephyr_7b_v1_full
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.backend=vllm_dev \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=bfloat16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_trec-dl-2019.log
