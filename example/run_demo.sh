#!/bin/sh
#SBATCH --job-name=test
#SBATCH --partition gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate autollmreranker

LOGDIR=log.vllm.new
mkdir -p $LOGDIR
# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
python -m autollmrerank.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_trec-dl-2019.log
