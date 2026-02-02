#!/bin/sh
#SBATCH --job-name=demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

source ~/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
LOGDIR=log.vllm
mkdir -p $LOGDIR

MODEL=castorini/rank_zephyr_7b_v1_full
python -m autollmrerank.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.backend=vllm_dev \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_trec-dl-2019.log
