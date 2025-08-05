#!/bin/sh
#SBATCH --job-name=llmrerank
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# RankGPT
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=listwise

# RankZepyhr
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=listwise
