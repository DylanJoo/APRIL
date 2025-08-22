#!/bin/sh
#SBATCH --job-name=dl20-nrun
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

for num_runs in 1 2 3 4; do

echo "Run number: $num_runs"
# RankGPT
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2020/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2020.txt \
    --llm.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=$num_runs \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > logs/rankgpt_trec-dl-2020_${num_runs}.log

# RankZepyhr
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2020/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2020.txt \
    --llm.model_name_or_path=castorini/rank_zephyr_7b_v1_full \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=$num_runs \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > logs/rankzephyr_trec-dl-2020_${num_runs}.log

# RankFirst
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2020/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2020.txt \
    --llm.model_name_or_path=castorini/first_mistral \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=RankFirst \
    --num_runs=$num_runs \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > logs/rankfirst_trec-dl-2020_${num_runs}.log

done
