#!/bin/sh
#SBATCH --job-name=crux-rerank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.out

source ${HOME}/.bashrc
initconda
conda activate autollmrerank

LOGDIR=log.vllm.new
mkdir -p $LOGDIR

dataset=crux-mds-duc04

# Pointwise YES NO
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --llm.use_logits=true \
    --rerank_mode=Point \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/point_crux-mds-duc04.log

# LANCER
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --num_runs=2 \
    --rerank_mode=Lancer \
    --dtype=float16 \
    --result_parser_name=text > $LOGDIR/lancer_crux-mds-duc04.log

# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_crux-mds-duc04.log

# RankFirst:dist_logp:castorini/first_mistral
MODEL=castorini/first_mistral
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=RankFirst \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/rankfirst_crux-mds-duc04.log

# RankGPT:list_gen:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankgpt_crux-mds-duc04.log

# SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=SetTopK \
    --num_runs=10 \
    --window_size=5 --step_size=1 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/settop10_crux-mds-duc04.log

# SetMaxHeapTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=SetMaxHeapTopK \
    --num_runs=10 \
    --window_size=5 --step_size=1 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/setmaxheaptop10_crux-mds-duc04.log

# PairTopK:binary_prob:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=PairTopK \
    --num_runs=10 \
    --window_size=2 --step_size=1 \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/pairtop10_crux-mds-duc04.log

# PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.batch_size=128 \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=cruxmds \
#     --data.input_diversity_qrels=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
#     --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=8196 \
#     --llm.use_logits=true \
#     --rerank_mode=PairAll \
#     --score_aggregation=symsum \
#     --dtype=float16 \
#     --result_parser_name=binary_probability > $LOGDIR/pairall_crux-mds-duc04.log
