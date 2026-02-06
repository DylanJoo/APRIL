#!/bin/sh
#SBATCH --job-name=judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.out

source ~/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
LOGDIR=log.vllm.new/pointwise_judge
mkdir -p $LOGDIR

# Point:binary_prob:Qwen/Qwen2.5-7B-Instruct
for year in 2019 2020;do
    MODEL=Qwen/Qwen2.5-7B-Instruct

    # Pointwise YES NO
    python -m autollmrerank.wrapper \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.model_name_or_path=$MODEL \
        --llm.max_model_len=8196 \
        --llm.use_logits=true \
        --rerank_mode=Point \
        --dtype=float16 \
        --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
        --result_parser_name=binary_probability > $LOGDIR/point_trec-dl-${year}.log

    # Judge
    python -m autollmrerank.wrapper \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.model_name_or_path=$MODEL \
        --llm.max_model_len=8196 \
        --rerank_mode=Judge \
        --dtype=float16 \
        --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
        --result_parser_name=text > $LOGDIR/judge_trec-dl-${year}.log

    # Judge with max-rating logP
    python -m autollmrerank.wrapper \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.model_name_or_path=$MODEL \
        --llm.max_model_len=8196 \
        --llm.use_logits=true \
        --rerank_mode=Judge \
        --dtype=float16 \
        --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
        --result_parser_name=rating_logp > $LOGDIR/judge_logp_trec-dl-${year}.log

    # Judge with expected rating
    python -m autollmrerank.wrapper \
        --data.dataset_name=msmarco-passage/trec-dl-${year}/judged \
        --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-${year}.txt \
        --llm.model_name_or_path=$MODEL \
        --llm.max_model_len=8196 \
        --llm.use_logits=true \
        --rerank_mode=Judge \
        --dtype=float16 \
        --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
        --result_parser_name=expected_rating > $LOGDIR/judge_exprating_trec-dl-${year}.log
done
