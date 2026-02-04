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

# Pointwise YES NO
dataset=crux-mds-duc04
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.batch_size=128 \
    --data.ir_datasets_name=null \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=cruxmds \
    --data.input_diversity_qrel=$CRUX_ROOT/$dataset/qrels/div_qrels-tau3.txt \
    --data.input_ratings=$CRUX_ROOT/$dataset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=Point \
    --dtype=float16 \
    --result_parser_name=binary_probability 
    # > $LOGDIR/point_beir-${dataset%%/*}.log

    # # Judge
    # python -m autollmrerank.wrapper \
    #     --data.batch_size=128 \
    #     --data.ir_datasets_name=beir/${subset} \
    #     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    #     --llm.model_name_or_path=$MODEL \
    #     --llm.max_model_len=8196 \
    #     --rerank_mode=Judge \
    #     --dtype=float16 \
    #     --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    #     --result_parser_name=text > $LOGDIR/judge_beir-${subset%%/*}.log
    #
    # # Judge with max-rating logP
    # python -m autollmrerank.wrapper \
    #     --data.batch_size=128 \
    #     --data.ir_datasets_name=beir/${subset} \
    #     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    #     --llm.model_name_or_path=$MODEL \
    #     --llm.max_model_len=8196 \
    #     --llm.use_logits=true \
    #     --rerank_mode=Judge \
    #     --dtype=float16 \
    #     --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    #     --result_parser_name=rating_logp > $LOGDIR/judge_logp_beir-${subset%%/*}.log
    #
    # # Judge with expected rating
    # python -m autollmrerank.wrapper \
    #     --data.batch_size=128 \
    #     --data.ir_datasets_name=beir/${subset} \
    #     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    #     --llm.model_name_or_path=$MODEL \
    #     --llm.max_model_len=8196 \
    #     --llm.use_logits=true \
    #     --rerank_mode=Judge \
    #     --dtype=float16 \
    #     --system_message "You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    #     --result_parser_name=expected_rating > $LOGDIR/judge_expr_beir-${subset%%/*}.log
