#!/bin/sh
#SBATCH --job-name=judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=11
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%a.out

source ~/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
LOGDIR=log.vllm.new/pointwise_judge
mkdir -p $LOGDIR

BEIR_DATASETS=(
"arguana"
"climate-fever"
"dbpedia-entity/test"
"fever/test"
"fiqa/test"
"hotpotqa/test"
"nfcorpus/test"
"nq"
"quora/test"
"scidocs"
"scifact/test"
"trec-covid"
"webis-touche2020/v2"
)

# Point:binary_prob:Qwen/Qwen2.5-7B-Instruct
subset=${BEIR_DATASETS[$SLURM_ARRAY_ID]}
MODEL=Qwen/Qwen2.5-7B-Instruct

# Pointwise YES NO
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/point.yaml \
    --data.batch_size=128 \
    --data.dataset_name=beir/${subset} \
    --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    --llm.model_name_or_path=$MODEL \
    --system_message="You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --result_parser_name=binary_probability > $LOGDIR/point_beir-${subset%%/*}.log

# Judge
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/judge.yaml \
    --data.batch_size=128 \
    --data.dataset_name=beir/${subset} \
    --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    --llm.use_logits=true \
    --llm.model_name_or_path=$MODEL \
    --system_message="You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --result_parser_name=text > $LOGDIR/judge_beir-${subset%%/*}.log

# Judge with max-rating logP
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/judge.yaml \
    --data.batch_size=128 \
    --data.dataset_name=beir/${subset} \
    --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    --data.output_run=runs/Dev/judge-rating-logp.txt \
    --llm.use_logits=true \
    --llm.model_name_or_path=$MODEL \
    --system_message="You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --result_parser_name=rating_logp > $LOGDIR/judge_logp_beir-${subset%%/*}.log

# Judge with expected rating
python -m autollmrerank.wrapper \
    --config=$HOME/APRIL/src/autollmrerank/configs/judge.yaml \
    --data.batch_size=128 \
    --data.dataset_name=beir/${subset} \
    --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    --data.output_run=runs/Dev/judge-exp-rating.txt \
    --llm.use_logits=true \
    --llm.model_name_or_path=$MODEL \
    --system_message="You are JudgeLLM, an intelligent assistant that can judge a passage based on its relevancy to the query" \
    --result_parser_name=expected_rating > $LOGDIR/judge_expr_beir-${subset%%/*}.log
