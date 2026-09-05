#!/bin/sh
#SBATCH --job-name=neuclir
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.out

source ${HOME}/.bashrc
initconda
conda activate autollmrerank

LOGDIR=log.vllm.new
mkdir -p $LOGDIR

dataset=neuclir

# Pointwise YES NO
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m autollmrerank.wrapper_dev \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=neuclir \
    --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
    --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=10000 \
    --llm.use_logits=true \
    --max_doc_length=1000 \
    --rerank_mode=Point \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/point_neuclir.log

# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
python -m autollmrerank.wrapper_dev \
    --data.dataset_name=$dataset \
    --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
    --data.loader_type=neuclir \
    --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
    --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=20480 \
    --max_doc_length=1000 \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_neuclir.log

# # RankFirst:dist_logp:castorini/first_mistral
# MODEL=castorini/first_mistral
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=20480 \
#     --llm.use_logits=true \
#     --max_doc_length=1000 \
#     --rerank_mode=RankFirst \
#     --num_runs=1 \
#     --window_size=20 --step_size=10 \
#     --dtype=float16 \
#     --use_alphabetical=true \
#     --result_parser_name=distribution_logp > $LOGDIR/rankfirst_neuclir.log
#
# # RankGPT:list_gen:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=20480 \
#     --max_doc_length=1000 \
#     --rerank_mode=RankGPT \
#     --num_runs=1 \
#     --window_size=20 --step_size=10 \
#     --dtype=float16 \
#     --result_parser_name=list_generation > $LOGDIR/rankgpt_neuclir.log
#
# # SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=10000 \
#     --llm.use_logits=true \
#     --max_doc_length=1000 \
#     --rerank_mode=SetTopK \
#     --num_runs=10 \
#     --window_size=5 --step_size=1 \
#     --dtype=float16 \
#     --use_alphabetical=true \
#     --result_parser_name=distribution_logp > $LOGDIR/settop10_neuclir.log
#
# # SetMaxHeapTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=10000 \
#     --llm.use_logits=true \
#     --max_doc_length=1000 \
#     --rerank_mode=SetMaxHeapTopK \
#     --num_runs=10 \
#     --window_size=5 --step_size=1 \
#     --dtype=float16 \
#     --use_alphabetical=true \
#     --result_parser_name=distribution_logp > $LOGDIR/setmaxheaptop10_neuclir.log
#
# # PairTopK:binary_prob:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=10000 \
#     --llm.use_logits=true \
#     --max_doc_length=1000 \
#     --rerank_mode=PairTopK \
#     --num_runs=10 \
#     --window_size=2 --step_size=1 \
#     --dtype=float16 \
#     --result_parser_name=binary_probability > $LOGDIR/pairtop10_neuclir.log
#
# # PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
# MODEL=Qwen/Qwen2.5-7B-Instruct
# python -m autollmrerank.wrapper_dev \
#     --data.dataset_name=$dataset \
#     --data.input_run=runs/run.bm25.${dataset%%/*}.txt \
#     --data.loader_type=neuclir \
#     --data.input_diversity_qrels=$CRUX_ROOT/crux-$dataset/qrels/neuclir24-test-request.qrel \
#     --data.input_ratings=$CRUX_ROOT/crux-$dataset/judge/ratings.human.jsonl \
#     --llm.model_name_or_path=$MODEL \
#     --llm.max_model_len=10000 \
#     --llm.use_logits=true \
#     --max_doc_length=1000 \
#     --rerank_mode=PairAll \
#     --score_aggregation=symsum \
#     --dtype=float16 \
#     --result_parser_name=binary_probability > $LOGDIR/pairall_neuclir.log
