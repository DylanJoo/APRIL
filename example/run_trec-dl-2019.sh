#!/bin/sh
#SBATCH --job-name=dl19-all
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate autollmreranker

LOGS_DIR=log.vllm
# RankZephyr:list_gen:castorini/rank_zephyr_7b_v1_full
MODEL=castorini/rank_zephyr_7b_v1_full
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankzephyr_trec-dl-2019.log

# RankFirst:dist_logp:castorini/first_mistral
MODEL=castorini/first_mistral
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=RankFirst \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/rankfirst_trec-dl-2019.log

# RankGPT:list_gen:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=RankGPT \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=list_generation > $LOGDIR/rankgpt_trec-dl-2019.log

# Point:binary_prob:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --rerank_mode=Point \
    --num_runs=1 \
    --window_size=20 --step_size=10 \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/point_trec-dl-2019.log

# SetTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=SetTopK \
    --num_runs=10 \
    --window_size=2 --step_size=1 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/settop10_trec-dl-2019.log

# SetMaxHeapTopK:dist_logp:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=SetMaxHeapTopK \
    --num_runs=10 \
    --window_size=5 --step_size=1 \
    --dtype=float16 \
    --use_alphabetical=true \
    --result_parser_name=distribution_logp > $LOGDIR/setmaxheaptop10_trec-dl-2019.log

# PairTopK:binary_prob:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=PairTopK \
    --num_runs=10 \
    --window_size=2 --step_size=1 \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/pairtop10_trec-dl-2019.log

# PairAll:binary_prob:Qwen/Qwen2.5-7B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct
python -m reranking.wrapper \
    --data.ir_datasets_name=msmarco-passage/trec-dl-2019/judged \
    --data.input_run=runs/run.msmarco-passage.bm25.trec-dl-2019.txt \
    --llm.model_name_or_path=$MODEL \
    --llm.max_model_len=8196 \
    --llm.use_logits=true \
    --rerank_mode=PairAll \
    --score_aggregation=symsum \
    --dtype=float16 \
    --result_parser_name=binary_probability > $LOGDIR/pairall_trec-dl-2019.log
