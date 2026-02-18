#!/bin/sh
#SBATCH --job-name=rerank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-10%4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%a.out

source ~/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
seed=5
LOGDIR=bbier/result_$seed
mkdir -p $LOGDIR
OUTDIR=bbier/run_$seed
mkdir -p $OUTDIR

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

subset=${BEIR_DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=Qwen/Qwen2.5-7B-Instruct

# # POINTWISE BINARY PROBABILITY
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/point.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/point.beir-${subset%%/*}.txt \
#     --llm.model_name_or_path=$MODEL > $LOGDIR/point.beir-${subset%%/*}.log
#
# # JUDGE TEXT
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/judge.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/judge.beir-${subset%%/*}.txt \
#     --llm.model_name_or_path=$MODEL > $LOGDIR/judge.beir-${subset%%/*}.log
#
# # JUDGE+ (EXPECTED RATING)
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/judge.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/judge_expr.beir-${subset%%/*}.txt \
#     --llm.use_logits=true \
#     --llm.model_name_or_path=$MODEL \
#     --result_parser_name=expected_rating > $LOGDIR/judge_expr.beir-${subset%%/*}.log
#
# # SETWISE TOPK
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/setmaxheaptopk.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/setmaxheap.beir-${subset%%/*}.txt \
#     --llm.max_model_len=10240 \
#     --llm.model_name_or_path=$MODEL > $LOGDIR/setmaxheap.beir-${subset%%/*}.log
#
# # LISTWISE
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/rankgpt.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/rankgpt.beir-${subset%%/*}.txt \
#     --llm.max_model_len=20480  \
#     --llm.model_name_or_path=$MODEL > $LOGDIR/rankgpt.beir-${subset%%/*}.log

# LISTWISE (Supervised)
# python -m autollmrerank.wrapper_bootstrap \
#     --config=$HOME/APRIL/src/autollmrerank/configs/rankgpt.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
#     --data.output_run=$OUTDIR/rankzephyr.beir-${subset%%/*}.txt \
#     --llm.model_name_or_path="castorini/rank_zephyr_7b_v1_full" \
#     --llm.max_model_len=20480 \
#     --result_parser_name=list_generation > $LOGDIR/rankzephyr.beir-${subset%%/*}.log

# LISTWISE-RankFirst 
python -m autollmrerank.wrapper_bootstrap \
    --config=$HOME/APRIL/src/autollmrerank/configs/rankfirst.yaml \
    --bootstrapping=true \
    --bootstrapping_size=50 \
    --bootstrapping_seed=$seed \
    --data.dataset_name=beir/${subset} \
    --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \
    --data.output_run=$OUTDIR/rankfirst.beir-${subset%%/*}.txt \
    --llm.model_name_or_path="castorini/first_mistral" \
    --llm.max_model_len=20480 \
    --rerank_mode=RankFirst \
    --result_parser_name=distribution_logp > $LOGDIR/rankfirst.beir-${subset%%/*}.log

# PAIRWISE TOPK
# python -m autollmrerank.wrapper \
#     --config=$HOME/APRIL/src/autollmrerank/configs/pairtopk.yaml \
#     --bootstrapping=true \
#     --bootstrapping_size=50 \
#     --bootstrapping_seed=$seed \
#     --data.dataset_name=beir/${subset} \
#     --data.input_run=runs/run.beir.bm25.${subset%%/*}.txt \ 
#     --data.output_run=$OUTDIR/pairtopk.beir-${subset%%/*}.txt \
#     --llm.model_name_or_path=$MODEL > $LOGDIR/pairtopk.beir-${subset%%/*}.log
