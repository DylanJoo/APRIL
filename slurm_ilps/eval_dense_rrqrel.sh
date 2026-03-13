#!/bin/sh
#SBATCH --job-name=qrel-analysis
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:00
#SBATCH --output=%x-%a.out
#SBATCH --error=%x-%a.err

source $HOME/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL

R=(
"bm25"
"splade-v3"
"nomicai-modernbert-embed"
"qwen3-embed-600m"
"colbert-small"
)
r=${R[$SLURM_ARRAY_TASK_ID]}
rerank4judge=point
rerank4judge=setmaxheaptopk
rerank4judge=judge
rerank4judge=judge_expr
rerank4judge=rankgpt

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

MODEL=Qwen/Qwen2.5-7B-Instruct
OUTDIR=${HOME}/APRIL/qrel-analysis/results

# for dataset in "${DATASETS[@]}"; do
#     benchmark=$(echo $dataset | cut -d'@' -f1)
#     subset=$(echo $dataset | cut -d'@' -f2)
#     judge_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${rerank4judge}.${subset%%/*}.txt
#
#     output_path=$OUTDIR/${r}-rerank-${rerank4judge}/raaj-${subset%%/*}.jsonl
#     mkdir -p $(dirname $output_path)
#     : > $output_path
#
#     for r2 in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
#         eval_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r2}.${subset%%/*}.txt
#         python qrel-analysis/eval_autoqrels.py \
#             --dataset_name ${benchmark}/${subset} \
#             --loader_type irds \
#             --judge_run $judge_run \
#             --evaluate_run $eval_run \
#             --strategies direct \
#             --strategies thresholding --threshold=3 \
#             --strategies rank --rank_cutoff=10 \
#             --strategies quantile --quantile_cutoff=0.75 \
#             --strategies largest_gap --gap_k 1 \
#             --strategies optimal_per_topic --min_relevance 2 \
#             --strategies optimal_global \
#             --exp ${r}-rerank-${rerank4judge}:${r2} >> $output_path
#
#         # second stage reranking
#         for rerank in point judge judge_expr rankgpt setmaxheaptopk;do
#             eval_run=runs/${MODEL##*/}/run.${benchmark}.${r2}-rerank-${rerank}.${subset%%/*}.txt
#             python qrel-analysis/eval_autoqrels.py \
#                 --dataset_name ${benchmark}/${subset} \
#                 --loader_type irds \
#                 --judge_run $judge_run \
#                 --evaluate_run $eval_run \
#                 --strategies direct \
#                 --strategies thresholding --threshold=3 \
#                 --strategies rank --rank_cutoff=10 \
#                 --strategies quantile --quantile_cutoff=0.75 \
#                 --strategies largest_gap --gap_k 1 \
#                 --strategies optimal_per_topic --min_relevance 2 \
#                 --strategies optimal_global \
#                 --exp ${r}-rerank-${rerank4judge}:${r2}-rerank-${rerank} >> $output_path
#         done
#     done
# done

for dataset in "${DATASETS[@]}"; do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    judge_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${rerank4judge}.${subset%%/*}.txt

    output_path=$OUTDIR/${r}-rerank-${rerank4judge}/raaj-${subset%%/*}.jsonl
    for r2 in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do

        # second stage reranking
        # for rerank in rankfirst rankzepyhr;do
        for rerank in rankzephyr;do
            eval_run=runs/supervised/run.${benchmark}.${r2}-rerank-${rerank}.${subset%%/*}.txt
            python qrel-analysis/eval_autoqrels.py \
                --dataset_name ${benchmark}/${subset} \
                --loader_type irds \
                --judge_run $judge_run \
                --evaluate_run $eval_run \
                --strategies direct \
                --strategies thresholding --threshold=3 \
                --strategies rank --rank_cutoff=10 \
                --strategies quantile --quantile_cutoff=0.75 \
                --strategies largest_gap --gap_k 1 \
                --strategies optimal_per_topic --min_relevance 2 \
                --strategies optimal_global \
                --exp ${r}-rerank-${rerank4judge}:${r2}-rerank-${rerank} >> $output_path
        done
    done
done
