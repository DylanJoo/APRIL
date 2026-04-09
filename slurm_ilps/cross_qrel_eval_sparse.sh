#!/bin/sh
#SBATCH --job-name=crossqrel-sparse
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%a.out

source ~/.bashrc
initconda
conda activate autollmrerank
cd $HOME/APRIL

judge_llm=Qwen2.5-7B-Instruct
output_dir=${HOME}/APRIL/qrel-analysis/sparse-7b
judge_llm=supervised
output_dir=${HOME}/APRIL/qrel-analysis/sparse-7b

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
benchmark=$(echo $dataset | cut -d'@' -f1)
subset=$(echo $dataset | cut -d'@' -f2)

## Pointwise reranker as judgment
for seed in $(seq 1 10); do
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for method in judge judge_exp pointwise rankgpt setmaxheaptopk; do
judge_run=${HOME}/APRIL/runs/$judge_llm/sample-$seed/run.$benchmark.$r-rerank-$method.${subset%%/*}.txt
for evaluate_run in ${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.*.${subset%%/*}*;do
    mkdir -p $output_dir/${r}-rerank-${method}/
    python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --judge_run $judge_run \
        --evaluate_run $evaluate_run \
        --strategies all \
        --output $output_dir/${r}-rerank-${method}/${subset%%/*}-sample-${seed}.jsonl
done
done
done
done
