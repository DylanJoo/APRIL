#!/bin/bash -l
#SBATCH --job-name=crossqrel-sparse
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%a.out

source ~/.bashrc
initconda
conda activate autollmrerank
cd $HOME/APRIL

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

RETRIEVALS=(bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small)
RERANKERS=(judge judge_expr point rankgpt setmaxheaptopk)
POOL=()
for r in "${RETRIEVALS[@]}"; do
for rr in "${RERANKERS[@]}"; do
POOL+=("$r-rerank-$rr")
done
done

## Retrieval + reranking as judgment
for r1 in "${RETRIEVALS[@]}";do
for r2 in judge judge_expr pointwise rankgpt setmaxheaptopk; do
for seed in $(seq 1 10); do
judge_run=${HOME}/APRIL/runs/Qwen2.5-7B-Instruct/sample-$seed/run.$benchmark.$r1-rerank-$r2.${subset%%/*}.txt

for evaluate_run in ${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.*.${subset%%/*}*;do
    mkdir -p $output_dir/${r}-rerank-${method}/
    python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --exp Qwen2.5-7B-Instruct \
        --judge_run $judge_run \
        --evaluate_run $evaluate_run \
        --strategies all \
        --output $output_dir/${r}-rerank-${method}/${subset%%/*}-sample-${seed}.jsonl
done
done
done
done
