#!/bin/bash -l
#SBATCH --job-name=crossqrel-dense
#SBATCH --partition=small
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=project_465002532
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

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
POOL+=("${r}")
for rr in "${RERANKERS[@]}"; do
POOL+=("$r-rerank-$rr")
done
done

## Retrieval + reranking as judgment
for r in "${RETRIEVALS[@]}"; do
for reranker in "${RERANKERS[@]}"; do
judge_run=${HOME}/APRIL/runs/Llama-3.3-70B-Instruct/run.$benchmark.$r-rerank-$reranker.${subset%%/*}.txt
mkdir -p ${HOME}/APRIL/qrel-analysis/dense/${r}-rerank-${reranker}/${subset%%/*}/
for evaluate_setting in "${POOL[@]}"; do
    srun singularity exec $SIF python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --judge_run $judge_run \
        --evaluate_run ${HOME}/APRIL/runs/pool-for-dense/run.$benchmark.$evaluate_setting.${subset%%/*}.txt \
        --strategies all \
        --output ${HOME}/APRIL/qrel-analysis/dense/${r}-rerank-${reranker}/${subset%%/*}/$evaluate_setting.jsonl
done
done
done
