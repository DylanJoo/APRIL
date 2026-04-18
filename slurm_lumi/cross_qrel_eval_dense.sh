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
for r1 in "${RETRIEVALS[@]}"; do
for r2 in judge judge_expr point rankgpt setmaxheaptopk;do
judge_run=${HOME}/datasets/all-runs/run.$benchmark.$r1-rerank-$r2.${subset%%/*}.txt
mkdir -p ${HOME}/APRIL/qrel-analysis/dense-70b/${r1}-rerank-${r2}/${subset%%/*}/

echo Judge: $r1-rerank-$r2
for evaluate_setting in "${POOL[@]}"; do
    srun singularity exec $SIF python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --exp Llama-3.3-70B-Instruct \
        --judge_run $judge_run \
        --evaluate_run ${HOME}/APRIL/runs/Llama-3.3-70B-Instruct/run.$benchmark.$evaluate_setting.${subset%%/*}.txt \
        --strategies all \
        --output ${HOME}/APRIL/qrel-analysis/dense-70b/${r1}-rerank-${r2}/${subset%%/*}/$evaluate_setting.jsonl
done

for eval_r1 in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small;do
for eval_r2 in rankfirst rankzephyr;do
    srun singularity exec $SIF python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --exp Llama-3.3-70B-Instruct \
        --judge_run $judge_run \
        --evaluate_run ${HOME}/APRIL/runs/supervised/run.$benchmark.$eval_r1-rerank-$eval_r2.${subset%%/*}.txt \
        --strategies all \
        --output ${HOME}/APRIL/qrel-analysis/dense-70b/${r1}-rerank-${r2}/${subset%%/*}/$eval_r1-rerank-$eval_r2.jsonl
done
done

done
done
