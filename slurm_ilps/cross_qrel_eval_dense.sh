#!/bin/bash -l
#SBATCH --job-name=crossqrel-dense
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
for r1 in "${RETRIEVALS[@]}"; do
for r2 in judge judge_expr point rankgpt setmaxheaptopk;do
judge_run=${HOME}/APRIL/runs/Qwen2.5-7B-Instruct/run.$benchmark.$r1-rerank-$r2.${subset%%/*}.txt
mkdir -p ${HOME}/APRIL/qrel-analysis/dense-7b/${r1}-rerank-${r2}/${subset%%/*}/

echo Judge: $r1-rerank-$r2
for evaluate_setting in "${POOL[@]}"; do
    echo $evaluate_setting
    python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --exp Qwen2.5-7B-Instruct \
        --judge_run $judge_run \
        --evaluate_run ${HOME}/datasets/all-runs/Qwen2.5-7B-Instruct/run.$benchmark.$evaluate_setting.${subset%%/*}.txt \
        --strategies all \
        --output ${HOME}/APRIL/qrel-analysis/dense-7b/${r1}-rerank-${r2}/${subset%%/*}/$evaluate_setting.jsonl
done

for eval_r1 in "${RETRIEVALS[@]}";do
    echo $eval_r1
    python qrel-analysis/eval_autoqrels.py \
        --dataset_name ${dataset/@//} \
        --loader_type irds \
        --exp Qwen2.5-7B-Instruct \
        --judge_run $judge_run \
        --evaluate_run ${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$eval_r1.${subset%%/*}.txt \
        --strategies all \
        --output ${HOME}/APRIL/qrel-analysis/dense-7b/${r1}-rerank-${r2}/${subset%%/*}/$eval_r1.jsonl

    for eval_r2 in rankfirst rankzephyr;do
        python qrel-analysis/eval_autoqrels.py \
            --dataset_name ${dataset/@//} \
            --loader_type irds \
            --exp Qwen2.5-7B-Instruct \
            --judge_run $judge_run \
            --evaluate_run ${HOME}/datasets/all-runs/supervised/run.$benchmark.$eval_r1-rerank-$eval_r2.${subset%%/*}.txt \
            --strategies all \
            --output ${HOME}/APRIL/qrel-analysis/dense-7b/${r1}-rerank-${r2}/${subset%%/*}/$eval_r1-rerank-$eval_r2.jsonl
    done
done
done
done
