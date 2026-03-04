#!/bin/sh
#SBATCH --job-name=eval-judge
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --array=0-6%4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=%x-%a.out

source $HOME/.bashrc
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
name=${subset%%/*}

MODEL=Qwen/Qwen2.5-7B-Instruct

for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    judge_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-judge.${name}.txt

    # Retrieval baseline against judge qrels
    echo "=== ${r} | - | ${name} ==="
    python qrel-analysis/eval_judge_qrel.py \
        --dataset_name=${benchmark}/${subset} \
        --loader_type=irds \
        --judge_run=${judge_run} \
        --evaluate_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${name}.txt \
        --strategies direct

    # Reranked runs against judge qrels
    for method in point judge judge_expr setmaxheaptopk rankgpt; do
        echo "=== ${r} | ${method} | ${name} ==="
        python qrel-analysis/eval_judge_qrel.py \
            --dataset_name=${benchmark}/${subset} \
            --loader_type=idrs \
            --judge_run=${judge_run} \
            --evaluate_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${method}.${name}.txt \
            --strategies direct
    done
done
