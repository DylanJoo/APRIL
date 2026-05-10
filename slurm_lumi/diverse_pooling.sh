#!/bin/bash -l
#SBATCH --job-name=diverse-pool
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=project_465002438
#SBATCH --output=logs/%x.%a.out
#SBATCH --error=logs/%x.%a.err

cd $HOME/APRIL

MODEL=meta-llama/Llama-3.3-70B-Instruct

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

RUN_FILES=()
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    f=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt
    [ -f "$f" ] && echo Y || echo "MISSING: $f"
    RUN_FILES+=("$f")
    for rr in judge judge_expr point rankgpt setmaxheaptopk; do
        f=$HOME/APRIL/runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-${rr}.${subset%%/*}.txt
        [ -f "$f" ] && echo Y || echo "MISSING: $f"
        RUN_FILES+=("$f")
    done
    f=$HOME/APRIL/runs/supervised/run.${benchmark}.${r}-rerank-rankfirst.${subset%%/*}.txt
    [ -f "$f" ] && echo Y || echo "MISSING: $f"
    RUN_FILES+=("$f")
    f=$HOME/APRIL/runs/supervised/run.${benchmark}.${r}-rerank-rankzephyr.${subset%%/*}.txt
    [ -f "$f" ] && echo Y || echo "MISSING: $f"
    RUN_FILES+=("$f")
done
echo ${#RUN_FILES[@]}

output=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.pool-40-systems-top100.${subset%%/*}.txt
python3 qrel-analysis/diverse_pooling.py \
    --run_files "${RUN_FILES[@]}" \
    --topk 100 \
    --output $output
