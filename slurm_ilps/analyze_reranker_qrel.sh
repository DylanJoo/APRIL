#!/bin/sh
#SBATCH --job-name=qrel-analysis
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:00
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

MODEL=Qwen/Qwen2.5-7B-Instruct
OUTDIR=qrel-analysis/results
mkdir -p $OUTDIR

## Use the same retrieval's reranked results as judge
for r in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
    judge_run=runs/${MODEL##*/}/run.${benchmark}.${r}-rerank-judge.${subset%%/*}.txt

    rm -r $OUTDIR/${benchmark}.${subset%%/*}.${r}-rerank-judge.jsonl
    for r2 in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
        eval_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r2}.${subset%%/*}.txt
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
            --output $OUTDIR/${benchmark}.${subset%%/*}.${r}-rerank-judge.jsonl
    done
done
