#!/bin/sh
#SBATCH --job-name=spv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6%4
#SBATCH --ntasks-per-node=1
#SBATCH --time=25:00:00
#SBATCH --output=%x-%a.out

source $HOME/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
LOGDIR=log.dense-judged
mkdir -p $LOGDIR
mkdir -p runs/${MODEL##*/}

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

for r in bm25 qwen3-embed-600m;do
for method in rankfirst rankzephyr; do
    inital_run=$HOME/runs-and-qrels/runs/${benchmark}/run.${benchmark}.${r}.${subset%%/*}.txt 
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=${inital_run} \
        --data.output_run=runs/supervised/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt > $LOGDIR/${method}.${benchmark}-${subset%%/*}.log
done
done
