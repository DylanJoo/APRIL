#!/bin/sh
#SBATCH --job-name=dqrel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%a.out

source $HOME/.bashrc
initconda
conda activate autollmrerank

cd $HOME/APRIL
LOGDIR=log.dense-judged
mkdir -p $LOGDIR

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
for r in bm25;do
for method in setmaxheaptopk; do
    python -m autollmrerank.wrapper \
        --config=$HOME/APRIL/src/autollmrerank/configs/${method}.yaml \
        --data.dataset_name=${benchmark}/${subset} \
        --data.input_run=runs/run.${benchmark}.${r}.${subset%%/*}.txt \
        --data.output_run=runs/dense-judged/run.${benchmark}.${r}-rerank-${method}.${subset%%/*}.txt \
        --llm.model_name_or_path=$MODEL > $LOGDIR/${method}.${benchmark}-${subset%%/*}.log
done
done
