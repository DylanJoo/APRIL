#!/bin/sh
#SBATCH --job-name=toy
#SBATCH --partition v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.out

python -m reranking.wrapper \
    --config=../src/reranking/configs/default.yaml \
    --data.input_run=../runs/run.msmarco-passage.bm25.trec-dl-2019.txt
