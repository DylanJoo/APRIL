#!/bin/bash
# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

DATASETS=(
"beir@arguana"
"beir@climate-fever"
"beir@fever/test"
"beir@fiqa/test"
"beir@hotpotqa/test"
"beir@nq"
"beir@quora/test"
"beir@scifact/test"
)

MODEL_DIR=Qwen/Qwen2.5-7B-Instruct
SEEDS=$(seq 1 10)

# retrieval
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for dataset in ${DATASETS[@]}; do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    irds_tag="${benchmark}/${subset}"
    name=${subset%%/*}

    init_run=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.$name.txt
    run_path=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.$name.txt
    for seed in ${SEEDS[@]};do
        nDCG=$(python ${HOME}/APRIL/src/eval_sample.py --irds_tag $irds_tag --path $run_path --sampling_seed $seed --init_run $init_run)
        echo "${retrieval} | - | ${name} | $seed | $nDCG"
    done
done
done

# reranking (average over seeds)
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for rerank in point judge judge_expr setmaxheaptopk rankgpt; do
for dataset in ${DATASETS[@]}; do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    irds_tag="${benchmark}/${subset}"
    name=${subset%%/*}

    init_run=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.$name.txt
    for seed in ${SEEDS[@]};do
        run_path=${HOME}/APRIL/runs/${MODEL_DIR##*/}/sample-${seed}/run.$benchmark.$retrieval-rerank-$rerank.$name.txt
        nDCG=$(python ${HOME}/APRIL/src/eval_sample.py --irds_tag $irds_tag --path $run_path --sampling_seed $seed --init_run $init_run)
        echo "${retrieval} | ${rerank} | ${name} | ${seed} | $nDCG"
    done
done
done

# supervised reranking
for retrieval in bm25 splade-v3 nomicai-modernbert-embed qwen3-embed-600m colbert-small; do
for rerank in rankfirst rankzephyr; do
for dataset in ${DATASETS[@]}; do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    irds_tag="${benchmark}/${subset}"
    name=${subset%%/*}

    init_run=${HOME}/runs-and-qrels/runs/$benchmark/run.$benchmark.$retrieval.$name.txt
    for seed in ${SEEDS[@]};do
        run_path=${HOME}/APRIL/runs/supervised/sample-${seed}/run.$benchmark.$retrieval-rerank-$rerank.$name.txt
        nDCG=$(python ${HOME}/APRIL/src/eval_sample.py --irds_tag $irds_tag --path $run_path --sampling_seed $seed --init_run $init_run)
        echo "${retrieval} | ${rerank} | ${name} | ${seed} | $nDCG"
    done
done
done
done
