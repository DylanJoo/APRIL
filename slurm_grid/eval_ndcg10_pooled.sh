# nDCG@10 of the pool-reranked judge runs (pool -> LLM rerank) against human qrels.
# These are the same runs output_autoqrel.sh turns into auto-qrels.

# ENV
source ${HOME}/.bashrc
initconda
conda activate autollmreranker

DATASETS=(
"msmarco-passage@trec-dl-2019/judged"
"msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)
MODEL_DIR=Llama-3.3-70B-Instruct
POOLS=(pool-55-systems-top10)
JUDGES=(point umbrela judge judge_expr setmaxheaptopk umbrela_setmaxheaptopk rankgpt umbrela_rankgpt)

for pool in ${POOLS[@]};do
for judge in ${JUDGES[@]};do
for dataset in ${DATASETS[@]};do
    benchmark=$(echo $dataset | cut -d'@' -f1)
    subset=$(echo $dataset | cut -d'@' -f2)
    name=${subset%%/*}
    run_path=${HOME}/APRIL/runs/${MODEL_DIR}/run.$benchmark.$pool-rerank-$judge.$name.txt

    if [ ! -s "$run_path" ]; then
        echo "${pool} | ${judge} | ${name} | MISSING"
        continue
    fi
    nDCG=$(python -m ir_measures $benchmark/$subset $run_path nDCG@10 | cut -f2)
    echo "${pool} | ${judge} | ${name} | $nDCG"
done
done
done
