# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 

DATASETS=(
# "msmarco-passage@trec-dl-2019/judged"
# "msmarco-passage@trec-dl-2020/judged"
"beir@dbpedia-entity/test"
"beir@nfcorpus/test"
"beir@scidocs"
"beir@trec-covid"
"beir@webis-touche2020/v2"
)

# run_path=${HOME}/APRIL/runs/run.beir.bm25.dataset.txt
MODEL_DIR=Qwen/Qwen2.5-7B-Instruct

# BM25 
for retrieval in bm25 qwen3-embed-600m;do
for rerank in point judge judge_expr setmaxheaptopk rankgpt;do
    run_path=${HOME}/APRIL/runs/${MODEL_DIR##*/}/run.beir.$retrieval-rerank-$rerank.dataset.txt
    for dataset in ${DATASETS[@]};do
        benchmark=$(echo $dataset | cut -d'@' -f1)
        subset=$(echo $dataset | cut -d'@' -f2)
        name=${subset%%/*}

        short_name=$(basename "$name" | cut -c1-3)
        nDCG=$(python -m ir_measures $benchmark/$subset ${run_path/dataset/$name} nDCG@10 | cut -f2) 
        # R=$(python -m ir_measures ${QRELS[$id]} ${run_path/dataset/$dataset} R@100 | cut -f2)
        # echo "${retrieval} | ${rerank} | ${short_name} | $nDCG | $R"
        echo "${retrieval} | ${rerank} | ${short_name} | $nDCG"
    done
done
done
